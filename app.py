"""Brain tumor MRI classifier with Grad-CAM — Streamlit app."""
import json
import os
from pathlib import Path

import keras
import streamlit as st
from PIL import Image

from gradcam import (CLASS_NAMES, build_grad_model, make_gradcam_heatmap,
                     overlay_heatmap, preprocess)

st.set_page_config(page_title="Brain Tumor MRI + Grad-CAM", page_icon="🧠", layout="wide")

MODEL_PATH = Path(os.getenv("MODEL_PATH", "brain_tumor_vgg16_final.keras"))
META_PATH = Path("metadata.json")


@st.cache_resource(show_spinner="Loading model...")
def load():
    model = keras.models.load_model(MODEL_PATH, compile=False)
    return model, build_grad_model(model)


model, grad_model = load()
class_names = CLASS_NAMES
if META_PATH.exists():  # keep class order in sync with training
    class_names = json.loads(META_PATH.read_text())["class_names"]

st.title("🧠 Brain Tumor MRI Classifier with Grad-CAM")
st.warning(
    "Research/educational demo only. Not a medical device and not for diagnosis. "
    "Grad-CAM shows what influenced the model, not where a tumor is. "
    "Consult a qualified clinician."
)

with st.sidebar:
    st.header("Settings")
    target = st.selectbox("Explain which class?", ["Predicted class"] + class_names)
    alpha = st.slider("Heatmap opacity", 0.1, 0.9, 0.4, 0.05)

file = st.file_uploader("Upload an MRI image", type=["jpg", "jpeg", "png", "bmp", "webp"])

if file is not None:
    pil = Image.open(file)
    x, rgb = preprocess(pil)
    idx = None if target == "Predicted class" else class_names.index(target)
    heatmap, probs, used = make_gradcam_heatmap(grad_model, x, idx)
    overlay = overlay_heatmap(rgb, heatmap, alpha=alpha, out_size=(480, 480))

    pred = int(probs.argmax())
    c1, c2, c3 = st.columns(3)
    c1.subheader("Input")
    c1.image(pil.convert("RGB"), use_container_width=True)
    c2.subheader(f"Grad-CAM ({class_names[used]})")
    c2.image(overlay, use_container_width=True)
    c3.subheader("Prediction")
    c3.metric("Top class", class_names[pred], f"{probs[pred]:.1%} confidence")
    for name, p in sorted(zip(class_names, probs), key=lambda t: -t[1]):
        c3.progress(float(p), text=f"{name}: {p:.1%}")
else:
    st.info("Upload an image to get started.")
