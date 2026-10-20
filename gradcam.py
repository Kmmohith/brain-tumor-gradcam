"""Grad-CAM utilities for the VGG16 brain tumor classifier."""
import numpy as np
import tensorflow as tf
import keras
from PIL import Image
import matplotlib

IMG_SIZE = (160, 160)  # must match training
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]  # must match training order
DEFAULT_CONV_LAYER = "block5_conv3"  # last conv layer of VGG16


def preprocess(pil_img: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    """Replicates training preprocessing.

    flow_from_directory loads RGB, resizes with 'nearest' interpolation (Keras default),
    then applies vgg16.preprocess_input. Returns (model_input[1,H,W,3], display_rgb[H,W,3] uint8).
    """
    img = pil_img.convert("RGB").resize(IMG_SIZE, Image.NEAREST)
    rgb = np.asarray(img, dtype=np.float32)
    x = keras.applications.vgg16.preprocess_input(rgb.copy())
    return x[None, ...], rgb.astype(np.uint8)


def find_last_conv_layer(model) -> str:
    """Falls back to the last Conv2D layer if block5_conv3 is not found."""
    try:
        model.get_layer(DEFAULT_CONV_LAYER)
        return DEFAULT_CONV_LAYER
    except ValueError:
        for layer in reversed(model.layers):
            if isinstance(layer, keras.layers.Conv2D):
                return layer.name
    raise ValueError("No Conv2D layer found in model.")


def build_grad_model(model, conv_layer_name: str | None = None):
    conv_layer_name = conv_layer_name or find_last_conv_layer(model)
    conv_out = model.get_layer(conv_layer_name).output
    return keras.Model(model.inputs, [conv_out, model.output])


def make_gradcam_heatmap(grad_model, x: np.ndarray, class_index: int | None = None):
    """Returns (heatmap[h,w] in 0..1, probabilities[num_classes], class_index used)."""
    x = tf.convert_to_tensor(x)
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x, training=False)
        if class_index is None:
            class_index = int(tf.argmax(preds[0]))
        score = preds[:, class_index]
    grads = tape.gradient(score, conv_out)                  # [1,h,w,C]
    weights = tf.reduce_mean(grads, axis=(1, 2))            # [1,C] channel importance
    cam = tf.reduce_sum(conv_out * weights[:, None, None, :], axis=-1)[0]  # [h,w]
    cam = tf.nn.relu(cam).numpy()
    cam = cam / (cam.max() + 1e-8)
    return cam, preds[0].numpy(), class_index


def overlay_heatmap(rgb_uint8: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4,
                    out_size: tuple[int, int] | None = None) -> np.ndarray:
    """Blend a jet-colored heatmap onto the image. Optionally upscale for display."""
    if out_size is not None:
        rgb_uint8 = np.asarray(Image.fromarray(rgb_uint8).resize(out_size, Image.BICUBIC))
    h, w = rgb_uint8.shape[:2]
    hm = Image.fromarray((heatmap * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)
    colored = matplotlib.colormaps["jet"](np.asarray(hm) / 255.0)[..., :3]
    blended = (1 - alpha) * rgb_uint8.astype(np.float32) + alpha * colored * 255
    return np.clip(blended, 0, 255).astype(np.uint8)
