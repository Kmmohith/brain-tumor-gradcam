# ============================================================
# SECTION 11: GRAD-CAM VISUALISATION (append to the notebook)
# ============================================================
import numpy as np, matplotlib.pyplot as plt, matplotlib
import tensorflow as tf
from tensorflow.keras.models import Model

LAST_CONV = "block5_conv3"
grad_model = Model(final_model.inputs,
                   [final_model.get_layer(LAST_CONV).output, final_model.output])

def gradcam(x, class_index=None):
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x, training=False)
        if class_index is None:
            class_index = int(tf.argmax(preds[0]))
        score = preds[:, class_index]
    grads = tape.gradient(score, conv_out)
    weights = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.nn.relu(tf.reduce_sum(conv_out * weights[:, None, None, :], axis=-1)[0]).numpy()
    return cam / (cam.max() + 1e-8), preds[0].numpy(), class_index

def show_gradcam_grid(gen, n_per_class=2, alpha=0.4):
    """Show Grad-CAM for n test images per class. Uses preprocessed batches from `gen`."""
    from tensorflow.keras.applications.vgg16 import preprocess_input
    fig, axes = plt.subplots(NUM_CLASSES * n_per_class, 2, figsize=(7, 3.3 * NUM_CLASSES * n_per_class))
    row = 0
    for cls_idx, cls in enumerate(CLASS_NAMES):
        idxs = np.where(gen.classes == cls_idx)[0][:n_per_class]
        for i in idxs:
            x = gen[i // gen.batch_size][0][i % gen.batch_size][None]
            # undo VGG16 preprocessing for display (BGR mean-subtracted -> RGB uint8)
            disp = x[0].copy()
            disp += [103.939, 116.779, 123.68]; disp = disp[..., ::-1]
            disp = np.clip(disp, 0, 255).astype("uint8")
            cam, probs, pred = gradcam(x)
            hm = np.asarray(tf.image.resize(cam[..., None], IMG_SIZE))[..., 0]
            colored = matplotlib.colormaps["jet"](hm)[..., :3] * 255
            overlay = np.clip((1 - alpha) * disp + alpha * colored, 0, 255).astype("uint8")
            axes[row, 0].imshow(disp); axes[row, 0].set_title(f"True: {cls}")
            axes[row, 1].imshow(overlay)
            axes[row, 1].set_title(f"Pred: {CLASS_NAMES[pred]} ({probs[pred]:.1%})")
            for a in axes[row]: a.axis("off")
            row += 1
    plt.tight_layout(); plt.show()

test_gen.reset()
show_gradcam_grid(test_gen, n_per_class=2)
