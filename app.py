import streamlit as st
from PIL import Image
import numpy as np
import io

st.set_page_config(page_title="Object Detection & Captioning", layout="centered")

st.title("🖼️ Object Detection (YOLOv8) + Image Captioning (BLIP)")
st.write(
    "Upload an image to run YOLOv8 object detection and generate a natural "
    "language caption using BLIP. This app runs on CPU, so it is compatible "
    "with Streamlit Community Cloud."
)

# ---------------------------------------------------------------------------
# Cached model loaders
# Streamlit Cloud has no GPU, so everything is forced to run on CPU (device="cpu").
# @st.cache_resource makes sure the (large) models are loaded only once per session.
# ---------------------------------------------------------------------------

import os

# If you've fine-tuned your own model (e.g. via model.train() in Colab),
# drop the resulting weights file into this repo next to app.py, named
# "best.pt" (that's the default filename Ultralytics saves under
# runs/detect/train*/weights/best.pt). The app will automatically use it
# instead of the stock pretrained checkpoint - no code changes needed.
CUSTOM_WEIGHTS_PATH = "best.pt"
STOCK_WEIGHTS_PATH = "yolov8n.pt"


@st.cache_resource(show_spinner="Loading YOLOv8 model...")
def load_yolo_model():
    from ultralytics import YOLO

    if os.path.exists(CUSTOM_WEIGHTS_PATH):
        weights_path = CUSTOM_WEIGHTS_PATH
        st.sidebar.success(f"Using custom fine-tuned weights: {CUSTOM_WEIGHTS_PATH}")
    else:
        # yolov8n.pt (nano) is the smallest/fastest checkpoint - best choice
        # for the free tier's limited CPU + RAM. It auto-downloads on first run.
        weights_path = STOCK_WEIGHTS_PATH
        st.sidebar.info(f"No {CUSTOM_WEIGHTS_PATH} found - using stock {STOCK_WEIGHTS_PATH}")

    model = YOLO(weights_path)
    return model


@st.cache_resource(show_spinner="Loading BLIP captioning model...")
def load_blip_model():
    import torch
    from transformers import BlipProcessor, BlipForConditionalGeneration

    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    model.to("cpu")
    model.eval()
    return processor, model


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def run_detection(model, pil_image, conf, iou):
    # Ultralytics accepts a PIL image directly - no manual cv2/BGR handling needed.
    results = model.predict(source=pil_image, conf=conf, iou=iou, device="cpu", verbose=False)
    r = results[0]
    annotated_bgr = r.plot()                      # returns a BGR numpy array
    annotated_rgb = annotated_bgr[..., ::-1]       # convert BGR -> RGB for display
    annotated_img = Image.fromarray(annotated_rgb)

    detections = []
    for box in r.boxes:
        cls_id = int(box.cls[0])
        detections.append(
            {
                "label": model.names[cls_id],
                "confidence": float(box.conf[0]),
                "box_xyxy": [round(v, 1) for v in box.xyxy[0].tolist()],
            }
        )
    return annotated_img, detections


def run_caption(processor, model, pil_image):
    import torch

    inputs = processor(images=pil_image, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=20,
            num_beams=3,
            do_sample=False,
            repetition_penalty=1.4,
            length_penalty=1.2,
            early_stopping=True,
        )
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

st.sidebar.header("Settings")
conf_thresh = st.sidebar.slider("Detection confidence threshold", 0.1, 0.9, 0.6, 0.05)
iou_thresh = st.sidebar.slider("IoU (overlap) threshold", 0.1, 0.9, 0.6, 0.05)
run_caption_toggle = st.sidebar.checkbox("Generate caption (BLIP)", value=True)

# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload an image", type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    try:
        image_bytes = uploaded_file.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        st.error(f"Could not read the uploaded file as an image: {e}")
        st.stop()

    st.subheader("Original Image")
    st.image(pil_image, use_container_width=True)

    if st.button("Run Detection & Captioning", type="primary"):
        # ---- Object detection ----
        with st.spinner("Running object detection..."):
            try:
                yolo_model = load_yolo_model()
                annotated_img, detections = run_detection(
                    yolo_model, pil_image, conf_thresh, iou_thresh
                )
            except Exception as e:
                st.error(f"Detection failed: {e}")
                annotated_img, detections = None, []

        if annotated_img is not None:
            st.subheader("Detections")
            st.image(annotated_img, use_container_width=True)

            if detections:
                st.table(detections)
            else:
                st.info("No objects detected above the current confidence threshold.")

        # ---- Captioning ----
        if run_caption_toggle:
            with st.spinner("Generating caption..."):
                try:
                    processor, blip_model = load_blip_model()
                    caption = run_caption(processor, blip_model, pil_image)
                    st.subheader("Caption")
                    st.success(caption)
                except Exception as e:
                    st.error(f"Captioning failed: {e}")
else:
    st.info("👆 Upload an image to get started.")

st.caption(
    "Note: on Streamlit Community Cloud's free tier (limited CPU/RAM), the "
    "first run may take a while while the YOLOv8 and BLIP weights download "
    "and load into memory."
)
