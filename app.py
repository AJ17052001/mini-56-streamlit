import streamlit as st
from PIL import Image
import io
import os

st.set_page_config(page_title="Object Detection & Captioning", layout="centered")

# Debug (remove later if you want)
st.sidebar.write(f"Streamlit Version: {st.__version__}")

st.title("🖼️ Object Detection (YOLOv8) + Image Captioning (BLIP)")
st.write(
    "Upload an image to detect objects using YOLOv8 and generate a natural "
    "language caption using BLIP."
)

# -------------------------------------------------------------------
# Model paths
# -------------------------------------------------------------------

CUSTOM_WEIGHTS_PATH = "best.pt"
STOCK_WEIGHTS_PATH = "yolov8n.pt"

# -------------------------------------------------------------------
# Load YOLO
# -------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading YOLOv8 model...")
def load_yolo_model():
    from ultralytics import YOLO

    if os.path.exists(CUSTOM_WEIGHTS_PATH):
        weights = CUSTOM_WEIGHTS_PATH
        st.sidebar.success("Using custom trained model (best.pt)")
    else:
        weights = STOCK_WEIGHTS_PATH
        st.sidebar.info("Using pretrained yolov8n.pt")

    return YOLO(weights)

# -------------------------------------------------------------------
# Load BLIP
# -------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading BLIP model...")
def load_blip_model():
    import torch
    from transformers import (
        BlipProcessor,
        BlipForConditionalGeneration,
    )

    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    model.to("cpu")
    model.eval()

    return processor, model

# -------------------------------------------------------------------
# Detection
# -------------------------------------------------------------------

def run_detection(model, image, conf, iou):
    results = model.predict(
        source=image,
        conf=conf,
        iou=iou,
        device="cpu",
        verbose=False,
    )

    result = results[0]

    annotated = result.plot()
    annotated = annotated[..., ::-1]

    annotated_image = Image.fromarray(annotated)

    detections = []

    for box in result.boxes:
        cls = int(box.cls[0])

        detections.append(
            {
                "Object": model.names[cls],
                "Confidence": round(float(box.conf[0]), 3),
            }
        )

    return annotated_image, detections

# -------------------------------------------------------------------
# Captioning
# -------------------------------------------------------------------

def run_caption(processor, model, image):
    import torch

    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=20,
            num_beams=3,
            do_sample=False,
            repetition_penalty=1.4,
            length_penalty=1.2,
        )

    return processor.decode(
        output[0],
        skip_special_tokens=True,
    )

# -------------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------------

st.sidebar.header("Settings")

confidence = st.sidebar.slider(
    "Confidence",
    0.1,
    0.9,
    0.6,
    0.05,
)

iou = st.sidebar.slider(
    "IoU",
    0.1,
    0.9,
    0.6,
    0.05,
)

generate_caption = st.sidebar.checkbox(
    "Generate Caption",
    value=True,
)

# -------------------------------------------------------------------
# Upload Image
# -------------------------------------------------------------------

uploaded = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded:

    image = Image.open(io.BytesIO(uploaded.read())).convert("RGB")

    st.subheader("Original Image")
    st.image(
        image,
        caption="Uploaded Image",
        use_column_width=True,
    )

    if st.button("Run Detection & Captioning", type="primary"):

        # ---------------- Detection ----------------

        with st.spinner("Detecting objects..."):

            yolo = load_yolo_model()

            annotated_image, detections = run_detection(
                yolo,
                image,
                confidence,
                iou,
            )

        st.subheader("Detection Result")

        st.image(
            annotated_image,
            caption="YOLOv8 Output",
            use_column_width=True,
        )

        if detections:
            st.table(detections)
        else:
            st.warning("No objects detected.")

        # ---------------- Caption ----------------

        if generate_caption:

            with st.spinner("Generating caption..."):

                processor, blip = load_blip_model()

                caption = run_caption(
                    processor,
                    blip,
                    image,
                )

            st.subheader("Image Caption")
            st.success(caption)

else:
    st.info("Upload an image to begin.")

st.caption(
    "The first run may take some time because the models "
    "are downloaded and loaded into memory."
)
    
