"""
Edge AI Aircraft Defect Inspection - Demo App
-----------------------------------------------
Run with: streamlit run app.py

HOW TO PLUG IN YOUR TRAINED MODEL (once training finishes):
1. Download your best.pt weights from Roboflow (or export from Ultralytics)
2. Put the file in the same folder as this script, e.g. "best.pt"
3. Set USE_REAL_MODEL = True below
4. Set MODEL_PATH = "best.pt"
5. pip install ultralytics
That's it - the rest of the app already expects YOLO-style outputs.
"""

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import random
import time
import io

# ============================================================
# CONFIG - flip this switch once your model is trained
# ============================================================
USE_REAL_MODEL = True         # <-- set True after training finishes
MODEL_PATH = "best.pt"          # <-- path to your trained weights

# CLASS_NAMES = ["crack", "dent", "corrosion", "missing_fastener", "paint_damage", "fluid_leak"]

# # Severity rules: map defect type -> base severity
# SEVERITY_MAP = {
#     "crack": "high",
#     "missing_fastener": "high",
#     "fluid_leak": "high",
#     "corrosion": "medium",
#     "dent": "medium",
#     "paint_damage": "low",
# }

CLASS_NAMES = ["Dent", "Fastener Damage", "Rupture"]

SEVERITY_MAP = {
    "Rupture": "high",
    "Fastener Damage": "high",
    "Dent": "medium",
}

SEVERITY_COLOR = {
    "high": "#e63946",
    "medium": "#f4a261",
    "low": "#2a9d8f",
}

SEVERITY_LABEL = {
    "high": "\U0001F534 High severity - flag for immediate inspector review",
    "medium": "\U0001F7E1 Medium severity - flag for inspector review",
    "low": "\U0001F7E2 Low severity - log for routine maintenance",
}

# ============================================================
# MODEL LOADING
# ============================================================
@st.cache_resource
def load_model():
    if USE_REAL_MODEL:
        from ultralytics import YOLO
        return YOLO(MODEL_PATH)
    return None


def run_dummy_inference(image: Image.Image):
    """Generates fake but plausible detections so the UI/demo works
    before the real model is trained. Replace calls to this function
    with run_real_inference() once USE_REAL_MODEL = True."""
    w, h = image.size
    n_detections = random.randint(1, 3)
    detections = []
    for _ in range(n_detections):
        cls = random.choice(CLASS_NAMES)
        bw, bh = random.randint(int(w * 0.1), int(w * 0.3)), random.randint(int(h * 0.1), int(h * 0.3))
        x1 = random.randint(0, max(1, w - bw))
        y1 = random.randint(0, max(1, h - bh))
        x2, y2 = x1 + bw, y1 + bh
        conf = round(random.uniform(0.55, 0.97), 2)
        detections.append({"class": cls, "conf": conf, "box": (x1, y1, x2, y2)})
    return detections


def run_real_inference(model, image: Image.Image):
    """Runs the actual trained YOLO model and converts results
    into the same dict format used by run_dummy_inference()."""
    results = model.predict(image, verbose=False)[0]
    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        cls_name = results.names[cls_id]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
        detections.append({"class": cls_name, "conf": round(conf, 2), "box": (x1, y1, x2, y2)})
    return detections


def draw_detections(image: Image.Image, detections):
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    for d in detections:
        sev = SEVERITY_MAP.get(d["class"], "medium")
        color = SEVERITY_COLOR[sev]
        x1, y1, x2, y2 = d["box"]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        label = f'{d["class"]} {d["conf"]*100:.0f}%'
        draw.rectangle([x1, y1 - 20, x1 + len(label) * 7 + 8, y1], fill=color)
        draw.text((x1 + 4, y1 - 18), label, fill="white")
    return img


# ============================================================
# UI
# ============================================================
st.set_page_config(page_title="Edge AI Aircraft Inspection", layout="wide")

st.title("\u2708\ufe0f Edge AI Aircraft Surface Defect Inspection")
st.caption(
    "Real-time defect detection and severity flagging for aircraft MRO, "
    "optimized for edge deployment (YOLO26-Nano)."
)

if not USE_REAL_MODEL:
    st.info(
        "Demo mode: showing placeholder detections. Flip USE_REAL_MODEL = True "
        "in app.py once your trained weights are ready.",
        icon="\u2139\ufe0f",
    )

model = load_model()

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("1. Upload an inspection image")
    uploaded_file = st.file_uploader("Upload aircraft surface image", type=["jpg", "jpeg", "png"])
    use_sample = st.button("Or use a sample image")

image = None
if uploaded_file is not None:
    image = Image.open(uploaded_file)
elif use_sample:
    # simple placeholder gray image so the demo works with no upload
    image = Image.new("RGB", (640, 480), color=(120, 120, 130))

if image is not None:
    with col_left:
        st.image(image, caption="Input image", use_container_width=True)

    with st.spinner("Running edge inference..."):
        start = time.time()
        if USE_REAL_MODEL:
            detections = run_real_inference(model, image)
        else:
            time.sleep(0.6)  # simulate inference latency for demo feel
            detections = run_dummy_inference(image)
        latency_ms = (time.time() - start) * 1000

    annotated = draw_detections(image, detections)

    with col_right:
        st.subheader("2. Detection results")
        st.image(annotated, caption="Detected defects", use_container_width=True)
        st.caption(f"Inference time: {latency_ms:.0f} ms")

    st.subheader("3. Inspection report")
    if not detections:
        st.success("No defects detected.")
    else:
        for i, d in enumerate(detections, start=1):
            sev = SEVERITY_MAP.get(d["class"], "medium")
            with st.container(border=True):
                c1, c2, c3 = st.columns([1, 1, 2])
                c1.markdown(f"**Defect {i}:** {d['class'].replace('_', ' ').title()}")
                c2.markdown(f"**Confidence:** {d['conf']*100:.0f}%")
                c3.markdown(SEVERITY_LABEL[sev])

        # downloadable report
        report_lines = ["Aircraft Surface Inspection Report", "=" * 35]
        for i, d in enumerate(detections, start=1):
            sev = SEVERITY_MAP.get(d["class"], "medium")
            report_lines.append(
                f"{i}. {d['class']} | confidence: {d['conf']*100:.0f}% | severity: {sev}"
            )
        report_text = "\n".join(report_lines)
        st.download_button(
            "Download inspection report (.txt)",
            data=report_text,
            file_name="inspection_report.txt",
        )
else:
    st.info("Upload an image or click 'use a sample image' to see a detection demo.")

st.divider()
st.caption(
    "Architecture: YOLO26-Nano fine-tuned on aircraft surface defect data "
    "(crack, dent, corrosion, missing fastener, paint damage, fluid leak). "
    "Exportable to ONNX/TFLite for on-device inference on edge hardware "
    "(e.g. Jetson Nano/Orin, Raspberry Pi)."
)