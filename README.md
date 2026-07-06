# ✈️ Edge AI for Aircraft Surface Defect Detection

Real-time, edge-deployable computer vision system for detecting and classifying aircraft surface defects during Maintenance, Repair & Overhaul (MRO) inspections — built for **InnoVent-27** (AI at the Edge: Aerospace track).

## Problem

Aircraft MRO relies heavily on manual visual inspection to detect surface defects such as cracks, dents, corrosion, missing fasteners, paint damage, and fluid leaks. This process is slow, labor-intensive, and dependent on individual inspector experience — leading to inconsistent results and longer aircraft turnaround times.

## Solution

This project uses an edge-optimized object detection model to automatically detect, localize, and classify surface defects from inspection images in real time — acting as **decision support for inspectors**, not a replacement for certified human sign-off. Detected defects are flagged with a severity level (high / medium / low) to help prioritize inspector attention, and results are compiled into a downloadable inspection report.

> **Note:** This tool is designed to assist and speed up human inspection workflows. Final airworthiness decisions remain the responsibility of certified maintenance engineers.

## Features

- Real-time defect detection with bounding boxes and confidence scores
- Three defect classes:  `dent`, `rupture`, `fastener_damage`
- Automatic severity flagging (🔴 high / 🟡 medium / 🟢 low) to support inspection triage
- Downloadable inspection report
- Lightweight architecture designed for on-device / edge deployment (no cloud dependency required at inference time)

## Tech Stack

| Layer | Tool |
|---|---|
| Object detection model | YOLO26-Nano (Ultralytics) |
| Training / dataset management | Roboflow |
| Model export (edge deployment) | ONNX / TFLite |
| Demo application | Streamlit |
| Image processing | Pillow, OpenCV |
| Target edge hardware | NVIDIA Jetson Nano/Orin, Raspberry Pi + accelerator |

## Dataset

Trained on a public aircraft surface defect dataset (~9,352 images) sourced and annotated via Roboflow Universe, covering real defect categories relevant to aerospace MRO inspection.

- Train / Valid / Test split: 70% / 20% / 10%
- Image size: 640×640
- Preprocessing: auto-orient, resize

## Model Performance

_To be filled:_
- mAP@50: `0.7401408`
- Precision: `0.795374`
- Recall: `0.6705528`
- Inference latency (edge, quantized): `3939.58312 ms`

## How It Works

1. An inspection image is captured or uploaded
2. YOLO26-Nano detects and classifies defects in the image
3. Each detected defect is mapped to a severity level based on defect type
4. Results are displayed with bounding boxes, confidence scores, and severity flags
5. An inspection report is generated for the maintenance team

## Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run the demo app
```bash
streamlit run app.py
```

### Using your own trained weights
1. Place your trained `best.pt` file in the project root
2. In `app.py`, set:
   ```python
   USE_REAL_MODEL = True
   MODEL_PATH = "best.pt"
   ```
3. Re-run the app

## Project Structure
```
├── app.py              # Streamlit demo application
├── requirements.txt     # Python dependencies
├── best.pt              # Trained model weights (added after training)
└── README.md
```

## Roadmap / Future Work

- Expand dataset with additional real-world MRO imagery
- Deploy and benchmark on physical edge hardware (Jetson Nano/Orin)
- Add video/live-camera inference mode
- Integrate with existing MRO documentation/logging systems
- Add model confidence calibration and false-positive reduction

## Acknowledgements

Built for **InnoVent-27**, AI at the Edge track — Aerospace: Intelligent Inspection & Defect Detection.

Dataset sourced from public contributions on Roboflow Universe.
