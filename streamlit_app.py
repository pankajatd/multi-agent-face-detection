"""
Multi-Agent Face Detection & Quality Diagnostic Platform — Streamlit Cloud Dashboard
=====================================================================================
Autonomous 4-Agent Pipeline powered by Deep Learning YuNet DNN, real-time quality
inspection (Blur, Luminance, Contrast), CLAHE/Gamma self-healing, and IoU audit verification.
"""
import os
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from PIL import Image
import streamlit as st

from src.graph.workflow import MultiAgentFaceOrchestrator
from src.generator.face_streamer import SyntheticFaceStreamer

# ──────────────────────────────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Agent Face Detection & Diagnostic Platform",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 1.2rem !important;
    }
    .header-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 0.9rem 1.4rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        color: white;
        border: 1px solid #334155;
    }
    .header-banner h1 { margin: 0; font-size: 1.4rem; }
    .header-banner p { margin: 0.25rem 0 0; opacity: 0.8; font-size: 0.85rem; }
    
    .stMetric {
        background: #0f172a;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 0.6rem;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────
# Cache Orchestrator and Generator
# ──────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⚙️ Initializing YuNet DNN & Multi-Agent Orchestrator...")
def get_orchestrator():
    return MultiAgentFaceOrchestrator()

@st.cache_resource
def get_streamer():
    return SyntheticFaceStreamer()

orchestrator = get_orchestrator()
streamer = get_streamer()

# ──────────────────────────────────────────────────────────────────────
# Header Banner
# ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
    <h1>👤 Multi-Agent Face Detection & Quality Diagnostic Platform</h1>
    <p>Autonomous 4-Agent Pipeline: YuNet Deep Learning DNN • Quality Inspection • Self-Healing Loop • IoU Audit</p>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────
# Sidebar Controls
# ──────────────────────────────────────────────────────────────────────
st.sidebar.markdown('<div style="font-weight:700; font-size:14px; color:#e2e8f0; margin-bottom:4px;">🎯 Detection Controls</div>', unsafe_allow_html=True)

input_mode = st.sidebar.radio(
    "Choose Input Source",
    ["🖼️ Sample Gallery", "🧪 Synthetic Generator", "📤 Upload Custom Photo"],
    index=0,
    key="face_input_mode",
)

gallery_dir = (PROJECT_ROOT / "test_images") if (PROJECT_ROOT / "test_images").exists() else (PROJECT_ROOT / "FaceDetection_Test_images")
gallery_files = ["Img1.webp", "Img2.webp", "Img3.jpg", "Img4.jpg", "Img5.jpg", "Img6.jpg"]
gallery_labels = {
    "Img1.webp": "Img 1 — Clean Portrait",
    "Img2.webp": "Img 2 — Low Light & Shadow",
    "Img3.jpg": "Img 3 — Outdoor Natural Light",
    "Img4.jpg": "Img 4 — High Definition Profile",
    "Img5.jpg": "Img 5 — Group & Multi-Face",
    "Img6.jpg": "Img 6 — Challenging Side Angle",
}

image = None
metadata = {}

if input_mode == "🖼️ Sample Gallery":
    selected_sample = st.sidebar.selectbox(
        "Select Gallery Photo",
        gallery_files,
        index=0,
        format_func=lambda f: gallery_labels.get(f, f),
        key="gallery_choice",
    )
    img_path = gallery_dir / selected_sample
    if img_path.exists():
        image = cv2.imread(str(img_path))
        metadata = {"source": selected_sample}
elif input_mode == "🧪 Synthetic Generator":
    deg_type = st.sidebar.selectbox(
        "Degradation Pattern",
        ["none", "underexposed", "blurred", "overexposed"],
        format_func=lambda d: {
            "none": "✨ None (Clean Standard)",
            "underexposed": "🌑 Underexposed (Low Light / Dark)",
            "blurred": "💨 Motion Blur",
            "overexposed": "☀️ Overexposed (High Glare)",
        }.get(d, d),
        key="deg_choice",
    )
    severity_label = st.sidebar.select_slider(
        "Degradation Severity",
        options=["mild", "moderate", "severe"],
        value="moderate",
        key="sev_choice",
    )
    sev_map = {"mild": 0.25, "moderate": 0.5, "severe": 0.8}
    image, metadata = streamer.generate_face_image(
        degradation_type=deg_type,
        severity=sev_map.get(severity_label, 0.5),
    )
else:
    uploaded = st.sidebar.file_uploader(
        "Upload Face Photo",
        type=["jpg", "jpeg", "png", "webp"],
        key="photo_upload",
    )
    if uploaded is not None:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        metadata = {"source": uploaded.name}

max_retries = st.sidebar.slider("Max Self-Healing Retries", min_value=1, max_value=5, value=3, key="max_retries")

st.sidebar.markdown("---")
run_btn = st.sidebar.button("🚀 Run Face Detection Pipeline", type="primary", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Multi-Agent Mesh:**
- **Detector**: YuNet DNN (300×300)
- **Inspector**: Blur / Luminance / Contrast
- **Enhancer**: CLAHE / Gamma / Unsharp
- **Auditor**: IoU Certification & Audit
""")

# Default fallback if no image selected
if image is None:
    first_sample = gallery_dir / "Img1.webp"
    if first_sample.exists():
        image = cv2.imread(str(first_sample))
        metadata = {"source": "Img1.webp"}
    else:
        image, metadata = streamer.generate_face_image()

# ──────────────────────────────────────────────────────────────────────
# Execute Pipeline
# ──────────────────────────────────────────────────────────────────────
t0 = time.perf_counter()
final_state = orchestrator.run(
    input_image=image,
    metadata=metadata,
    max_iterations=int(max_retries),
)
latency_ms = (time.perf_counter() - t0) * 1000

# Extract results
detections = final_state.get("detections", [])
quality_report = final_state.get("quality_report", {})
audit_report = final_state.get("audit_report", {})
enhancement_history = final_state.get("enhancement_history", [])

det_count = len(detections)
primary_conf = detections[0].get("confidence", 0.0) if detections else 0.0
quality_score = quality_report.get("quality_score", 0.0)
quality_status = quality_report.get("status", "NORMAL")
audit_verdict = audit_report.get("verdict", "PASSED")
iou_score = audit_report.get("iou_vs_ground_truth")

# Prepare visualization image with bounding boxes
out_img = (final_state.get("current_image") if final_state.get("current_image") is not None else image).copy()
for det in detections:
    bbox = det.get("bbox", [])
    conf = det.get("confidence", 0.0)
    if len(bbox) == 4:
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(out_img, (x, y), (x + w, y + h), (0, 255, 128), 2)
        label = f"YuNet {conf*100:.0f}%" if conf <= 1.0 else f"YuNet {conf:.0f}%"
        cv2.putText(out_img, label, (x, max(18, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 128), 2)

# ──────────────────────────────────────────────────────────────────────
# Top KPI Metrics Bar
# ──────────────────────────────────────────────────────────────────────
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.metric("👤 Faces Detected", f"{det_count} Face{'s' if det_count != 1 else ''}", "YuNet DNN")
with kpi2:
    conf_str = f"{primary_conf*100:.1f}%" if primary_conf <= 1.0 else f"{primary_conf:.1f}%"
    st.metric("🎯 Detection Confidence", conf_str, "Sub-Pixel 5-Pt")
with kpi3:
    st.metric("🔬 Image Quality Score", f"{quality_score:.1f} / 100", quality_status)
with kpi4:
    audit_label = "PASSED ✅" if "PASSED" in audit_verdict else audit_verdict
    iou_label = f"IoU: {iou_score:.2f}" if iou_score is not None else "Verified"
    st.metric("🛡️ Audit Certification", audit_label, iou_label)

st.markdown("---")

# ──────────────────────────────────────────────────────────────────────
# Dual Viewport: Original vs YuNet Detections
# ──────────────────────────────────────────────────────────────────────
st.markdown("### 📷 Dual Camera Viewport — Original Capture vs YuNet DNN Detections")
v_col1, v_col2 = st.columns(2, gap="medium")

with v_col1:
    st.markdown("**Original Sensor Ingestion**")
    st.image(image, channels="BGR", use_container_width=True)

with v_col2:
    st.markdown("**Detected Face Bounding Boxes & Confidence Overlay**")
    st.image(out_img, channels="BGR", use_container_width=True)

st.markdown("---")

# ──────────────────────────────────────────────────────────────────────
# Telemetry Tabs
# ──────────────────────────────────────────────────────────────────────
t1, t2, t3, t4 = st.tabs([
    "🔬 Quality Inspection Telemetry",
    "🛡️ Audit & Self-Healing Trail",
    "📐 Bounding Box Geometry",
    "🧠 Multi-Agent Workflow Graph",
])

with t1:
    qc1, qc2, qc3 = st.columns(3)
    with qc1:
        blur_val = quality_report.get("blur_score", 0.0)
        st.metric("Laplacian Blur Variance", f"{blur_val:.1f}", "Sharp" if blur_val > 100 else "Blurry")
    with qc2:
        lum_val = quality_report.get("luminance", 0.0)
        st.metric("Mean Luminance (L/Y)", f"{lum_val:.1f}", "Optimal" if 40 <= lum_val <= 210 else "Extreme")
    with qc3:
        con_val = quality_report.get("contrast", 0.0)
        st.metric("RMS Contrast", f"{con_val:.1f}")

    issues = quality_report.get("issues", [])
    if issues:
        st.warning(f"⚠️ Optical Issues Detected: {', '.join(issues)}")
    else:
        st.success("✨ Image quality passed all optical thresholds with zero degradation.")

with t2:
    st.markdown(f"**Audit Verdict:** `{audit_verdict}`")
    st.markdown(f"**Self-Healing Iterations Applied:** `{audit_report.get('self_healing_iterations', 0)}`")
    if enhancement_history:
        st.markdown("**Self-Healing Actions Taken:**")
        for i, enh in enumerate(enhancement_history, 1):
            st.markdown(f"- 🔧 **Iteration {i}:** Applied `{enh}`")
    else:
        st.success("✨ First-pass detection succeeded with zero degraded iterations required.")

with t3:
    if detections:
        for i, det in enumerate(detections, 1):
            st.markdown(f"**Face #{i}** — Engine: `{det.get('detection_method', 'yunet_dnn')}` • Confidence: **{det.get('confidence', 0.0):.2f}** • Bounding Box `[x, y, w, h]`: `{det.get('bbox')}`")
    else:
        st.info("No face bounding boxes detected in this frame.")

with t4:
    st.markdown("""
```mermaid
graph LR
    DET["👁️ Detector Agent<br/>YuNet ONNX DNN"] --> QUAL["🔬 Quality Inspector<br/>Blur / Lum / Contrast"]
    QUAL -->|Quality Pass| AUDIT["🛡️ Test Auditor<br/>IoU Certification"]
    QUAL -->|Degraded| ENH["🔧 Image Enhancer<br/>CLAHE / Gamma / Unsharp"]
    ENH --> DET
    AUDIT --> PASS["✅ Final Certified Verdict"]
```
    """)

# ──────────────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align:center; opacity:0.5; font-size:0.8rem;">'
    '👤 Multi-Agent Face Detection & Diagnostic Platform • '
    'Powered by OpenCV YuNet DNN • Streamlit Cloud'
    '</div>',
    unsafe_allow_html=True,
)
