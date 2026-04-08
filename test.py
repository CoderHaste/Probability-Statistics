import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import timm
import matplotlib.pyplot as plt
import cv2
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import io

# -----------------------
# CONFIG
# -----------------------
st.set_page_config(
    page_title="NeuroScan AI",
    layout="centered",
    page_icon="🧠"
)

# -----------------------
# CSS
# -----------------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #020617, #0f172a, #1e293b);
    color: white;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    background: linear-gradient(90deg, #38bdf8, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.card {
    background: rgba(255,255,255,0.05);
    padding: 25px;
    border-radius: 20px;
    backdrop-filter: blur(12px);
    box-shadow: 0px 0px 30px rgba(56,189,248,0.2);
    margin-top: 20px;
    border: 1px solid rgba(255,255,255,0.1);
}

img {
    border-radius: 12px;
    border: 2px solid rgba(56,189,248,0.3);
}

.good { color: #22c55e; font-size: 26px; text-align:center; }
.bad { color: #ef4444; font-size: 26px; text-align:center; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🧠 NeuroScan AI</div>', unsafe_allow_html=True)

# -----------------------
# CLASSES
# -----------------------
classes = ['glioma', 'meningioma', 'notumor', 'pituitary']

descriptions = {
    "glioma": "Aggressive tumor in glial cells.",
    "meningioma": "Usually benign tumor from meninges.",
    "pituitary": "Tumor affecting hormone regulation.",
    "notumor": "No tumor detected."
}

# -----------------------
# REPORT GENERATOR
# -----------------------
def generate_report(pred_class, confidence, description):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()

    content = []
    content.append(Paragraph("<b>NeuroScan AI - Diagnostic Report</b>", styles['Title']))
    content.append(Spacer(1, 20))
    content.append(Paragraph(f"<b>Prediction:</b> {pred_class.upper()}", styles['Normal']))
    content.append(Paragraph(f"<b>Confidence:</b> {confidence:.2f}%", styles['Normal']))
    content.append(Spacer(1, 10))
    content.append(Paragraph(f"<b>Analysis:</b> {description}", styles['Normal']))
    content.append(Spacer(1, 20))
    content.append(Paragraph("Note: This is AI-assisted and not a medical diagnosis.", styles['Italic']))

    doc.build(content)
    buffer.seek(0)
    return buffer

# -----------------------
# LOAD MODEL
# -----------------------
@st.cache_resource
def load_model():
    base_model = timm.create_model("xception", pretrained=False, num_classes=0)

    model = nn.Sequential(
        base_model,
        nn.Flatten(),
        nn.Dropout(0.3),
        nn.Linear(base_model.num_features, 128),
        nn.ReLU(),
        nn.Dropout(0.25),
        nn.Linear(128, 4)
    )

    model.load_state_dict(torch.load("brain_tumor_model.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# -----------------------
# FIND LAST CONV LAYER
# -----------------------
def get_last_conv_layer(model):
    for layer in reversed(list(model.modules())):
        if isinstance(layer, torch.nn.Conv2d):
            return layer

# -----------------------
# GRAD-CAM
# -----------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.forward_hook)
        target_layer.register_full_backward_hook(self.backward_hook)

    def forward_hook(self, module, inp, out):
        self.activations = out

    def backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def generate(self, x, class_idx):
        self.model.zero_grad()

        out = self.model(x)
        loss = out[0, class_idx]
        loss.backward()

        grads = self.gradients[0].detach().cpu().numpy()
        acts = self.activations[0].detach().cpu().numpy()

        weights = np.mean(grads, axis=(1,2))
        cam = np.zeros(acts.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * acts[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (299,299))
        cam = (cam - cam.min()) / (cam.max() + 1e-8)

        return cam

target_layer = get_last_conv_layer(model)
gradcam = GradCAM(model, target_layer)

# -----------------------
# TRANSFORM
# -----------------------
transform = transforms.Compose([
    transforms.Resize((299,299)),
    transforms.ToTensor()
])

# -----------------------
# UPLOAD
# -----------------------
uploaded_file = st.file_uploader("📤 Upload MRI Scan", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((512,512), Image.LANCZOS)

    # centered image
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image(image, caption="MRI Scan", width=350)

    show_heatmap = st.checkbox("🔥 Show AI Attention Map", value=True)

    if st.button("🚀 Analyze Scan"):

        with st.spinner("🧠 Running AI Analysis..."):

            img_tensor = transform(image).unsqueeze(0)

            # prediction (no grad needed)
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1).detach().numpy()[0]

            pred_idx = np.argmax(probs)
            pred_class = classes[pred_idx]
            confidence = probs[pred_idx] * 100

            # Grad-CAM
            cam = gradcam.generate(img_tensor, pred_idx)

        # -----------------------
        # RESULT
        # -----------------------
        st.markdown('<div class="card">', unsafe_allow_html=True)

        if pred_class == "notumor":
            st.markdown('<div class="good">🟢 No Tumor Detected</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bad">🔴 {pred_class.upper()}</div>', unsafe_allow_html=True)

        st.write(f"**Confidence:** {confidence:.2f}%")
        st.write(f"**Insight:** {descriptions[pred_class]}")

        st.markdown('</div>', unsafe_allow_html=True)

        # -----------------------
        # DOWNLOAD REPORT
        # -----------------------
        report = generate_report(pred_class, confidence, descriptions[pred_class])

        st.download_button(
            "📄 Download Report",
            data=report,
            file_name="NeuroScan_Report.pdf",
            mime="application/pdf"
        )

        # -----------------------
        # CHART
        # -----------------------
        st.markdown('<div class="card">', unsafe_allow_html=True)

        fig, ax = plt.subplots()
        ax.barh(classes, probs)
        ax.set_title("Prediction Breakdown")

        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

        # -----------------------
        # HEATMAP
        # -----------------------
        if show_heatmap:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("🔥 AI Attention Map")

            img_np = np.array(image.resize((299,299)))
            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            heatmap = cv2.GaussianBlur(heatmap, (15,15), 0)

            overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

            col1, col2 = st.columns(2)

            with col1:
                st.image(img_np, caption="Original", width=280)

            with col2:
                st.image(overlay, caption="AI Focus", width=280)

            st.markdown('</div>', unsafe_allow_html=True)

# -----------------------
# SIDEBAR
# -----------------------
st.sidebar.title("⚙️ Model Info")
st.sidebar.write("""
- Xception (timm)
- Input: 299x299
- 4 Classes
""")