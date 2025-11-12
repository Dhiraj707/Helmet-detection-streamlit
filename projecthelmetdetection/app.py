from services.helmet_det import HelmetDetectionPipeline
from services.audio import play_alert
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile

st.set_page_config(page_title="Helmet Detection", layout="centered")
st.title("Helmet Detection")

pipeline = HelmetDetectionPipeline(model_path="model/best.pt")

# ------------------ Image Upload ------------------
st.subheader("Detect from Image")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    annotated, detected_classes = pipeline.detect(frame)
    print("helooooooooooooooooooooooooooooo",detected_classes)

    if 'nohelmet' in detected_classes:
        play_alert()
        st.warning("No helmet detected!")

    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Detection Result", use_column_width=True)

# ------------------ Video Upload ------------------
st.subheader("Detect from Video File")
video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
if video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    cap = cv2.VideoCapture(tfile.name)

    stframe = st.empty()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        annotated, detected_classes = pipeline.detect(frame)
        

        if 'nohelmet' in detected_classes:
            play_alert()
            st.warning("No helmet detected!")

        stframe.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB")

    cap.release()

# ------------------ Webcam Detection ------------------
st.subheader("Live Webcam Detection")
run_webcam = st.checkbox("Start Webcam")
if run_webcam:
    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    while run_webcam and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        annotated, detected_classes = pipeline.detect(frame)

        if 'nohelmet' in detected_classes:
            play_alert()
            st.warning("No helmet detected!")

        stframe.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB")

    cap.release()