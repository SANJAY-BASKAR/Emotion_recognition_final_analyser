import streamlit as st
import torch
import cv2
import time
import numpy as np
from collections import Counter, defaultdict
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image
from torchvision.models import convnext_tiny

# Load models
face_model = YOLO("yolov8n-face.pt")
emotion_model = convnext_tiny(num_classes=8)
emotion_model.load_state_dict(torch.load("emotion_model.pth", map_location=torch.device("cpu")))
emotion_model.eval()

emotion_labels = ['neutral', 'happy', 'sad', 'angry', 'surprised', 'fear', 'disgust', 'contempt']
negative_emotions = {'angry', 'disgust', 'fear', 'sad'}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def calculate_score(emotion_sequence):
    total = len(emotion_sequence)
    if total == 0:
        return 100
    negative_count = sum(1 for e in emotion_sequence if e in negative_emotions)
    transitions = sum(1 for i in range(1, total) if emotion_sequence[i] != emotion_sequence[i-1])

    max_streak = 0
    current_streak = 0
    for e in emotion_sequence:
        if e in negative_emotions:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0

    score = 100
    score -= (negative_count / total) * 40
    score -= (transitions / total) * 20
    score -= (max_streak / total) * 40

    return max(0, round(score, 2))

def predict_emotion(frame):
    results = face_model(frame)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    emotions = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            continue
        face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        face_tensor = transform(face_pil).unsqueeze(0)
        with torch.no_grad():
            output = emotion_model(face_tensor)
            pred = output.argmax(dim=1).item()
            emotions.append(emotion_labels[pred])
    return emotions

def majority_vote(emotions):
    if not emotions:
        return 'neutral'
    return Counter(emotions).most_common(1)[0][0]

st.set_page_config(page_title="Interview Emotion Scoring", layout="wide")
st.title("🎯 Emotion-based Interview Scoring Dashboard")
st.markdown("Smoothly track emotional performance during an interview.")

duration = st.slider("Set interview duration (seconds)", 5, 120, 30)
run_btn = st.button("Start Interview Scoring")

if run_btn:
    stframe = st.empty()
    progress_bar = st.progress(0)
    live_chart = st.empty()
    cap = cv2.VideoCapture(0)

    emotion_buckets = defaultdict(list)
    start_time = time.time()
    frame_count = 0
    window_size = 0.5  # in seconds

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time() - start_time
        bucket_idx = int(current_time / window_size)

        emotions = predict_emotion(frame)
        if emotions:
            emotion_buckets[bucket_idx].extend(emotions)
        else:
            emotion_buckets[bucket_idx].append('neutral')

        frame_count += 1
        most_common = majority_vote(emotion_buckets[bucket_idx])
        cv2.putText(frame, f"Emotion: {most_common}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        progress = min(1.0, current_time / duration)
        progress_bar.progress(progress)

        if current_time > duration:
            break

    cap.release()
    st.success(f"Completed! Processed {frame_count} frames in {duration} seconds.")

    final_emotion_sequence = [majority_vote(emotion_buckets[i]) for i in range(int(duration / window_size))]
    st.subheader("📊 Timeline of Dominant Emotions")
    st.write(final_emotion_sequence)

    final_score = calculate_score(final_emotion_sequence)
    st.metric("💯 Final Score", f"{final_score}/100")

    import pandas as pd
    score_timeline = [calculate_score([e]) for e in final_emotion_sequence]
    time_axis = [round(i * window_size, 2) for i in range(len(score_timeline))]
    df = pd.DataFrame({'Time': time_axis, 'Score': score_timeline})

    st.subheader("📈 Score Over Time")
    st.line_chart(df.set_index("Time")["Score"])

    st.subheader("📊 Emotion Distribution")
    flat = [e for lst in emotion_buckets.values() for e in lst]
    dist = Counter(flat)
    st.bar_chart({k: dist.get(k, 0) for k in emotion_labels})
