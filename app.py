import gradio as gr
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models import convnext_tiny
from ultralytics import YOLO

# Load face detector model (YOLOv8)
face_detector = YOLO("yolov8n-face.pt")  # Replace with correct model if different

# Load emotion recognition model (ConvNeXt)
model = convnext_tiny(num_classes=8)
model.load_state_dict(torch.load("emotion_model.pth", map_location="cpu"))
model.eval()

# Emotion labels
emotion_labels = ['neutral', 'happy', 'sad', 'angry', 'surprised', 'fear', 'disgust', 'contempt']

# Transforms
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 🔁 Gradio inference function
def detect_and_classify(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detector(image)
    annotated = image.copy()

    for box in results[0].boxes.xyxy:
        x1, y1, x2, y2 = map(int, box[:4])
        face = image[y1:y2, x1:x2]
        if face.size == 0: continue

        # Preprocess face
        face_pil = Image.fromarray(face)
        face_tensor = preprocess(face_pil).unsqueeze(0)

        # Predict emotion
        with torch.no_grad():
            output = model(face_tensor)
            pred = torch.argmax(output, dim=1).item()
            label = emotion_labels[pred]

        # Draw on image
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255, 0, 0), 2)

    return annotated

# 🎛️ Gradio interface
gr.Interface(
    fn=detect_and_classify,
    inputs=gr.Image(),
    outputs="image",
    live=True,
    title="🎭 Real-Time Emotion Recognition with ConvNeXt + YOLO"
).launch(share=True)
