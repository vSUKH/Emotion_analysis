import streamlit as st
import cv2
import numpy as np
from keras.models import model_from_json

# Load the emotion detection model
def load_model():
    with open("emotiondetector_.json", "r") as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    model.load_weights("Emotiondetector_.h5")
    return model

# Preprocess the image (resize and normalize)
def preprocess_image(image):
    image = cv2.resize(image, (48, 48))  # Resize to 48x48
    img_array = np.array(image)
    img_array = img_array.reshape(1, 48, 48, 1)  # Reshape for the model
    img_array = img_array / 255.0  # Normalize
    return img_array

# Emotion labels
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# Emotion detection on a frame
def detect_emotion(model, frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)  # Detect faces

    detected_emotions = []
    for (x, y, w, h) in faces:
        roi = gray_frame[y:y+h, x:x+w]  # Region of interest
        if roi.size == 0:  # Skip invalid ROIs
            continue

        processed_image = preprocess_image(roi)
        predictions = model.predict(processed_image)
        emotion = labels[np.argmax(predictions)]
        detected_emotions.append(emotion)

        # Draw rectangle and emotion text on frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return frame, detected_emotions

# Load the model and Haar Cascade for face detection
model = load_model()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Streamlit UI
st.title("Real-Time Emotion Detector")
st.write("Real-time emotion detection using your webcam.")

# Webcam Option
use_webcam = st.checkbox("Use Webcam for Real-Time Emotion Detection")
if use_webcam:
    st.write("Webcam is running. Press 'Stop Webcam' to exit.")
    FRAME_WINDOW = st.image([])  # For real-time display

    camera = cv2.VideoCapture(0)  # Open the webcam
    if not camera.isOpened():
        st.error("Error: Webcam not accessible.")
    else:
        stop_webcam = st.button("Stop Webcam")  # Button to stop the webcam
        while not stop_webcam:
            ret, frame = camera.read()
            if not ret:
                st.error("Failed to capture a frame from the webcam.")
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert color to RGB
            processed_frame, emotions = detect_emotion(model, frame)  # Perform emotion detection
            FRAME_WINDOW.image(processed_frame)  # Display the processed frame

            # Display detected emotions as text
            if emotions:
                st.write(f"Detected Emotions: {', '.join(emotions)}")
            else:
                st.write("No face detected.")

        camera.release()  # Release the webcam when the loop exits
        st.write("Webcam stopped.")
 


