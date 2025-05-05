import cv2
import numpy as np
import tensorflow as tf
import os

print("Starting the script...")

# Model file path
model_path = r'C:\Users\barig\OneDrive\Documents\opencv\.vscode\facialemotionmodel.h5'
print(f"Looking for model at: {model_path}")
print(f"File exists: {os.path.exists(model_path)}")

# Try to load the model
try:
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Load Haar Cascade
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Preprocessing function
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Start webcam
def find_working_camera(max_index=5):
    for i in range(max_index):
        cam = cv2.VideoCapture(i)
        if cam.isOpened():
            print(f"Webcam found at index {i}")
            return cam
        cam.release()
    return None

webcam = find_working_camera()

if webcam is None:
    print("Error: Could not access any webcam.")
    exit()
else:
    print("Webcam opened successfully")


labels = {
    0: 'angry', 1: 'disgust', 2: 'fear',
    3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'
}

while True:
    print("Capturing frame...")
    ret, im = webcam.read()

    if not ret:
        print("Error: Failed to capture image from webcam.")
        break
    print("Frame captured.")

    # Flip the image horizontally (mirroring effect)
    im = cv2.flip(im, 1)  # 1 flips the image horizontally
    

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        print("No faces detected.")

    try:
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            cv2.rectangle(im, (x, y), (x+w, y+h), (255, 0, 0), 2)
            face = cv2.resize(face, (48, 48))
            img = extract_features(face)
            pred = model.predict(img)
            label = labels[pred.argmax()]
            cv2.putText(im, label, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (0, 0, 255), 2)

        cv2.imshow("Facial Emotion Recognition", im)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

    except cv2.error as e:
        print(f"OpenCV error: {e}")
        pass

# Cleanup
webcam.release()
cv2.destroyAllWindows()
