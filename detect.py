### detect.py
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the trained model
model = load_model('model.h5')  # You should have a trained model saved as model.h5

# Load Haar cascades
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')


def detect_state(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4)
    state = 'Normal'

    for (x, y, w, h) in eyes:
        eye = gray[y:y+h, x:x+w]
        eye = cv2.resize(eye, (224, 224))
        eye = eye.astype("float") / 255.0
        eye = img_to_array(eye)
        eye = np.expand_dims(eye, axis=0)

        pred = model.predict(eye)[0][0]

        if pred > 0.5:
            state = 'Drowsy'
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    return state, frame
