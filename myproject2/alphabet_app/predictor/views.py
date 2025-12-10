import os
import cv2
import numpy as np
import tensorflow as tf
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load model once when server starts
model_path = os.path.join(BASE_DIR, "alphabet_model.h5")
model = tf.keras.models.load_model(model_path)

# Class labels
class_labels = ['A', 'B', 'C', 'D', 'E']

def preprocess_image(img_path, img_size=64):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
    resized = cv2.resize(rgb, (img_size, img_size))
    img_array = resized.astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_view(request):
    result = None
    uploaded_url = None

    if request.method == "POST" and request.FILES.get("letter_image"):
        # Save uploaded image
        uploaded_file = request.FILES["letter_image"]
        fs = FileSystemStorage()
        filename = fs.save(uploaded_file.name, uploaded_file)
        file_path = fs.path(filename)
        uploaded_url = fs.url(filename)

        # Predict
        img_array = preprocess_image(file_path)
        prediction = model.predict(img_array)
        class_idx = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        result = f"Predicted Letter: {class_labels[class_idx]} ({confidence:.2f}%)"
    
    return render(request, "predictor/index.html", {"result": result, "uploaded_url": uploaded_url})
