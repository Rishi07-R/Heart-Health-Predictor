import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.preprocessing import image # type: ignore

model = tf.keras.models.load_model("alphabet_model.h5")

class_labels = ['A', 'B', 'C', 'D', 'E']

# Preprocess Function
def preprocess_image(img_path, img_size=64):
    
    img = cv2.imread(img_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

    resized = cv2.resize(rgb, (img_size, img_size))

    img_array = resized.astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)  

    return img_array

# Predict Function
def predict_letter(img_path):
    img_array = preprocess_image(img_path)
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    return class_labels[class_idx], confidence

# Test with a Google Image
img_path = "C:\\Users\\rohan\\Desktop\\myproject2\\myproject2\\images.jpg"  
letter, conf = predict_letter(img_path)
img2_path="C:\\Users\\rohan\Desktop\\myproject2\\myproject2\\download.jpeg"
letter2, conf2 = predict_letter(img2_path)
print(f"✅ Predicted Letter: {letter}")
print(f"✅ Predicted Letter: {letter2}")
