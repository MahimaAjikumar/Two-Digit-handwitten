from tensorflow.keras.models import load_model
import streamlit as st
from PIL import Image
import numpy as np
import cv2


model = load_model("digit_2digit_model.keras")

st.title("Handwritten Two-Digit Recognition (00-99)")

uploaded_file = st.file_uploader(
    "Upload a Handwritten Image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    
    image = Image.open(uploaded_file).convert("L")

    img = np.array(image)

    img = cv2.resize(img, (64, 64))

    img = img.astype("float32") / 255.0

    st.image(image, caption="Uploaded Image", width=200)

    img = img.reshape(1, 64, 64, 1)

    prediction = model.predict(img, verbose=0)

    predicted_class = np.argmax(prediction)

    confidence = np.max(prediction) * 100

    st.success(f"Predicted Number : {predicted_class:02d}")

    st.write(f"Confidence : {confidence:.2f}%")