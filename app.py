import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

st.title("BatikLens App")

MODEL_PATH = "batiklens_model.h5"

if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/uc?id=1Bke9qFhYmLDbD1C8a_IJQoSPl5q4razz"
    gdown.download(url, MODEL_PATH, quiet=False)

model = tf.keras.models.load_model(MODEL_PATH)

uploaded_file = st.file_uploader("Upload gambar batik", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)

    st.write("Hasil prediksi (class index):", class_index)
