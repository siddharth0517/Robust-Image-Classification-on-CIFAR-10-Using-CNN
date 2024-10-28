import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model("model.keras")

# CIFAR-10 classes
classes = ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]

# Function to preprocess and predict the image
def preprocess_image(img):
    img = img.resize((32, 32))
    img = np.array(img) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def predict_image(img):
    img = preprocess_image(img)
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)
    return classes[predicted_class]

# Streamlit app
st.title("CIFAR-10 Image Classifier")

# File uploader
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Make a prediction
    st.write("Classifying...")
    label = predict_image(image)
    st.write(f"Prediction: {label}")
