import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import os

# Load the saved model
model = tf.keras.models.load_model('C:/Users/chaitanya/Downloads/week 2 day 1/best_model.keras')

# Define function to preprocess the uploaded image
def preprocess_image(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    # Resize the image to 64x64 pixels
    resized_image = cv2.resize(gray_image, (64, 64))
    # Normalize the pixel values
    normalized_image = resized_image / 255.0
    # Add batch dimension and return
    return np.expand_dims(normalized_image, axis=0)

# Load font names from the training data
def load_font_names(data_dir):
    return sorted(os.listdir(data_dir))

# Define function to make predictions
def predict_font(image):
    # Preprocess the uploaded image
    preprocessed_image = preprocess_image(image)
    # Perform prediction using the model
    prediction = model.predict(preprocessed_image)
    # Load font names from training data
    font_names = load_font_names('C:/Users/chaitanya/Downloads/week 2 day 1/font detection/dataset')  # Adjust path accordingly
    # Get the font name corresponding to the predicted class
    predicted_font_index = np.argmax(prediction)
    if predicted_font_index < len(font_names):
        predicted_font = font_names[predicted_font_index]
    else:
        predicted_font = "Font not found "

    # Check confidence if within range
    confidence = prediction[0][predicted_font_index]  # Assuming first element is class probabilities
    if confidence < 0.9:  # Adjust this threshold based on your model's performance
        predicted_font = "Font not found due to less computational power unable to train on all fonts"

    return predicted_font

# Create the Streamlit app
st.title('Font Recognition App')

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Make prediction when a button is clicked
    if st.button('Predict Font'):
        # Perform prediction and display the result
        predicted_font = predict_font(image)
        st.write("Predicted Font:", predicted_font)




