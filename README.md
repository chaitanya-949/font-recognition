
# Font Detection with Convolutional Neural Networks

This project aims to develop a font detection system using Convolutional Neural Networks (CNNs). The system takes images of text written in various fonts as input and predicts the font used in the text.

Installation
To run the code in this repository, you need to have the following dependencies installed:

Python 3.x

OpenCV (cv2)

NumPy

TensorFlow

Pandas

Matplotlib

scikit-learn

You can install the required Python packages using pip install opencv-python numpy tensorflow pandas matplotlib scikit-learn

# Dataset

The dataset consists of images of text samples written in various fonts. Each image is labeled with the name of the font used. The dataset is organized into subdirectories, where each subdirectory represents a different font.

# Data Loading and Preprocessing

Images are loaded from the dataset directory, resized to a target size, converted to grayscale, and normalized.

The dataset is split into training, validation, and test sets.

# Model Architecture

The font detection model is implemented using a CNN architecture. The model consists of four convolutional layers followed by max-pooling layers, a flatten layer, a fully connected layer, dropout regularization, and an output layer.

# Training

The model is trained using the training set and validated using the validation set. Training is stopped early if the validation loss does not improve for a specified number of epochs.

# Evaluation

The trained model is evaluated on the test set to assess its performance in font detection. The test loss and accuracy are reported.

# Front End

# A Streamlit app is provided for font recognition using the trained model. To run the front end:

Ensure all dependencies are installed.

Download the trained model file ('best_model.keras') and place it in the appropriate directory.

Run the Streamlit app script using the command:

streamlit run app.py

Upload an image containing text and click the "Predict Font" button to see the predicted font.

