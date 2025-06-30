import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import tensorflow_hub as hub
from tensorflow.keras.utils import get_custom_objects

# Define the custom HubLayerWrapper
class HubLayerWrapper(tf.keras.layers.Layer):
    def __init__(self, handle, trainable=False, **kwargs):
        super(HubLayerWrapper, self).__init__(trainable=trainable, **kwargs)
        self.handle = handle
        self.hub_layer = hub.KerasLayer(handle, trainable=trainable)

    def call(self, inputs):
        return self.hub_layer(inputs)

# Register the custom layer
get_custom_objects().update({'HubLayerWrapper': HubLayerWrapper})

# Load the trained model
model = tf.keras.models.load_model('waste_classification_model.h5')

# Define a function to preprocess the image
def preprocess_image(image):
    image = image.convert('RGB')
    image_resized = image.resize((224, 224))
    image_array = np.array(image_resized)
    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)
    input_array = np.expand_dims(image_array, axis=0)
    return input_array


# Streamlit app UI
st.title("Waste Classification App")
st.write("Upload an image to classify it as **Organic** or **Recyclable**.")

# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    image_display = image.resize((300, 300))
    st.image(image_display, caption="Uploaded Image")


    # Preprocess the image
    input_image = preprocess_image(image)

    # Make a prediction
    prediction = model.predict(input_image)
    predicted_label = np.argmax(prediction)
    probabilities = tf.nn.softmax(prediction)
    confidence = float(np.max(probabilities))*100

    # Display the result
    if predicted_label == 0:
      st.success(f"✅ The waste is classified as **Organic** with {confidence:.2f}% confidence.")
    else:
      st.success(f"♻️ The waste is classified as **Recyclable** with {confidence:.2f}% confidence.")
