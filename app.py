import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
st.set_page_config(page_title="Handwritten Digit Recognition", page_icon="✍️")

@st.cache_resource  
def load_model():
    try:
        model = tf.keras.models.load_model('hand_written_model.keras')
        return model
    except ValueError as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model = load_model()



st.title("🖊️Handwritten Digit Recognition")
st.write("Upload an image of a handwritten digit (0-9), and the model will predict it!")


uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    
    try:
        
        gray_image = image.convert('L')
        
        
        resized_image = gray_image.resize((28, 28))
        
        img_array = np.array(resized_image) / 255.0
        
        img_array = img_array.reshape(1, 784)
        
        prediction = model.predict(img_array)
        predicted_digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        
        st.success(f"### ✨ Predicted Digit: **{predicted_digit}**")
        st.write(f"Confidence: {confidence:.2f}%")
        
        
    except Exception as e:
        st.error(f"Error processing image: {e}. Ensure the image is a valid handwritten digit (e.g., 28x28 or resizable).")
else:
    st.info("Please upload an image to get started.")
