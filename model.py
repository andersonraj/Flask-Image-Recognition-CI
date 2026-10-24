"""
Machine Learning module for loading the model 
and processing images for prediction.
"""
import numpy as np
from keras.models import load_model
from keras.utils import img_to_array
from PIL import Image

# Loading model
model = load_model("digit_model.h5")

def preprocess_img(img_path):
    """
    Open, resize, and reshape the uploaded image 
    to match the input requirements of the Keras model.
    """
    op_img = Image.open(img_path)
    img_resize = op_img.resize((224, 224))
    img2arr = img_to_array(img_resize) / 255.0
    img_reshape = img2arr.reshape(1, 224, 224, 3)
    return img_reshape

def predict_result(predict):
    """
    Run the preprocessed image through the loaded Keras model 
    and return the predicted integer class.
    """
    pred = model.predict(predict)
    return np.argmax(pred[0], axis=-1)
