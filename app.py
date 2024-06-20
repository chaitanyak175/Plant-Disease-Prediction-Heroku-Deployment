from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import time

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Enable CORS for all routes

# Load your model
model = tf.keras.models.load_model("trained_plant_disease_model.keras")

# Define the list of class names
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    app.logger.info("Received a request")

    try:
        if 'file' not in request.files:
            app.logger.error('No file part in the request')
            return jsonify({'error': 'No file part in the request'}), 400

        file = request.files['file']
        app.logger.info(f"File received: {file.filename}")

        if file.filename == '':
            app.logger.error('No selected file')
            return jsonify({'error': 'No selected file'}), 400

        # Read the image
        read_start_time = time.time()
        img = Image.open(file.stream)
        img = img.resize((128, 128))  # Resize the image to match the model's expected input size
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Create a batch
        read_end_time = time.time()
        app.logger.info(f"Image read and processed in {read_end_time - read_start_time} seconds")

        # Make a prediction
        predict_start_time = time.time()
        predictions = model.predict(img_array)
        predict_end_time = time.time()
        app.logger.info(f"Prediction completed in {predict_end_time - predict_start_time} seconds")
        
        predicted_class = class_names[np.argmax(predictions)]

        end_time = time.time()
        app.logger.info(f"Total prediction process completed in {end_time - start_time} seconds")

        return jsonify({'disease': predicted_class})

    except Exception as e:
        app.logger.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/')
def home():
    return "Plant Disease Prediction API"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)