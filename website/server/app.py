import joblib
import numpy as np
from flask import Flask, request,jsonify, render_template
from flask_cors import CORS
import cv2
import os


model = joblib.load('../../model/knn_model.pkl')
scaler = joblib.load('../../model/scaler.pkl')

app = Flask(__name__, template_folder = 'templates')
CORS(app)

def imageToHsv (img):
    image = cv2.imread(img)
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h_mean = np.mean(img_hsv[:,:,0])
    s_mean = np.mean(img_hsv[:,:,1])
    v_mean = np.mean(img_hsv[:,:,2])
    h_std = np.std(img_hsv[:,:,0])
    s_std = np.std(img_hsv[:,:,1])
    v_std = np.std(img_hsv[:,:,2])

    features = np.array([h_mean,s_mean, v_mean,h_std,s_std,v_std])
    return features

@app.route('/')
def home():
    return render_template('predict_img.html')

@app.route('/knn_predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error':"no file part"})
    
    image = request.files['file']

    if image.filename == '':
        return jsonify({'error': 'No selected file'})
    

    filename = image.filename

    image_path = os.path.join("./image",filename)

    image.save(image_path)

    features = imageToHsv(image_path)
    features_scaled = scaler.transform(features.reshape(1,-1))
    pred = model.predict(features_scaled)

    return jsonify({'prediction': pred[0]})


if __name__ == '__main__':
    app.run(debug=True, host = '0.0.0.0', port=8000)