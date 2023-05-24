import os
import sys

from flask import Flask, render_template, request
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from local_utils import detect_lp
from os.path import splitext,basename
from keras.models import model_from_json
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.applications.mobilenet_v2 import preprocess_input
from sklearn.preprocessing import LabelEncoder
import glob
from load_model import load_model

app = Flask(__name__)

# Load pre-trained MobileNets model and predict
def predict_from_model(image, model, labels):
    image = cv2.resize(image, (80, 80))
    image = np.stack((image,) * 3, axis=-1)
    prediction = labels.inverse_transform([np.argmax(model.predict(image[np.newaxis, :]))])
    return prediction
@app.route('/',methods=['GET'])
def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    try:
        image = request.files['image']
        image_path = 'temp_image.jpg'
        image.save(image_path)

        # Process the image using the algorithm mentioned above
        img = load_img(image_path, target_size=(224, 224))
        img = img_to_array(img, dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.convertScaleAbs(img, alpha=(255.0))

        plate_image = img
        wpod_net_path = "wpod-net.json"
        wpod_net = load_model(wpod_net_path)

        def preprocess_image(image_path, resize=False):
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255
            if resize:
                img = cv2.resize(img, (224, 224))
            return img

        def get_plate(image_path, Dmax=608, Dmin=608):
            vehicle = preprocess_image(image_path)
            ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
            side = int(ratio * Dmin)
            bound_dim = min(side, Dmax)
            _, LpImg, _, cor = detect_lp(wpod_net, vehicle, bound_dim, lp_threshold=0.5)
            return vehicle, LpImg, cor

        test_image_path = "temp_image.jpg"
        vehicle, LpImg, cor = get_plate(test_image_path)
        if len(LpImg) == 0:
            plate_number = "No License Plate Detected"
        else:
            plate_image = cv2.convertScaleAbs(LpImg[0], alpha=(255.0))

            # convert to grayscale and blur the image
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (7, 7), 0)

            # Applied inversed thresh_binary
            binary = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

            kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)

            def sort_contours(cnts, reverse=False):
                i = 0
                boundingBoxes = [cv2.boundingRect(c) for c in cnts]
                (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                                    key=lambda b: b[1][i], reverse=reverse))
                return cnts

            cont, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # create a copy version "test_roi" of plate_image to draw bounding box
            test_roi = plate_image.copy()

            # Initialize a list which will be used to append character image
            crop_characters = []

            # define standard width and height of character
            digit_w, digit_h = 30, 60

            for c in sort_contours(cont):
                (x, y, w, h) = cv2.boundingRect(c)
                ratio = h / w
                if 1 <= ratio <= 3.5:  # Only select contour with defined ratio
                    if h / plate_image.shape[0] >= 0.5:  # Select contour which has the height larger than 50% of the plate
                        # Draw bounding box around digit number
                        cv2.rectangle(test_roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

                        # Separate number and give prediction
                        curr_num = thre_mor[y:y + h, x:x + w]
                        curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                        _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        crop_characters.append(curr_num)
            
            json_file = open('MobileNets_character_recognition.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = model_from_json(loaded_model_json)
            model.load_weights("License_character_recognition_weight.h5")
            # print("[INFO] Model loaded successfully...")

            labels = LabelEncoder()
            labels.classes_ = np.load('license_character_classes.npy')
            # pre-processing input images and pedict with model
            def predict_from_model(image,model,labels):
                image = cv2.resize(image,(80,80))
                image = np.stack((image,)*3, axis=-1)
                prediction = labels.inverse_transform([np.argmax(model.predict(image[np.newaxis,:]))])
                # print(prediction)
                return prediction
            if len(crop_characters) == 0:
                plate_number = "No License Plate Detected"
            else:
                plate_number = ""
                for i,character in enumerate(crop_characters):
                    title = np.array2string(predict_from_model(character,model,labels))
                    print(title)
                    plate_number+=title.strip("'[]")

            return render_template('result.html', plate_number=plate_number, image_path='result.png')


    except Exception as e:
        # Handle any exceptions that may occur during processing
        plate_number = "Error occurred: " + str(e)
        return render_template('result.html', plate_number=plate_number)

if __name__ == '__main__':
 
    # run() method of Flask class runs the application
    # on the local development server.
    app.run()