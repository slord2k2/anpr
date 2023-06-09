# remove warning message
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# required library
import cv2
import numpy as np
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

# Extract license plate from sample image
        
wpod_net_path = "wpod-net.json"
wpod_net = load_model(wpod_net_path)

# if wpod_net is not None:
#     print("Model loaded successfully...")

def preprocess_image(image_path,resize=False):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    if resize:
        img = cv2.resize(img, (224,224))
    return img

def get_plate(image_path, Dmax=608, Dmin = 608):
    vehicle = preprocess_image(image_path)
    ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)
    _ , LpImg, _, cor = detect_lp(wpod_net, vehicle, bound_dim, lp_threshold=0.5)
    return vehicle, LpImg, cor

# test_image_path =" E:\Harsh\Anonymous\File\Intern\Machine Learning\automatic-number-plate-recognition-python\Plate_detect_and_recognize\Indian_car\15e7c19f-da3c-4f0d-91f3-552afabc8961___41FxrEl7jHL.jpg.jpeg"
test_image_path = "temp_image.jpg"
vehicle, LpImg,cor = get_plate(test_image_path)

# Segementing license characters
if(len(LpImg)==0):
    print("No License plate detected")
    sys.exit()
#check if there is at least one license image
if (len(LpImg)): #check if there is at least one license image
    # Scales, calculates absolute values, and converts the result to 8-bit.
    plate_image = cv2.convertScaleAbs(LpImg[0], alpha=(255.0))
    
    # convert to grayscale and blur the image
    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(7,7),0)
    
    # Applied inversed thresh_binary 
    binary = cv2.threshold(blur, 180, 255,
                         cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)

plot_image = [plate_image, gray, blur, binary,thre_mor]
plot_name = ["plate_image","gray","blur","binary","dilation"]


# for image, name in zip(plot_image, plot_name):
#     cv2.imshow(name, image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

def sort_contours(cnts,reverse = False):
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return cnts


cont, _  = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# creat a copy version "test_roi" of plat_image to draw bounding box
test_roi = plate_image.copy()

# Initialize a list which will be used to append charater image
crop_characters = []

# define standard width and height of character
digit_w, digit_h = 30, 60


for c in sort_contours(cont):
    (x, y, w, h) = cv2.boundingRect(c)
    ratio = h/w
    if 1<=ratio<=3.5: # Only select contour with defined ratio
        if h/plate_image.shape[0]>=0.5: # Select contour which has the height larger than 50% of the plate
            # Draw bounding box arroung digit number
            cv2.rectangle(test_roi, (x, y), (x + w, y + h), (0, 255,0), 2)

            # Sperate number and gibe prediction
            curr_num = thre_mor[y:y+h,x:x+w]
            curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
            _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            crop_characters.append(curr_num)
            
if(len(crop_characters)==0):
    print("No characters detected")
    sys.exit()    
print("Detect {} letters...".format(len(crop_characters)))

# cv2.imshow('test',test_roi)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# for i in range(len(crop_characters)):
#     cv2.imshow(crop_characters[i],cmap="gray")
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Load pre-trained MobileNets model and predict
# Load model architecture, weight and labels

json_file = open('MobileNets_character_recognition.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("License_character_recognition_weight.h5")
# print("[INFO] Model loaded successfully...")

labels = LabelEncoder()
labels.classes_ = np.load('license_character_classes.npy')
print("[INFO] Labels loaded successfully...")

# pre-processing input images and pedict with model
def predict_from_model(image,model,labels):
    image = cv2.resize(image,(80,80))
    image = np.stack((image,)*3, axis=-1)
    prediction = labels.inverse_transform([np.argmax(model.predict(image[np.newaxis,:]))])
    return prediction
# fig = plt.figure(figsize=(15,3))
cols = len(crop_characters)
# grid = gridspec.GridSpec(ncols=cols,nrows=1,figure=fig)
final_string = ''
for i,character in enumerate(crop_characters):
#     fig.add_subplot(grid[i])
    title = np.array2string(predict_from_model(character,model,labels))
    print(title)
    plt.title('{}'.format(title.strip("'[]"),fontsize=20))
    final_string+=title.strip("'[]")
#     plt.axis(False)
#     plt.imshow(character,cmap='gray')
print(final_string)
cv2.imshow(final_string,vehicle)
cv2.waitKey(0)
cv2.destroyAllWindows()