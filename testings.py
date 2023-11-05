import animalpose
import cv2
import math

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
TEST_IMAGE_PATH = "test_resting.jpg"  # Replace with the path to the image you want to test



#There are total three outcomes eating, standing, resting.

results = ["Standing","Eating","Resting"]

img= cv2.imread(TEST_IMAGE_PATH,cv2.IMREAD_UNCHANGED)

prediction = animalpose.recognize(img)
print(results[prediction])



# Parameters
MODEL_PATH = "mobilenetv2_image_classifier.h5"


# Manually define the labels based on the order they were used during training
labels_list = ['cognitive', 'Injured', 'mange']  # Replace with your actual labels



# Load the trained model
model = load_model(MODEL_PATH)

# Load and preprocess the test image
test_image = image.load_img(TEST_IMAGE_PATH, target_size=(224, 224))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
test_image /= 255.  # Normalize to [0,1]

# Make a prediction
predictions = model.predict(test_image)
predicted_class = np.argmax(predictions[0])

# Get the predicted label from the labels_list
predicted_label = labels_list[predicted_class]
   
print(f"Predicted class: {predicted_class}, label: {predicted_label}")



if results[prediction] == "Standing" and predicted_label == "Injured":
    print("The animal is standing and appears to be injured.")

elif results[prediction] == "Eating" and predicted_label == "Injured":
    print("The animal is eating and appears to be injured.")

elif results[prediction] == "Resting" and predicted_label == "Injured":
    print("The animal is resting and appears to be injured.")

elif results[prediction] == "Standing" and predicted_label == "mange":
    print("The animal is standing and might have mange.")

elif results[prediction] == "Eating" and predicted_label == "mange":
    print("The animal is eating and might have mange.")

elif results[prediction] == "Resting" and predicted_label == "mange":
    print("The animal is resting and might have mange.")

elif results[prediction] == "Standing" and predicted_label == "cognitive":
    print("The animal is standing and might have cognitive issues.")

elif results[prediction] == "Eating" and predicted_label == "cognitive":
    print("The animal is eating and might have cognitive issues.")

elif results[prediction] == "Resting" and predicted_label == "cognitive":
    print("The animal is resting and might have cognitive issues.")




