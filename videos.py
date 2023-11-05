import animalpose
import cv2
from collections import Counter

# There are total three outcomes: eating, standing, resting.
results = ["Standing", "Eating", "Resting"]

video_capture = cv2.VideoCapture('test.mp4')

frame_predictions = []

while True:
    ret, frame = video_capture.read()

    if not ret:
        break

    prediction = animalpose.recognize(frame)
    frame_predictions.append(results[prediction])

video_capture.release()

if frame_predictions:
    most_common_prediction = Counter(frame_predictions).most_common(1)[0][0]
    print("Cumulative Prediction:", most_common_prediction)
else:
    print("No frames processed.")
