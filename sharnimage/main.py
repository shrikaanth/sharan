import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# Load YOLOv4-tiny model
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")

# Load classes
with open("coco.names", "r") as f:
    classes = f.read().strip().split('\n')

# Define a global variable to store detected objects
detected_objects = []

def process_image(image_path):
    global detected_objects
    detected_objects = []  # Reset detected_objects list

    image = cv2.imread(image_path)
    height, width, _ = image.shape

    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                label = str(classes[class_id])
                detected_objects.append(label)

                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Detected Objects", cv2.resize(image, (800, 600)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_video(video_path):
    global detected_objects
    detected_objects = []  # Reset detected_objects list

    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(net.getUnconnectedOutLayersNames())

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and classes[class_id] == 'person':
                    detected_objects.append('person')

                    x, y, w, h = int(detection[0] * frame.shape[1]), int(detection[1] * frame.shape[0]), \
                                 int(detection[2] * frame.shape[1]), int(detection[3] * frame.shape[0])
                    person_roi = frame[y:y+h, x:x+w]

                    orb = cv2.ORB_create()
                    kp1, des1 = orb.detectAndCompute(frame, None)
                    kp2, des2 = orb.detectAndCompute(person_roi, None)

                    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                    matches = bf.match(des1, des2)
                    matches = sorted(matches, key=lambda x: x.distance)

                    if len(matches) > 10:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Detected Objects", cv2.resize(frame, (800, 600)))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg")])
    if file_path:
        process_image(file_path)

def upload_video():
    file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4")])
    if file_path:
        process_video(file_path)

# Create Tkinter GUI
root = tk.Tk()
root.title("Object Detection")

title_label = tk.Label(root, text="Object Detection System", font=("Helvetica", 16))
title_label.pack(pady=10)

upload_image_button = tk.Button(root, text="Upload Image", command=upload_image)
upload_image_button.pack(pady=10)

upload_video_button = tk.Button(root, text="Upload Video", command=upload_video)
upload_video_button.pack(pady=10)

detected_objects_label = tk.Label(root, text="Detected Objects:", font=("Helvetica", 12))
detected_objects_label.pack(pady=10)

detected_objects_display = tk.Label(root, text="", font=("Helvetica", 12))
detected_objects_display.pack()

root.mainloop()
