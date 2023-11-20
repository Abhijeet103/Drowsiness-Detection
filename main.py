import cv2
import torch
import numpy as np
import os
import pygame
import time
pygame.init()
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp6/weights/last.pt', force_reload=True)

# img = 'WhatsApp Image 2023-11-10 at 04.01.51_3d0d3155.jpg'
# results = model(img)
# print(results)
# results.save('output.png')
alarm = 'mixkit-classic-alarm-995.wav'
cap = cv2.VideoCapture(0)
count = 0
while cap.isOpened():
    ret, frame = cap.read()

    # Make detections
    results = model(frame)
    detections = results.xyxy[0]
    drowsy_detected = False

    for detection in detections:
        class_id = int(detection[5])
        confidence = float(detection[4])
        class_name = model.names[class_id]
        print(class_name)


        if class_name == 'drowsy' and confidence > 0.35: # Adjust the confidence threshold as needed
            drowsy_detected = True
            count += 1
            if count == 30 :
                pygame.mixer.music.load(alarm)
                pygame.mixer.music.play()
                time.sleep(5)
                pygame.mixer.music.stop()
                print(f"Drowsy detected with confidence: {confidence}")
                count =0
        if class_name == 'awake':
            count =0
    cv2.imshow('YOLO', np.squeeze(results.render()))

    if cv2.waitKey(10) & 0xFF == ord('e'):
        break
cap.release()
cv2.destroyAllWindows()