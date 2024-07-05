from ultralytics import YOLO
import cv2
import cvzone
import numpy as np
from sort import *

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

model = YOLO('runs/detect/train4/weights/best.pt')
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
passed = []
classNames = ["Elephant", "Giraffe", "Zebra", "Lion"]

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    detections = np.empty((0, 5))
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (119, 221, 119), 3)

            w = x2 - x1
            h = y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))

            print(x1, y1, x2, y2)

            confidence_rating = round(float(box.conf[0]), 2)
            class_name = int(box.cls[0])
            currentClass = classNames[class_name]

            cvzone.putTextRect(img, f'{currentClass} {confidence_rating}', (max(0, x1), max(35, y1)), scale=0.7, thickness=2)
            
            currentArray = np.array([x1, y1, x2, y2, confidence_rating])
            detections = np.vstack((detections, currentArray))
    
    trackerResults = tracker.update(detections)

    for result in trackerResults:
        x1,y1,x2,y2,id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        print(result)
        w = x2 - x1
        h = y2 - y1
        cx, cy = x1 + w//2, y1 + h//2
        cv2.circle(img, (cx, cy), 5, (0, 0, 255), cv2.FILLED)
        if id not in passed:
            passed.append(id)

    cvzone.putTextRect(img, f"Count: {len(passed)}", (0, 30), colorR = (0, 0, 255))

    passed.clear()
    cv2.imshow("Image", img)
    cv2.waitKey(1)