from ultralytics import YOLO
import cv2

model = YOLO('runs/detect/train4/weights/best.pt')
results = model("media/zebrareal.jpg", show=True)
cv2.waitKey(0)