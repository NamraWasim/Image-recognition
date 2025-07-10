from ultralytics import YOLO
import cv2

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Webcam not detected!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Error reading frame")
        break

    # Run detection
    results = model.predict(frame, show=False)
    annotated_frame = results[0].plot()

    # Show window
    try:
        cv2.imshow("YOLOv8 Real-Time Detection", annotated_frame)
    except:
        print("❌ cv2.imshow failed - GUI not supported in this environment.")
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()