import cv2
from ultralytics import YOLO
import pyttsx3
import time

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Initialize Text-to-Speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Start webcam
cap = cv2.VideoCapture(0)

prev_time = 0
last_spoken = ""
last_spoken_time = 0
cooldown = 3  # seconds between voice alerts

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    detected_objects = []

    for r in results:
        boxes = r.boxes
        for box in boxes:
            class_id = int(box.cls[0])
            label = model.names[class_id]
            detected_objects.append(label)

    # Speak only if new object detected
    current_time = time.time()
    if detected_objects:
        main_object = detected_objects[0]  # speak first detected object

        if (main_object != last_spoken) or (current_time - last_spoken_time > cooldown):
            engine.say(f"{main_object} detected")
            engine.runAndWait()
            last_spoken = main_object
            last_spoken_time = current_time

    annotated_frame = results[0].plot()

    # FPS calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    cv2.putText(annotated_frame, f"FPS: {int(fps)}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2)

    cv2.imshow("Real-Time Detection with Voice Alert", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
