import cv2
from ultralytics import YOLO

# 1. Load a pre-trained, lightweight YOLOv8 nano model
# The script will automatically download 'yolov8n.pt' the first time you run it
model = YOLO("yolov8n.pt")

# 2. Initialize video capture (0 is your built-in webcam)
cap = cv2.VideoCapture(0)

print("AI Tracking Started! Press 'q' inside the video window to exit.")
print("-" * 60)

while cap.isOpened():
    success, frame = cap.read()
    
    if not success:
        print("Webcam feed not detected. Exiting...")
        break

    # 3. Run YOLOv8 object tracking on the frame
    # persist=True ensures the AI gives a permanent tracking ID number to each object
    results = model.track(frame, persist=True, stream=True)

    # Initialize annotated_frame with the current raw frame as a fallback
    annotated_frame = frame
    for r in results:
        # Draw bounding boxes, item names, and tracking IDs onto the window
        annotated_frame = r.plot()

    # 4. Show the live camera window
    cv2.imshow("CodeAlpha AI Object Tracker", annotated_frame)

    # Stop the program if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Clean up window and close the camera resource properly
cap.release()
cv2.destroyAllWindows()
