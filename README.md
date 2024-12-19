# Hockey-Video-Analysis
To enhance a hockey video analysis platform using computer vision techniques, we can develop a system that can tag and analyze hockey footage, helping extract valuable insights. Below is a Python code example leveraging OpenCV and deep learning models like YOLO (You Only Look Once) for object detection, player tracking, and event recognition.

In this example, we focus on:

    Player Detection: Detecting players on the field during a hockey game.
    Movement Tracking: Tracking the movement of players over time.
    Key Event Recognition: Recognizing specific actions like goals, assists, or fouls.

We will use OpenCV for video processing and YOLO for object detection. YOLO is a state-of-the-art deep learning model used for real-time object detection, which can be used to identify players and pucks in a hockey game.
Steps to Implement:

    Install Dependencies: First, install the required libraries:

pip install opencv-python opencv-python-headless numpy

    Download YOLO Pre-trained Weights: You can download pre-trained YOLO weights for object detection models like YOLOv3 or YOLOv4. These weights will be used to detect players and other objects in the hockey footage. Here is the link to download the YOLOv3 weights: YOLOv3 weights.

    Python Code Example:

import cv2
import numpy as np

# Load YOLO model (using pre-trained YOLOv3 weights and config)
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Define classes (e.g., 'person' class to identify hockey players)
with open("coco.names", "r") as f:  # coco.names file contains class names
    classes = [line.strip() for line in f.readlines()]

# Function to process video and detect players
def detect_players(video_path):
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize frame for YOLO input
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)
        
        # Post-process output to detect players
        class_ids = []
        confidences = []
        boxes = []
        height, width, channels = frame.shape
        
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > 0.5:  # Confidence threshold for detecting players
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply Non-Maximum Suppression to filter overlapping boxes
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        
        # Draw the bounding boxes around detected players
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display the frame with tagged players
        cv2.imshow("Frame", frame)
        
        # Press 'q' to exit video processing
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Function to track player movement (Optical Flow or Background Subtraction could be used here)
def track_player_movement(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow (movement tracking)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Draw the optical flow on the frame (for visual tracking)
        hsv = np.zeros_like(frame)
        hsv[..., 1] = 255
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Display the optical flow (movement)
        cv2.imshow("Player Movement", flow_rgb)
        
        prev_gray = gray
        
        # Press 'q' to exit video processing
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Main function to enhance hockey video analysis
def enhance_hockey_video_analysis(video_path):
    print("Detecting players...")
    detect_players(video_path)
    
    print("Tracking player movements...")
    track_player_movement(video_path)

# Provide the path to your hockey video
video_path = "hockey_game_video.mp4"
enhance_hockey_video_analysis(video_path)

Breakdown of the Code:

    YOLO Object Detection:
        The code loads the YOLO model and uses it to detect objects in the video.
        Specifically, we are interested in detecting players on the ice. YOLO is used for real-time object detection.
        The detected players are highlighted by drawing bounding boxes around them and labeling them with the person class.

    Tracking Player Movement:
        For player movement, we use optical flow (using OpenCV's calcOpticalFlowFarneback method), which estimates the motion of objects between two consecutive video frames. This helps track the movement of players across the field.

    Real-time Analysis:
        The detect_players function performs real-time detection of players in the video, and track_player_movement tracks their movements using optical flow.
        The enhance_hockey_video_analysis function ties everything together and processes the given hockey video.

Key Improvements:

    Real-time Detection: Players are detected in real time, with bounding boxes drawn around them.
    Movement Tracking: Optical flow is used to track player movements, which could help analyze how players move on the field and whether their movements correlate with specific game events.
    Action Recognition (Advanced): If needed, more sophisticated algorithms or models like RNNs or CNNs could be used to recognize specific game events such as goals, assists, and penalties by combining movement patterns and player positions.

Next Steps:

    Enhancing Event Recognition: For recognizing specific events (goals, penalties), a separate deep learning model could be trained with labeled event data. For example, if a goal is scored, the puck's position and player movement near the goal area can trigger event detection.
    Data Augmentation: Since hockey footage might have a lot of noise or occlusions, consider augmenting your dataset to improve model accuracy.
    Additional Features: Introduce more advanced tracking (like Kalman filtering) or multi-object tracking algorithms for better player separation.

Conclusion:

This code demonstrates how to enhance a hockey video analysis platform by detecting players, tracking their movements, and eventually identifying specific game events. The system can be extended and integrated into a larger application to analyze performance metrics and provide insights into player behavior and game strategies.
