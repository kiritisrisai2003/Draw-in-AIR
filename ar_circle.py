import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Initialize parameters
drawing = False
points = []
target_radius = 100  # Target radius for the circle
center_point = (250, 250)  # Center of the drawing area
score = 0
color = (255, 255, 255)  # Default color for drawing
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 255, 255)]  # Increased color options
current_color_index = 0

# Initialize camera
cap = cv2.VideoCapture(0)

# Function to draw the start button
def draw_start_button(frame):
    cv2.rectangle(frame, (50, 50), (150, 100), (0, 255, 0), -1)  # Green button
    cv2.putText(frame, 'Start', (75, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# Function to draw the reset button
def draw_reset_button(frame):
    cv2.rectangle(frame, (50, 120), (150, 170), (0, 0, 255), -1)  # Red button
    cv2.putText(frame, 'Reset', (75, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# Function to draw color selection panel
def draw_color_panel(frame):
    for i, c in enumerate(colors):
        cv2.rectangle(frame, (10 + i*40, 350), (50 + i*40, 390), c, -1)

# Function to draw a dot where the user will draw
def draw_dot(frame, point):
    cv2.circle(frame, point, 10, color, -1)

# Function to calculate accuracy
def calculate_accuracy(points):
    if len(points) < 20:
        return "Not enough points!"
    
    distances = [np.linalg.norm(np.array(point) - np.array(center_point)) for point in points]
    mean_distance = np.mean(distances)
    accuracy = max(0, 100 - (np.abs(mean_distance - target_radius) / target_radius * 100))
    
    return f'Your accuracy: {accuracy:.2f}%'

# Main loop
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    
    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and detect hands
    results = hands.process(frame_rgb)
    
    # Draw start and reset buttons
    draw_start_button(frame)
    draw_reset_button(frame)
    draw_color_panel(frame)

    # Check if hands are detected
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        finger_tip_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1])
        finger_tip_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0])
        
        # Check if the user clicked the start button
        if 50 < finger_tip_x < 150 and 50 < finger_tip_y < 100 and not drawing:
            drawing = True  # Start drawing when the button is pressed
        
        # Check if the user clicked the reset button
        if 50 < finger_tip_x < 150 and 120 < finger_tip_y < 170:
            points.clear()  # Clear the points for reset
            drawing = False
            score = 0
        
        # Check color selection
        for i, c in enumerate(colors):
            if 10 + i * 40 < finger_tip_x < 50 + i * 40 and 350 < finger_tip_y < 390:
                color = c  # Change color on selection
                points.clear()  # Clear previous drawings when changing color
                break
        
        if drawing:
            # Add the current finger tip position to points
            points.append((finger_tip_x, finger_tip_y))
            draw_dot(frame, (finger_tip_x, finger_tip_y))

            # Draw the path on the frame
            for i in range(1, len(points)):
                cv2.line(frame, points[i - 1], points[i], color, 5)

    # Show accuracy score when finished
    if not drawing and points:
        accuracy_msg = calculate_accuracy(points)
        cv2.putText(frame, accuracy_msg, (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
    # Show the final frame
    cv2.imshow('AR Circle Drawing', frame)

    # Exit on 'q'
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
