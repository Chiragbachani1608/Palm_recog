import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def reshape_to_hand_shape(image, hand_landmarks, height, width):
    """Crop image to fit the hand shape."""
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    points = [
        (int(landmark.x * width), int(landmark.y * height))
        for landmark in hand_landmarks.landmark
    ]
    cv2.fillConvexPoly(mask, np.array(points, dtype=np.int32), 255)
    hand_image = cv2.bitwise_and(image, image, mask=mask)
    return hand_image

def process_and_detect_palm():
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access the camera.")
        return

    with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7) as hands:
        print("Press 'c' to capture the image and 'q' to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break

            # Flip and convert frame for consistency
            image = cv2.flip(frame, 1)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Process the frame to detect hand landmarks
            results = hands.process(rgb_image)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                    )

            # Show hand landmarks in real-time
            cv2.imshow("Hand Detection", image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):  # Capture the image for palm line detection
                captured_image = image.copy()
                break
            elif key == ord('q'):  # Quit
                cap.release()
                cv2.destroyAllWindows()
                return

    cap.release()
    cv2.destroyAllWindows()

    if results.multi_hand_landmarks:
        height, width, _ = captured_image.shape
        hand_image = reshape_to_hand_shape(captured_image, results.multi_hand_landmarks[0], height, width)

        # Convert to grayscale
        gray = cv2.cvtColor(hand_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Adaptive Thresholding
        adaptive_thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        # Combine with Canny Edge Detection
        edges = cv2.Canny(gray, threshold1=30, threshold2=100)
        combined_edges = cv2.bitwise_or(adaptive_thresh, edges)

        # Overlay the edges on the original image
        combined_color = cv2.cvtColor(combined_edges, cv2.COLOR_GRAY2BGR)
        final_output = cv2.addWeighted(hand_image, 0.8, combined_color, 0.5, 0)

        # Show the final image
        cv2.imshow("Palm Lines Detection", final_output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite("captured_palm_image.jpg", final_output)
        print("Palm image saved successfully.")


# Run the function
process_and_detect_palm()
