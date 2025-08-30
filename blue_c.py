import cv2
import numpy as np
import time

def color_detection_and_tracking_console(video_source=0):
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    # Define the lower and upper bounds for the blue color in HSV
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])

    prev_frame_time = 0
    new_frame_time = 0

    print("Starting blue object detection. Press 'Ctrl+C' to stop.")
    print("-" * 50)
    print(f"{'FPS':<10} | {'Center X':<10} | {'Center Y':<10} | {'Object Detected'}")
    print("-" * 50)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame. Exiting.")
                break

            # Convert the frame to HSV color space
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Create a mask for the blue color
            mask = cv2.inRange(hsv, lower_blue, upper_blue)

            # Apply morphological operations to remove small noise and fill gaps
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)

            # Find contours in the mask
            contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            center_x, center_y = "N/A", "N/A"
            object_detected_status = "No"

            if len(contours) > 0:
                # Find the largest contour
                largest_contour = max(contours, key=cv2.contourArea)

                # Only proceed if the area is large enough to be a significant object
                if cv2.contourArea(largest_contour) > 500:  # You can adjust this threshold
                    # Calculate the moments of the contour to find the center
                    M = cv2.moments(largest_contour)
                    if M["m00"] != 0:
                        center_x = int(M["m10"] / M["m00"])
                        center_y = int(M["m01"] / M["m00"])
                        object_detected_status = "Yes"

            # Calculate FPS
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time) if (new_frame_time - prev_frame_time) > 0 else 0
            prev_frame_time = new_frame_time

            # Print to console
            print(f"{int(fps):<10} | {str(center_x):<10} | {str(center_y):<10} | {object_detected_status}")

            # Introduce a small delay to prevent overwhelming the console and CPU
            # Adjust as needed, e.g., 0.03 for ~30 FPS output
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nStopping detection due to user interrupt (Ctrl+C).")
    finally:
        cap.release()
        print("-" * 50)
        print("Detection stopped and resources released.")

if __name__ == "__main__":
    color_detection_and_tracking_console(0) # Use 0 for webcam
