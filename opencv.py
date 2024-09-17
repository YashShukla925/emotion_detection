import cv2

# Initialize the webcam
webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print("Error: Could not open webcam.")
else:
    print("Webcam opened successfully.")

# Capture a single frame
ret, frame = webcam.read()

if not ret:
    print("Failed to grab frame")
else:
    # Save the captured image
    cv2.imwrite('captured_image.jpg', frame)
    print("Image captured and saved.")

# Release the webcam and close windows
webcam.release()
cv2.destroyAllWindows()
