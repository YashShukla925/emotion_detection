import cv2
import numpy as np
import tensorflow as tf

# Load the saved model
# model = tf.keras.models.load_model('C:\Users\YASH SHUKLA\OneDrive\Desktop\cv\cv1\emotional_detection.keras')
model = tf.keras.models.load_model('emotional_detection.keras')


# Load Haar Cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

def extract_features(image):
    """Preprocess the image to be suitable for the model."""
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)  # Reshape for model input
    return feature / 255.0  # Normalize

# Initialize webcam
webcam = cv2.VideoCapture(0)

# Define emotion labels
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

while True:
    ret, im = webcam.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    try:
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            cv2.rectangle(im, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Resize and preprocess the face image
            face = cv2.resize(face, (48, 48))
            img = extract_features(face)
            
            # Predict the emotion
            pred = model.predict(img)
            prediction_label = labels[pred.argmax()]
            
            # Display the prediction label on the image
            cv2.putText(im, prediction_label, (x-10, y-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
        
    except Exception as e:
        print(f"An error occurred: {e}")
    
    # Display the resulting frame
    cv2.imshow("Output", im)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
webcam.release()
cv2.destroyAllWindows()
