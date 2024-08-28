from deepface import DeepFace
import cv2

# Load the image (replace with the path to your image)
img_path = "test_image.jpg"
img = cv2.imread(img_path)

# Recognize the face using DeepFace
# You can use different models like VGG-Face, Facenet, OpenFace, DeepID, or Dlib.
model_name = 'VGG-Face'  # You can change this to 'Facenet', 'OpenFace', etc.
result = DeepFace.analyze(img_path=img_path, actions=['age', 'gender'], enforce_detection=False)

# Print the results
for face in result:
    age = face['age']
    gender = face['dominant_gender']
    print(f"Age: {age}, Gender: {gender}")


