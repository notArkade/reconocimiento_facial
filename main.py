import face_recognition
import cv2
import os
from matplotlib import pyplot as plt
from PIL import Image

def read_photo(path):
    photo = cv2.imread(path)
    if photo is None:
        return None
    (h, w) = photo.shape[:2]
    width = 500
    ratio = width / float(w)
    height = int(h * ratio)
    return cv2.resize(photo, (width, height))

def show_image(image, window_name="Image"):

    cv2.imshow(window_name, image)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()

known_encodings = []
known_data = []
data = 'dataset'

for file in os.listdir(data):
    photo = read_photo(data + '/' + file)
    if photo is not None: 
        photo_enc = face_recognition.face_encodings(photo)[0]
        known_encodings.append(photo_enc)
        known_data.append(file.split('.')[0])

sample = 'sample_images'
for file in os.listdir(sample):
    photo = read_photo(sample + '/' + file)
    if photo is not None:
        photo_enc = face_recognition.face_encodings(photo)[0]

        results = face_recognition.compare_faces(known_encodings, photo_enc)

        for i in range(len(results)):
            if results[i]:
                name = known_data[i]
                (top, right, bottom, left) = face_recognition.face_locations(photo)[0]
                cv2.rectangle(photo, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(photo, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
                show_image(photo)  
