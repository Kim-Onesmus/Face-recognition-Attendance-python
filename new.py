import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import time

time_spend = time.time() + 30
path = 'images'
images = []
classNames = []
myList = os.listdir(path)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

def findEncondings(images):
    encodeList = []
    for img in images:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model='hog')
        encodings = face_recognition.face_encodings(rgb, boxes)
        for encoding in encodings:
            encodeList.append(encoding)
    return encodeList

def markAttendance(name):
    with open('attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            today = datetime.today()
            tdString = today.strftime('%d:%m:%Y')
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{tdString},{dtString}')

encodeListKnown = findEncondings(images)

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    cv2.imshow('Class Attendance', img)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb, model='hog')
    encodings = face_recognition.face_encodings(rgb, boxes)

    for encoding, box in zip(encodings, boxes):
        matches = face_recognition.compare_faces(encodeListKnown, encoding, tolerance=0.4)
        face_distances = face_recognition.face_distance(encodeListKnown, encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = classNames[best_match_index]
            markAttendance(name)
            y1, x2, y2, x1 = box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Class Attendance', img)
    if cv2.waitKey(1) == ord('q'):
        break
    if time.time() > time_spend:
        break

cap.release()
cv2.destroyAllWindows()
