import cv2
import numpy as np
import face_recognition


Kim_Onesmus = face_recognition.load_image_file('images/CSC_027_2020.jpg')
Kim_Onesmus = cv2.cvtColor(Kim_Onesmus, cv2.COLOR_BGR2RGB)

Kim = face_recognition.load_image_file('images/kim.jpg')
Kim = cv2.cvtColor(Kim, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(Kim_Onesmus)[0]
encodeKima = face_recognition.face_encodings(Kim_Onesmus)[0]
cv2.rectangle(Kim_Onesmus,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255), 2)

faceLocKim = face_recognition.face_locations(Kim)[0]
encodeKim = face_recognition.face_encodings(Kim)[0]
cv2.rectangle(Kim,(faceLocKim[3],faceLocKim[0]),(faceLocKim[1],faceLocKim[2]),(255,0,255), 2)

results = face_recognition.compare_faces([encodeKima], encodeKim)
faceDis = face_recognition.face_distance([encodeKima],encodeKim)
print (results, faceDis)
cv2.putText(Kim, f'{results} {round(faceDis[0],2)}', (50,50), cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255), 2)

cv2.imshow('Kim Onesmus', Kim_Onesmus)
cv2.imshow('Kim Onesmus1', Kim)

cv2.waitKey(0)
  