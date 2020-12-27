import cv2
import numpy as np
import face_recognition
import os

path = 'ImageAttendance'
images = []
classList = []
classNames = []
classId = []
myList = os.listdir(path)
print(myList)
for cls in myList:
    curImg = cv2.imread(f'{path}/{cls}')
    images.append(curImg)
    classList.append(os.path.splitext(cls)[0])
for i in classList:
    classId.append(i.split('_')[0])
    classNames.append(i.split('_')[1])
print(classNames, classId)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeLisKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS,faceCurFrame)

    for encodeFace,faceLoc in zip(encodeCurFrame,faceCurFrame):
        matches = face_recognition.compare_faces(encodeLisKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeLisKnown,encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            id = classId[matchIndex]
            name = classNames[matchIndex]
            print(id,name)
