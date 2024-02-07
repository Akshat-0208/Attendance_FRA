import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime
import os
import matplotlib.pyplot as plt


video_capture = cv2.VideoCapture(0)

Amans_image = face_recognition.load_image_file("faces/aman.jpeg")
Amans_face_encoding = face_recognition.face_encodings(Amans_image)[0]
tamatar_image = face_recognition.load_image_file("faces/tamatar.jpeg")
tamatar_image_encoding = face_recognition.face_encodings(tamatar_image)[0]
AnshG_image = face_recognition.load_image_file("faces/Screenshot 2024-01-30 134018.png")
AnshG_image_encoding = face_recognition.face_encodings(AnshG_image)[0]
Ashish_image = face_recognition.load_image_file("faces/Screenshot 2024-01-30 134052.png")
Ashish_image_encoding = face_recognition.face_encodings(Ashish_image)[0]
AnshK_image = face_recognition.load_image_file("faces/Screenshot 2024-01-30 134058.png")
AnshK_image_encoding = face_recognition.face_encodings(AnshK_image)[0]
Abdiali_image = face_recognition.load_image_file("faces/Screenshot 2024-01-30 145328.png")
Abdiali_image_encoding = face_recognition.face_encodings(Abdiali_image)[0]
Adi_image = face_recognition.load_image_file("faces/Screenshot 2024-01-30 145321.png")
Adi_image_encoding = face_recognition.face_encodings(Adi_image)[0]
known_face_encodings = [Amans_face_encoding,tamatar_image_encoding,AnshG_image_encoding,Ashish_image_encoding,AnshK_image_encoding,Abdiali_image_encoding,Adi_image_encoding]
known_face_names = ["Aman","tamatar","AnshG","Ashish","AnshK","Abdiali","Adi"]

students = known_face_names.copy()

face_locations = []
face_encodings = []


now=datetime.now()
current_date = now.strftime("%d-%m-%y")

f = open(f"{current_date}.csv","w+", newline="")
lnwriter = csv.writer(f)
lnwriter.writerow(["Name","Time"])

while True:
    _, frame = video_capture.read()
    plt.imshow(frame)
    small_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame,cv2.COLOR_BGR2RGB)
    
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame,face_locations)
    
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings,face_encoding)
        name = "Unknown"
        
        face_distances = face_recognition.face_distance(known_face_encodings,face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            
        if name in known_face_names:
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (10,100)
            fontScale = 1.5
            fontColor = (255,0,0)
            thickness = 3
            linetype = 2
            cv2.putText(frame,"Hii " +name,bottomLeftCornerOfText,font,fontScale,fontColor,thickness,linetype)
            
            if name in students:
                students.remove(name)
                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                lnwriter.writerow([name,current_time])
    
    cv2.imshow("Attendance",frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()
