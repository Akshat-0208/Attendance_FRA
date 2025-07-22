import streamlit as st
import face_recognition
import cv2
import os
import numpy as np
from datetime import datetime
import pandas as pd

def load_known_faces(faces_dir):
    encodings, names = [], []
    for filename in os.listdir(faces_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(faces_dir, filename)
            image = face_recognition.load_image_file(path)
            faces = face_recognition.face_encodings(image)
            if faces:
                encodings.append(faces[0])
                names.append(os.path.splitext(filename)[0])
    return encodings, names


st.title("Face Recognition Attendance System")

faces_dir = "faces"
attendance_dir = "attendance"
os.makedirs(attendance_dir, exist_ok=True)
known_encodings, known_names = load_known_faces(faces_dir)
students = set(known_names)

# Start camera
run_recognition = st.button("Start Face Recognition")
attendance = {}

if run_recognition:
    st.info("Starting webcam... Press 'q' in video window to stop.")
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb_small_frame = cv2.cvtColor(cv2.resize(frame, (0,0), fx=0.25, fy=0.25), cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for enc in face_encodings:
            matches = face_recognition.compare_faces(known_encodings, enc)
            name = "Unknown"
            if matches:
                face_distances = face_recognition.face_distance(known_encodings, enc)
                best_match = np.argmin(face_distances)
                if matches[best_match]:
                    name = known_names[best_match]
            # Attendance marking
            if name != "Unknown" and name not in attendance:
                attendance[name] = datetime.now().strftime("%H:%M:%S")
            # Draw name on frame
            for (top, right, bottom, left) in face_locations:
                top, right, bottom, left = top*4, right*4, bottom*4, left*4
                cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
                cv2.putText(frame, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,255), 2)

        cv2.imshow("Face Recognition Attendance - Press 'q' to exit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    st.success("Recognition session ended.")

if attendance:
    date_str = datetime.now().strftime("%d-%m-%y")
    csv_path = os.path.join(attendance_dir, f"{date_str}.csv")
    df = pd.DataFrame([(name, time) for name, time in attendance.items()], columns=["Name", "Time"])
    df.to_csv(csv_path, index=False)
    st.write("## Today's Attendance")
    st.dataframe(df)
    st.download_button("Download Attendance CSV", df.to_csv(index=False), file_name=f"{date_str}.csv")
