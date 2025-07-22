# Face Recognition Attendance System - Attendance_FRA

This project is a dynamic, camera-based attendance system powered by face recognition and a modern web interface using Streamlit. It automatically logs attendance by matching faces captured from a webcam to images stored in a designated directory. The application is designed for easy enrollment, usability, and quick access to attendance records.

## Features

- **Real-time Face Recognition**: Streamlit interface detects attendees from your webcam.
- **Dynamic Enrollment**: Add new students by simply uploading their image to a folder.
- **Attendance Logging**: Saves daily attendance records in dated CSV files.
- **User-Friendly Web App**: No need for command line interaction.
- **Downloadable Reports**: Table view and instant CSV download of each day’s attendance.

## Directory Structure

```plaintext
project_root/
├── faces/              # Store enrolled face images here (e.g., aman.jpeg, jane.png)
├── attendance/         # All CSV attendance logs
├── app.py              # Streamlit interface script
```

## Setup Instructions

### 1. Install Dependencies

Ensure you have Python 3.8+ installed. Then run:

```bash
pip install streamlit face_recognition opencv-python-headless pandas numpy
```

**Note**: For `face_recognition`, you may need to install additional system dependencies, such as CMake, dlib, and OpenCV. Refer to the [face_recognition installation guide](https://github.com/ageitgey/face_recognition#installation) if you encounter problems.

### 2. Prepare Face Images

- Place clear frontal face images in the `faces/` directory.
- File names (e.g., `aman.jpeg`) will be used as the recognized person’s name.

### 3. Run the Application

Open your terminal in the project folder and execute:

```bash
streamlit run app.py
```

- The Streamlit app will open in your browser.
- Click "Start Face Recognition" to begin webcam-based attendance.

### 4. Attendance Logging

- Attendance for each recognized person is marked once per session.
- Once finished, the day’s attendance is available as a CSV to download.

## Usage

- Enroll students: Place images into `faces/`.
- Start face recognition: Use the web UI.
- View attendance: Current session’s results are displayed.
- Download CSV: Use the "Download Attendance CSV" button.

## Sample `app.py` Snippet

```python
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
```

## Tips

- Use clear, well-lit, single-headshot images for enrollment.
- The Streamlit app only marks attendance once per session for each person.
- For remote use or bulk uploads, extend the app with upload and admin access features.

## Credits

- [face_recognition library](https://github.com/ageitgey/face_recognition)
- [Streamlit](https://streamlit.io/)
