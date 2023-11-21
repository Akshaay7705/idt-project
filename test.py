import face_recognition
import cv2
import numpy as np


video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
aditya_img = face_recognition.load_image_file("aditya.jpeg")
aditya_face_encoding = face_recognition.face_encodings(aditya_img)[0]

# Load a second sample picture and learn how to recognize it.
akshaay_img = face_recognition.load_image_file("akshaay.jpg")
akshaay_face_encoding = face_recognition.face_encodings(akshaay_img)[0]

# Load a third sample picture and learn how to recognize it.
chandan_img = face_recognition.load_image_file("chandan.jpeg")
chandan_face_encoding = face_recognition.face_encodings(chandan_img)[0]

# Load a third sample picture and learn how to recognize it.
abhilash_img = face_recognition.load_image_file("abhilash.jpeg")
abhilash_face_encoding = face_recognition.face_encodings(abhilash_img)[0]


# Create arrays of known face encodings and their names
known_face_encodings = [
    aditya_face_encoding,
    akshaay_face_encoding,
    chandan_face_encoding,
    abhilash_face_encoding
]
known_face_names = [
    "Aditya",
    "Akshaay",
    "Chandan",
    "Abhilash"
]

while True:
    
    ret, frame = video_capture.read()
    rgb_frame = np.ascontiguousarray(frame[:, :, ::-1])

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        print(best_match_index,matches)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Idt Project', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()