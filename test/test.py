import cv2
from PIL import Image
import numpy as np
import face_recognition

# Open image file
image = Image.open("test1.jpg")

# Convert PIL Image to NumPy array (RGB format)
image_array = np.array(image)

# Resize the image using cv2.resize
# small_frame = cv2.resize(image_array, (0, 0), fx=0.25, fy=0.25)

# Now you can proceed with OpenCV functions like cv2.cvtColor, etc.
rgb_small_frame = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
# ... rest of your code
face_locations = face_recognition.face_locations(rgb_small_frame)
face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
print(face_encodings)