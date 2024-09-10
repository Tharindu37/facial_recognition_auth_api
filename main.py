from typing import Union

from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import face_recognition
from PIL import Image
import io
import cv2
import numpy as np
app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}

# In-memory database of known faces
known_faces = {}
known_face_names = []

class RegisterFace(BaseModel):
    name: str
    file: UploadFile = File(...)

# @app.post("/register_face/")
# # async def register_face(face: RegisterFace, file: UploadFile = File(...)):
# async def register_face(face: RegisterFace):
#     try:
#         # Load the image file
#         # image_data = await file.read()
#         image_data = await face.file.read()
#         image = Image.open(io.BytesIO(image_data))
#         image_np = face_recognition.load_image_file(io.BytesIO(image_data))

#         # Encode the face
#         face_encoding = face_recognition.face_encodings(image_np)
#         if len(face_encoding) == 0:
#             raise HTTPException(status_code=400, detail="No face found in the image")
        
#         known_faces[face.name] = face_encoding[0]
#         known_face_names.append(face.name)
#         return {"message": "Face registered successfully"}
    
#     except Exception as e:
#         return HTTPException(status_code=500, detail=str(e))

@app.post("/register_face/")
async def register_face(name: str = Form(...), file: UploadFile = File(...)):
    try:
        # Read the image file
        image_data = await file.read()
        print('done')
        # Convert image data to a PIL Image
        image = Image.open(io.BytesIO(image_data))
        image_array = np.array(image)
        
        rgb_small_frame = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        # face_encodings = face_recognition.face_encodings(rgb_small_frame)[0]
        # Load the image using face_recognition
        # know_image = face_recognition.load_image_file("test/test1_rgb.jpg")

        # Find face encodings
        # face_encodings = face_recognition.face_encodings(know_image)
        if not face_encodings:
            raise HTTPException(status_code=400, detail="No face found in the image")
        
        # # Store the face encoding
        known_faces[name] = face_encodings[0]
        known_face_names.append(name)
        return {"message": "Face registered successfully"}
    
    except Exception as e:
        # Log the exception for debugging
        print(f"Error occurred: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.post("/authenticate/")
async def authenticate(file: UploadFile = File(...)):
    try:
        # Load the image file
        image_data = await file.read()
        # image = Image.open(io.BytesIO(image_data))
        # image_np = face_recognition.load_image_file(io.BytesIO(image_data))
        
        image = Image.open(io.BytesIO(image_data))
        image_array = np.array(image)
        
        rgb_small_frame = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        unknown_face_encoding = face_recognition.face_encodings(rgb_small_frame, face_locations)

        # Encode the face
        # unknown_face_encoding = face_recognition.face_encodings(image_np)
        if len(unknown_face_encoding) == 0:
            raise HTTPException(status_code=400, detail="No face found in the image")
        
        # Compare with known faces
        matches = face_recognition.compare_faces(list(known_faces.values()), unknown_face_encoding[0])

        # Check if a match is found
        if True in matches:
            matched_name = known_face_names[matches.index(True)]
            return {"authenticated_as": matched_name}
        else:
            return {"authenticated_as": "Unknown"}

    except Exception as e:
        return HTTPException(status_code=500, detail=str(e))