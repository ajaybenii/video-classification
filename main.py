import logging

import os

import numpy as np
from PIL import Image, ImageOps
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

import moviepy.editor as mp
from keras.models import load_model
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware import Middleware
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware


origins = [
    "https://ai.propvr.tech",
    "http://ai.propvr.tech",
    "https://ai.propvr.tech/classify",
    "http://ai.propvr.tech/classify"
]

# Configure logging
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s]: %(message)s',
)

# Configure middleware to handle larger file uploads
middleware = [
    Middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"],  # Replace with your actual allowed hosts
    ),
    Middleware(
        CORSMiddleware,
        allow_origins=origins,  # Use the provided list of origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    ),
]

app = FastAPI(middleware=middleware,MAX_UPLOAD_SIZE = 100 * 1024 * 1024)


@app.get("/")
async def root():
    return "Server is up!"


class VideoClassificationResult(BaseModel):
    is_real_estate: bool


@app.post("/classify_video")
async def classify_video(video: UploadFile = File(...)):
    

    try:
        
        logging.info(f"Received video for classification: {video.filename}")
        
        # Temporary file path to save the uploaded video
        temp_video_path = "temp_video.mp4"
        
        # Save the uploaded video to a temporary file
        with open(temp_video_path, "wb") as temp_video:
            temp_video.write(video.file.read())

        # Ensure the file is properly closed before proceeding
        video.file.close()

        # Load the image classification model
        model = load_model("video_classification.h5", compile=False)
        class_names = open("labels.txt", "r").readlines()

        # Create a temporary directory to store frames as image files
        # temp_dir = "temp_frames"
        # os.mkdir(temp_dir, exist_ok=True)

        # Extract frames from the video and save them as image files
    
        real_estate = []
        non_real_estate = []

        clip = mp.VideoFileClip(temp_video_path)
        for i, frame in enumerate(clip.iter_frames(fps=1)):
            image = Image.fromarray(frame)

            # Resize and normalize the image for classification
            size = (224, 224)
            image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
            image_array = np.asarray(image)
            normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

            # Load the image into the array
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            data[0] = normalized_image_array

            # Predict the model
            prediction = model.predict(data)
            index = np.argmax(prediction)
            class_name = class_names[index]
            confidence_score = prediction[0][index]

            # Categorize frames as real estate or non-real estate
            if class_name[2:] == "Non real estate\n":
                non_real_estate.append(class_name[2:])
            else:
                real_estate.append(class_name[2:])

        # Determine the result based on frame classification
        is_real_estate = len(real_estate) > len(non_real_estate)

        # # Clean up temporary directory and video file
        # for frame_file in os.listdir(temp_dir):
        #     os.remove(os.path.join(temp_dir, frame_file))
        # os.rmdir(temp_dir)
        # Close the VideoFileClip object
        clip.close()
        os.remove(temp_video_path)
        if len(real_estate) > len(non_real_estate):
            response = "1"
            print("real_estate = ",response)
        else:
            response = "0"
            print("non_real_estate = ",response)
            
        # Logging the result
        logging.info(f"Video {video.filename} classified as {'real estate' if is_real_estate else 'non-real estate'}")
        
        return VideoClassificationResult(is_real_estate=is_real_estate)
    
    except Exception as e:
        # Log any errors
        logging.error(f"Error processing video {video.filename}: {str(e)}")
        
        return {"error": str(e)}