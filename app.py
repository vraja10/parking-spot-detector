import os
import gradio as gr
import cv2
import numpy as np
import pickle
from skimage.transform import resize
import tempfile

MODEL_PATH = "./model/model2.p"
MASK_PATH = "./mask_1920_1080.png"

with open(MODEL_PATH, "rb") as f:
    MODEL = pickle.load(f)

mask = cv2.imread(MASK_PATH, 0)
connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)

def get_parking_spots_bboxes(components):
    (totalLabels, label_ids, values, centroid) = components
    spots = []
    for i in range(1, totalLabels):  # 0 = background
        x1 = int(values[i, cv2.CC_STAT_LEFT])
        y1 = int(values[i, cv2.CC_STAT_TOP])
        w = int(values[i, cv2.CC_STAT_WIDTH])
        h = int(values[i, cv2.CC_STAT_HEIGHT])
        spots.append([x1, y1, w, h])
    return spots

spots = get_parking_spots_bboxes(connected_components)

def empty_or_not(spot_bgr):
    resized = resize(spot_bgr, (15, 15, 3))
    flat = resized.flatten().reshape(1, -1)
    prediction = MODEL.predict(flat)
    return prediction[0] == 0

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    
    temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_out.name, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        spots_status = []
        for spot in spots:
            x1, y1, w, h = spot
            crop = frame[y1:y1+h, x1:x1+w]
            if crop.size == 0:
                spots_status.append(False)
                continue
            status = empty_or_not(crop)
            spots_status.append(status)

        for i, spot in enumerate(spots):
            x1, y1, w, h = spot
            color = (0, 255, 0) if spots_status[i] else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x1+w, y1+h), color, 2)

        cv2.rectangle(frame, (80, 20), (550, 80), (0, 0, 0), -1)
        text = f"Available spots: {sum(spots_status)} / {len(spots_status)}"
        cv2.putText(frame, text, (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        out.write(frame)

    cap.release()
    out.release()

    return temp_out.name  

iface = gr.Interface(
    fn=process_video,
    inputs=gr.Video(label="Please upload your parking spot area video in .mp4 format"),
    outputs=gr.Video(label="Processed Video"),
    title="Parking Slot Detecter and Counter with OpenCV",
    description="Parking Slots will be colored by green (if it is empty) or colored by red (if it is not empty). Also you will be able to see the counter for avaiable slots in the top-left of the screen."
)

if __name__ == "__main__":
    iface.launch()
