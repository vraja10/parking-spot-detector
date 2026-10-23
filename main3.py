import cv2
import numpy as np
import os
import random
from util import get_parking_spots_bboxes, empty_or_not

MASK_PATH = './mask_1920_1080.png'
FRAMES_DIR = './frames'
OUTPUT_PATH = './output_single.png'

# Maskeyi yükle
mask = cv2.imread(MASK_PATH, 0)
connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
spots = get_parking_spots_bboxes(connected_components)

# Frames klasöründen 1 tane random fotoğraf seç
frame_files = [f for f in os.listdir(FRAMES_DIR) if f.endswith(('.png', '.jpg'))]
if not frame_files:
    raise Exception("Frames klasöründe uygun görsel bulunamadı.")

frame_file = random.choice(frame_files)
frame_path = os.path.join(FRAMES_DIR, frame_file)
frame = cv2.imread(frame_path)

if frame is None:
    raise Exception("Fotoğraf yüklenemedi.")

# Her park yeri için boş/dolu kontrolü yap
spots_status = []
for spot in spots:
    x1, y1, w, h = spot
    spot_crop = frame[y1:y1+h, x1:x1+w, :]
    status = empty_or_not(spot_crop)
    spots_status.append(status)

# Görsel üzerine kutuları çiz
for spot_indx, spot in enumerate(spots):
    x1, y1, w, h = spot
    color = (0, 255, 0) if spots_status[spot_indx] else (0, 0, 255)
    frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), color, 2)

# Bilgi kutusu
cv2.rectangle(frame, (80, 20), (550, 80), (0, 0, 0), -1)
cv2.putText(frame, f'Available spots: {sum(spots_status)} / {len(spots_status)}',
            (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# Sonucu kaydet
cv2.imwrite(OUTPUT_PATH, frame)
print(f"İşlenmiş görsel kaydedildi: {OUTPUT_PATH}")
