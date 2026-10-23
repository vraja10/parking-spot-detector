import os
import numpy as np
import cv2
from skimage.transform import resize
from sklearn.ensemble import RandomForestClassifier
import pickle
from tqdm import tqdm

# data path
data_dir = './clf-data'
categories = ['empty', 'not_empty']  # 0 = empty, 1 = notempty

X = []
y = []

# uploading images and taking them into the process
for label, category in enumerate(categories):
    folder_path = os.path.join(data_dir, category)
    for file in tqdm(os.listdir(folder_path), desc=f"{category}"):
        try:
            img_path = os.path.join(folder_path, file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img_resized = resize(img, (15, 15, 3))  # expectted model format
            flat_data = img_resized.flatten()
            X.append(flat_data)
            y.append(label)
        except Exception as e:
            print(f"Hata {file} dosyasında: {e}")

X = np.array(X)
y = np.array(y)

# training our model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# and then saving it
os.makedirs("model", exist_ok=True)
with open("./model/model2.p", "wb") as f:
    pickle.dump(model, f)

