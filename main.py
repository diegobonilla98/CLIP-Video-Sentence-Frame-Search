import time

import torch
import clip
from PIL import Image
import cv2


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)

text_query = input("What were you looking for? ")  # "a firetruck driving down the street"  # "a man with a white shirt and an orange box walking down the street"
text = clip.tokenize([text_query]).to(device)

video_path = './footage1.mp4'
video = cv2.VideoCapture(video_path)

every_seconds = 1.5
last_time = None
probs = 99

with torch.no_grad():
    while True:
        ret, frame = video.read()

        if not ret:
            break

        if last_time is None or time.time() - last_time > every_seconds:
            last_time = time.time()
            image = preprocess(Image.fromarray(frame[:, :, ::-1])).unsqueeze(0).to(device)
            logits_per_image, _ = model(image, text)
            probs = logits_per_image.cpu().numpy()

        if probs[0][0] > 20:
            print("We have a match")
            cv2.imshow("Match", frame)
            cv2.waitKey()

        # print(f"Label probs: {probs}")
        # cv2.imshow("Result", frame)
        # key = cv2.waitKey(1)
        # if key == ord('q'):
        #     break

cv2.destroyAllWindows()
video.release()
