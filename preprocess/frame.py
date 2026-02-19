import cv2
import numpy as np
import os

video_path = "/data3/ugq_kamal/preprocess/fixed.mp4"
output_dir = "/data3/ugq_kamal/preprocess/chunks"
os.makedirs(output_dir, exist_ok=True)

x_start, y_start = 0, 32
x_end, y_end = 563, 71

motion_area_threshold = 3000
no_motion_tolerance_seconds = 10
padding_seconds = 2              

cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

backSub = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=25,
    detectShadows=False
)

frame_buffer = []
buffer_size = fps * padding_seconds
no_motion_limit = fps * no_motion_tolerance_seconds

recording = False
no_motion_counter = 0
clip_id = 0
out = None

print("Processing video...")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame[y_start:y_end, x_start:x_end] = 0

    fgMask = backSub.apply(frame)
    kernel = np.ones((5,5), np.uint8)
    fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)
    fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_DILATE, kernel)

    contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    motion_detected = False
    for cnt in contours:
        if cv2.contourArea(cnt) > motion_area_threshold:
            motion_detected = True
            break
    frame_buffer.append(frame.copy())
    if len(frame_buffer) > buffer_size:
        frame_buffer.pop(0)

    if motion_detected:

        if not recording:
            print(f"Starting clip {clip_id}")
            output_path = os.path.join(output_dir, f"clip_{clip_id}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            for bf in frame_buffer:
                out.write(bf)

            recording = True

        out.write(frame)
        no_motion_counter = 0

    else:
        if recording:
            out.write(frame)
            no_motion_counter += 1

            if no_motion_counter > no_motion_limit:
                print(f"Ending clip {clip_id}")
                out.release()
                recording = False
                clip_id += 1
                no_motion_counter = 0

cap.release()

if recording:
    out.release()

print(f"\nSaved {clip_id} motion clips.")
