import cv2
import numpy as np
from sam3.model_builder import build_sam3_video_predictor

video_path = "/data3/ugq_kamal/preprocess/chunks/clip_7.mp4"
classes = ["person", "helmet"]

COLORS = {
    "person": (0, 255, 0),     
    "helmet": (255, 0, 0),      
}

def render_frame(frame, outputs, class_name, color):
    height, width = frame.shape[:2]

    masks = outputs.get("out_binary_masks", [])
    boxes = outputs.get("out_boxes_xywh", [])
    probs = outputs.get("out_probs", [])
    obj_ids = outputs.get("out_obj_ids", [])

    for i in range(len(masks)):
        mask = masks[i]
        mask = cv2.resize(mask.astype(np.uint8), (width, height))

        frame[mask == 1] = (
            int(color[0] * 0.7),
            int(color[1] * 0.7),
            int(color[2] * 0.7),
        )

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(frame, contours, -1, (0, 0, 0), 2)

        if len(boxes) > i:
            x, y, w, h = boxes[i]

            x1 = int(x * width)
            y1 = int(y * height)
            x2 = int((x + w) * width)
            y2 = int((y + h) * height)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            confidence = probs[i] if len(probs) > i else 0.0
            obj_id = obj_ids[i] if len(obj_ids) > i else i

            label = f"{class_name} | ID {obj_id} | {confidence:.2f}"

            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )

            cv2.rectangle(
                frame,
                (x1, y1 - text_h - 8),
                (x1 + text_w + 4, y1),
                color,
                -1,
            )

            cv2.putText(
                frame,
                label,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    return frame


print("Loading video info...")
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

all_class_outputs = {cls: {} for cls in classes}

for cls in classes:
    print(f"\nTracking class: {cls}")

    predictor = build_sam3_video_predictor()

    response = predictor.handle_request(
        dict(type="start_session", resource_path=video_path)
    )
    session_id = response["session_id"]

    predictor.handle_request(
        dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=0,
            text=cls,
        )
    )

    stream = predictor.handle_stream_request(
        dict(
            type="propagate_in_video",
            session_id=session_id,
            propagation_direction="forward",
            start_frame_index=0,
        )
    )

    for item in stream:
        frame_idx = item["frame_index"]
        all_class_outputs[cls][frame_idx] = item["outputs"]

    predictor.shutdown()

print("\nAll classes tracked.")

cap = cv2.VideoCapture(video_path)
out = cv2.VideoWriter(
    "tracked_multiclass_production.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height),
)

frame_idx = 0

print("Rendering final video...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    for cls in classes:
        if frame_idx in all_class_outputs[cls]:
            outputs = all_class_outputs[cls][frame_idx]
            frame = render_frame(frame, outputs, cls, COLORS[cls])

    out.write(frame)
    frame_idx += 1

cap.release()
out.release()

print("✅ Saved as tracked_multiclass_production.mp4")
