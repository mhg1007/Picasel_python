from flask import Flask, request, jsonify
from ultralytics import YOLO
import requests
from PIL import Image
from io import BytesIO

app = Flask(__name__)

# Load ONNX model
model = YOLO("./best.pt")

@app.route('/object_detection', methods=['POST'])
def object_detection():
    data = request.get_json()
    image_url = data.get("url")

    try:
        # 이미지 다운로드
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content)).convert("RGB")

        # 객체 인식
        results = model.predict(image, imgsz=1280)

        result_list = []
        for result in results:
            boxes = result.boxes
            orig_w, orig_h = result.orig_shape[1], result.orig_shape[0]

            for box in boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = round(float(box.conf[0]), 4)

                # Bounding box 좌표
                x1, y1, x2, y2 = box.xyxy[0]
                bbox_area = (x2 - x1) * (y2 - y1)
                image_area = orig_w * orig_h
                bbox_ratio = round(float(bbox_area / image_area), 4)

                result_list.append({
                    "class": class_name,
                    "confidence": confidence,
                    "bbox_ratio": bbox_ratio
                })

        return jsonify(result_list)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
