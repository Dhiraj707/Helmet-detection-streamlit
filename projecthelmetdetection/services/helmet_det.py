from ultralytics import YOLO
import cv2
import numpy as np
from services.audio import play_alert  
model_path = "model/best.pt"
class HelmetDetectionPipeline():
    def __init__(self, model_path="model/best.pt"):
        self.model_path = model_path
        self.model = YOLO(self.model_path)
        self.class_names = self.model.names 

    def postprocess(self, image, results):
        annotated_image = image.copy()
        detected_classes = []

        for result in results:
            boxes = result.boxes.xyxy
            scores = result.boxes.conf
            class_ids = result.boxes.cls

            for box, score, class_id in zip(boxes, scores, class_ids):
                x1, y1, x2, y2 = map(int, box)
                label = self.class_names[int(class_id)]
                confidence = float(score)

                if confidence > 0.5:
                    detected_classes.append(label)
                    
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    cv2.putText(annotated_image, f'{label} ({confidence:.2f})', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        
        if 'nohelmet' in detected_classes:
            play_alert()

        return annotated_image, detected_classes

    def detect(self, image):
        results = self.model.predict(source=image, save=False, imgsz=640, conf=0.25, device='cpu')
        annotated_image, detected_classes = self.postprocess(image, results)
        return annotated_image, detected_classes