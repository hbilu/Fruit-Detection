import cv2
from ultralytics import YOLO

def assign_color(label):
    color_map = {
        'Apel': (0, 0, 255),     # Red
        'Banann': (0, 255, 255), # Yellow
        'Limett': (0, 255, 0),   # Green
        'Zitroun': (0, 165, 255) # Orange
    }
    return color_map.get(label, (255, 255, 255))
        
def predict(model, img, classes=[], conf=0.8, rectangle=4, text=3): #color=(0, 255, 0) removed

    # perform prediction
    results = model.predict(img, classes=classes, conf=conf) if classes else model.predict(img, conf=conf)

    # visualize detections as captured in rectangle boxes with their class names
    for result in results:
        for box in result.boxes:
            label = f"{result.names[int(box.cls[0])]}"
            score = box.conf.item()
            color = assign_color(label)
            cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])), (int(box.xyxy[0][2]), int(box.xyxy[0][3])), 
                          color=color, thickness=rectangle)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, f"{label} ({score:.2f})", (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10), font, 1, color=color, thickness=text)
    return img, results

def main_func():
    
    # choose YOLO model
    model = YOLO("/Users/hacerbilu/Documents/Fruit-Detection/yolov9/runs/train/exp8/weights/best.pt")

    camera_live = cv2.VideoCapture(1)
    camera_live.set(3, 1280)
    camera_live.set(4, 720)

    while camera_live.isOpened():
        read, frame = camera_live.read()
        if not read:
            continue

        # object detection
        result_img, _ = predict(model, frame)
        # display detected objects 
        cv2.imshow('YOLOv9 Fruit Detection', result_img)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    camera_live.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main_func()