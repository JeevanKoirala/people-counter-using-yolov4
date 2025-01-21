import cv2
import numpy as np
import urllib.request
import os

urls = {
    "yolov4.weights": "https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4.weights",
    "yolov4.cfg": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg",
    "coco.names": "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
}



def download_model():
    for filename, url in urls.items():
        if os.path.exists(filename):
            os.remove(filename)
        try:
            urllib.request.urlretrieve(url, filename)
        except Exception as e:
            print(f"Error downloading {filename}: {e}")

download_model()

net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
layer_names = net.getLayerNames()
out_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")



def detect_people(frame):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(out_layers)
    
    boxes, confidences = [], []
    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3 and classes[class_id] == "person":
                center_x, center_y, w, h = (detection[:4] * [width, height, width, height]).astype("int")
                x, y = center_x - w // 2, center_y - h // 2
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
    
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    people_count = len(indices) if len(indices) > 0 else 0
    
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Person", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.putText(frame, f"People Count: {people_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return frame



def process_image():
    path = input("Enter image path: ")
    frame = cv2.imread(path)
    if frame is None:
        print("Error: Unable to read the image. Please check the path and try again.")
        return
    frame = detect_people(frame)
    cv2.imshow("People Counter", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def process_video():
    path = input("Enter video path: ")
    cap = cv2.VideoCapture(path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = detect_people(frame)
        cv2.imshow("People Counter", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def process_live():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = detect_people(frame)
        cv2.imshow("People Counter", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()



while True:
    print("Select an option:")
    print("1. Process Image")
    print("2. Process Video")
    print("3. Live Camera")
    print("4. Exit")
    choice = input("Enter choice: ")
    if choice == "1":
        process_image()
    elif choice == "2":
        process_video()
    elif choice == "3":
        process_live()
    elif choice == "4":
        break
    else:
        print("Invalid choice, try again.")