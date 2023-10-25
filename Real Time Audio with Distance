import numpy as np
import time
import cv2
import math
import subprocess
from gtts import gTTS
from pydub import AudioSegment
from cvzone.FaceMeshModule import FaceMeshDetector
# AudioSegment.converter = "D:/ffmpeg-2.1.1-win64-static/bin/ffmpeg.exe"
# load the COCO class labels our YOLO model was trained on
AudioSegment.converter = r"C://Users//HP//Desktop//YOLONEW//ffmpeg-2.1.1-win64-static//bin//ffmpeg.exe"

LABELS = open("coco.names").read().strip().split("\n")

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

# initialize
cap = cv2.VideoCapture(0)
detector = FaceMeshDetector(maxFaces=10)

frame_count = 0
start = time.time()
first = True
frames = []
while True:
    frame_count += 1
# Capture frame-by-frameq
    ret, frame = cap.read()
    frame, faces = detector.findFaceMesh(frame, draw=False)
    d = [-1 for i in range(10)]
    if faces:
        for j in range(len(faces)):
            face = faces[j]
            print(f"AT FACE======{j}")
            pointLeft = face[145]
            pointRight = face[374]
            # Drawing
            # cv2.line(img, pointLeft, pointRight, (0, 200, 0), 3)
            # cv2.circle(img, pointLeft, 5, (255, 0, 255), cv2.FILLED)
            # cv2.circle(img, pointRight, 5, (255, 0, 255), cv2.FILLED)
            w, _ = detector.findDistance(pointLeft, pointRight)
            W = 6.3

            # # Finding the Focal Length
            # d = 50
            # f = (w*d)/W
            # print(f)

            # Finding distance
            f = 840
            d[j] = (W * f) / w
        # print(d)
    frame = cv2.flip(frame, 1)
    cv2.imshow('frame', frame)
    frames.append(frame)

    # if frame_count == 300:
    # 	break
    if ret:
        key = cv2.waitKey(1)
        if frame_count % 60 == 0:

            end = time.time()
            # grab the frame dimensions and convert it to a blob
            (H, W) = frame.shape[:2]
            # construct a blob from the input image and then perform a forward
            # pass of the YOLO object detector, giving us our bounding boxes and
            # associated probabilities
            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                         swapRB=True, crop=False)
            net.setInput(blob)
            layerOutputs = net.forward(ln)

            # initialize our lists of detected bounding boxes, confidences, and
            # class IDs, respectively
            boxes = []
            confidences = []
            classIDs = []
            centers = []

            # loop over each of the layer outputs
            for output in layerOutputs:
                # loop over each of the detections
                for detection in output:
                    # extract the class ID and confidence (i.e., probability) of
                    # the current object detection
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]

                    # filter out weak predictions by ensuring the detected
                    # probability is greater than the minimum probability
                    if confidence > 0.5:
                        # scale the bounding box coordinates back relative to the
                        # size of the image, keeping in mind that YOLO actually
                        # returns the center (x, y)-coordinates of the bounding
                        # box followed by the boxes' width and height
                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")

                        # use the center (x, y)-coordinates to derive the top and
                        # and left corner of the bounding box
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))
                        apx_distance = round(
                            ((1 - (int(centerX + (width / 2)) - x))**4), 1)
                        print(apx_distance)
                        # update our list of bounding box coordinates, confidences,
                        # and class IDs
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)
                        centers.append((centerX, centerY))

            # apply non-maxima suppression to suppress weak, overlapping bounding
            # boxes
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

            texts = []

            # ensure at least one detection exists
            v = 0
            if len(idxs) > 0:
                # loop over the indexes we are keeping
                for i in idxs.flatten():
                    # find positions
                    centerX, centerY = centers[i][0], centers[i][1]
                    # print(i)
                    # print(centerX,centerY)
                    if centerX <= W/3:
                        W_pos = "left "
                    elif centerX <= (W/3 * 2):
                        W_pos = "center "
                    else:
                        W_pos = "right "

                    if centerY <= H/3:
                        H_pos = "top "
                    elif centerY <= (H/3 * 2):
                        H_pos = "mid "
                    else:
                        H_pos = "bottom "

                    print(d)
                    if LABELS[classIDs[i]] == "person" and d[v] != -1:
                        stringer = f"at distance {int(d[v])} with Estimated Steps = {int(math.ceil((d[v]/76)))}"
                        v += 1
                        texts.append(H_pos + W_pos +
                                     LABELS[classIDs[i]]+stringer)
                    else:
                        texts.append(H_pos + W_pos + LABELS[classIDs[i]])

                    # if(d[0]!=-1):
                    # 	d=int(d)
                    # 	stringer=f"at distance {d}"

                    # 	texts.append(H_pos + W_pos + LABELS[classIDs[i]]+stringer)
                    # else:
                    # 	texts.append(H_pos + W_pos + LABELS[classIDs[i]])

            # print(texts)

            if texts:
                description = ', '.join(texts)
                tts = gTTS(description, lang='en')
                tts.save('tts.mp3')
                tts = AudioSegment.from_mp3("tts.mp3")
                subprocess.call(["ffplay", "-nodisp", "-autoexit", "tts.mp3"])


cap.release()
cv2.destroyAllWindows()
# os.remove("tts.mp3")
