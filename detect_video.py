from imutils.video import VideoStream
import numpy as np
import time
import cv2

# Models
protxt = "./models/prototxt.txt"
model = "./models/res10_300x300_ssd_iter_140000.caffemodel"

# Parameters
CONFIDENCE = 0.5

if __name__ == "__main__":
	net = cv2.dnn.readNetFromCaffe(protxt, model)
	print("Model has been loaded")

	print("Starting webcam")
	cap = cv2.VideoCapture(0)
	# time.sleep(2.0)

	while True:
		_, frame = cap.read()
		# frame = cv2.resize(frame, (frame.size().height, 400), interpolation= cv2.INTER_CUBIC)

		h, w, channels = frame.shape
		blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
	
		# Setup blob
		net.setInput(blob)
		detections = net.forward()

		for i in range(0, detections.shape[2]):
			confidence = detections[0, 0, i, 2]

			if confidence < CONFIDENCE:
				continue

			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
	
			text = "{:.2f}%".format(confidence * 100)
			y = startY - 10 if startY - 10 > 10 else startY + 10

			cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
			cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

		cv2.imshow("Detection Result", frame)
		key = cv2.waitKey(1) & 0xFF
	
		if key == ord("q"):
			break

	cv2.destroyAllWindows()
	cap.release()