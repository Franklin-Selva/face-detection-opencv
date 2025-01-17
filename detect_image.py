import numpy as np
import cv2

# Models
protxt = "./models/prototxt.txt"
model = "./models/res10_300x300_ssd_iter_140000.caffemodel"

# Parameters
CONFIDENCE = 0.5

if __name__ == "__main__":
	net = cv2.dnn.readNetFromCaffe(protxt, model)
	print("Model has been loaded")

	image = cv2.imread("./src/elon musk.jpg")
	(h, w) = image.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the detections and predictions
	print("Computing face detections...")
	net.setInput(blob)
	detections = net.forward()

	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]

		if confidence > CONFIDENCE:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
	
			text = "{:.2f}%".format(confidence * 100)
			y = startY - 10 if startY - 10 > 10 else startY + 10
			cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
			cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

	cv2.imshow("Result image", image)
	cv2.waitKey(0)