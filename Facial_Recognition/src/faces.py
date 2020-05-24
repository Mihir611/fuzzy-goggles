import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_smile.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()#LBPH face recognizer
recognizer.read("trainer.yml")#reeading the yml (trained file)

labels = {"person_name":1}
with open("labels.pickle",'rb') as f:
	ge_labels = pickle.load(f)
	labels = {v:k for k,v in ge_labels.items()}#inverting the label values

cap = cv2.VideoCapture(0)

while True:
	#capture frame by frame
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors = 5)
	for (x, y, w, h) in faces:
		#print(x,y,w,h)
		#region of intrest
		roi_gray = gray[y:y+h, x:x+w]#gray region
		roi_color = frame[y:y+h, x:x+w]
		#saving the image
		#recognitizing the face eg tencerflow,pytorch,etc..(Algo)
		id_, conf = recognizer.predict(roi_gray)
		if conf >= 4 and conf <= 85:
			#print(id_)
			#print(labels[id_])
			#putting the text on the image
			font = cv2.FONT_HERSHEY_SIMPLEX
			name = labels[id_]
			color = (0, 255, 0)
			stroke = 2
			cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)

		img_item = "my-image.png"#saving image
		cv2.imwrite(img_item, roi_gray)
		#drawing rectangle
		color =  (0, 255, 0)#BGR Rectangle color
		stroke = 2 #thickness
		end_cord_x = x + w
		end_cord_y = y + h
		cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
		eyes = eye_cascade.detectMultiScale(roi_gray)
		for (ex,ey,ew,eh) in eyes:
			cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
		smiles = smile_cascade.detectMultiScale(roi_gray)
		for (ex,ey,ew,eh) in smiles:
			cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
	#Display resulting Frame
	cv2.imshow('frame', frame)
	if cv2.waitKey(20) & 0xFF == ord('q'):
		break

#when everything is done
cap.release()
cap.destroyAllWindows()