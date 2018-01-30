import numpy as np
import cv2, os

minimum = 75

# Create the haar cascade
cascPath = "./cascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Resize output window
cv2.namedWindow('Webcam Capture', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Webcam Capture', 640, 480)

# Video capture
cap = cv2.VideoCapture(0)

def get_images_and_labels():
    image_paths = []
    image_names = []
    path = "./faces"
    # Loop through faces directory
    for dir in os.listdir(path):
        # Loop through each person's folder
        if not dir.startswith('.'):
            for i in os.listdir(path + "/" + dir):
                # Loop through each image of the person
                if i.endswith(".jpg"):
                    image_paths.append(path + "/" + dir + "/" + i)
                    image_names.append(dir.replace("_", " "))
    images = []
    labels = []
    for index, image_path in enumerate(image_paths):
        img = cv2.imread(image_path, 0)
        faces = faceCascade.detectMultiScale(
            img,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags = cv2.CASCADE_SCALE_IMAGE
        )
        for (x, y, w, h) in faces:
            images.append(img[y: y + h, x: x + w])
            labels.append(index)
    return images, labels, image_names

images, labels, image_names = get_images_and_labels()
# Perform the training
recognizer.train(images, np.array(labels))

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    
    # Convert back to RGB
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    
    '''# Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)'''
    # Draw a rectangle around Peter's face
    for (x, y, w, h) in faces:
        nbr_predicted, conf = recognizer.predict(gray[y: y + h, x: x + w])
        accuracy = round((250 - conf) / 2.5 , 1)
        color = (0, 0, 255)
        name = str(accuracy) + "% " + image_names[nbr_predicted]
        if accuracy > minimum:
            color = (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, name, (x, y - 5), font, 0.9, color ,2, cv2.LINE_AA)
    
    # Display the resulting frame
    cv2.imshow('Webcam Capture',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
