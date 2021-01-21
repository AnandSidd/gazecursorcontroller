from mtcnn import MTCNN
import cv2

fd = MTCNN()


def facedetection(frame):
    frame = cv2.resize(frame, (600, 400))
    box = fd.detect_faces(frame)
    head_image = []
    if box:
        b = box[0]['box']
        conf = box[0]['confidence']
        x, y, w, h = b[0], b[1], b[2], b[3]
        head_image = frame[x:x+w, y:y+h]
        if conf > 0.5:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

    return box, head_image, frame

def main():
    cap = cv2.VideoCapture(0)
    while (True):
        ret, frame = cap.read()
        box, head, detectframe = facedetection(frame)
        cv2.imshow("Frame", detectframe)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
