from mtcnn import MTCNN
import cv2

fd = MTCNN()


def facedetection(frame):
    box = fd.detect_faces(frame)
    print(box)
    head_image = []
    if box:
        b = box[0]['box']
        conf = box[0]['confidence']
        x, y, w, h = b[0], b[1], b[2], b[3]
        l_e = box[0]['keypoints']['left_eye']
        r_e = box[0]['keypoints']['right_eye']
        nose = box[0]['keypoints']['nose']
        if conf > 0.5:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            head_image = frame[y:y+h,x:x+w]
            cv2.circle(frame, nose, 2, (0, 255, 0), thickness=5, lineType=8, shift=0)

            cv2.rectangle(frame, (int(r_e[0]) - 30, int(r_e[1]) - 30),
                          (int(r_e[0]) + 30, int(r_e[1]) + 30), (255, 255, 255), 2)
            cv2.rectangle(frame, (int(l_e[0]) - 30, int(l_e[1]) - 30),
                          (int(l_e[0]) + 30, int(l_e[1]) + 30), (255, 255, 255), 2)

    return box, head_image, frame

def main():
    cap = cv2.VideoCapture(0)
    while (True):
        ret, frame = cap.read()
        frame = cv2.resize(frame, (600, 400))
        box, head, detectframe = facedetection(frame)
        if len(head)!=0:
            cv2.imshow("Frame", detectframe)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
