import cv2

cap = cv2.VideoCapture(0)

degree = 0

def onChange(pos):
    global degree
    degree = pos

cv2.namedWindow("frame_rotate")
cv2.createTrackbar("rotate degree","frame_rotate", 0, 359, onChange)

if cap.isOpened():
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    dt = int(1000 // fps)
    center = (int(w/2), int(h/2))

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        m_rotate = cv2.getRotationMatrix2D(center, degree,1)
        frame_rotate = cv2.warpAffine(frame, m_rotate,(int(w), int(h)))

        cv2.imshow("frame", frame)
        cv2.imshow("frame_rotate", frame_rotate)

        if cv2.waitKey(dt) != -1:
            break

    