# All the imports go here
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import copy
from PIL import Image, ImageChops



def compare_images(draw_img : str, template : str) -> float:
    """
    Сравнение рисунка с шаблоном.
    Вывод информации, на сколько совпадает
    рисунок с шаблоном
    :param draw_img: путь на рисунок
    :param template: путь на шаблон
    :return: на сколько рисунок похож на шаблон
    """

    # Загрузка изображения рисунка и шаблона
    image = cv2.imread(draw_img)  # Путь к вашему изображению рисунка
    template = cv2.imread(template)  # Путь к вашему изображению шаблона

    # Преобразование изображений в оттенки серого
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # Создание маски изображения рисунка и шаблона
    _, drawn_mask = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY_INV)
    _, template_mask = cv2.threshold(gray_template, 50, 255, cv2.THRESH_BINARY_INV)

    # Проверка пересечения масок
    intersection_mask = np.logical_and(drawn_mask, template_mask)

    # Получение количества пикселей пересечения
    intersection_pixels_count = np.count_nonzero(intersection_mask)

    # Получение количества пикселей в нарисованной маске
    drawn_pixels_count = np.count_nonzero(drawn_mask)

    if drawn_pixels_count == 0:
        return 0
    return round((intersection_pixels_count / drawn_pixels_count) * 100, 2)


# Предоставление разных массивов для обработки цветовых точек разного цвета
bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
ypoints = [deque(maxlen=1024)]


# Эти индексы будут использоваться для обозначения точек в определенных массивах определенного цвета
blue_index = 0
green_index = 0
red_index = 0
yellow_index = 0


colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
colorIndex = 0

# Here is code for Canvas setup
paintWindow = np.zeros((471,636,3)) + 255
paintWindow = cv2.rectangle(paintWindow, (40,1), (140,65), (0,0,0), 2)
paintWindow = cv2.rectangle(paintWindow, (160,1), (255,65), (255,0,0), 2)
paintWindow = cv2.rectangle(paintWindow, (275,1), (370,65), (0,255,0), 2)
paintWindow = cv2.rectangle(paintWindow, (390,1), (485,65), (0,0,255), 2)
paintWindow = cv2.rectangle(paintWindow, (505,1), (600,65), (0,255,255), 2)


cv2.putText(paintWindow, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)


# Дополнительно копируем чисто окно
paintWindow_pred = copy.deepcopy(paintWindow)
paintWindow_first = copy.deepcopy(paintWindow)

# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils


camera_id = int(input('Введите номер камеры'))
# Initialize the webcam
cap = cv2.VideoCapture(camera_id)
ret = True

# custom logic
# флаг для сохранение рисунка в файл
flag_save_img = False

# флаг для вывода точек домика
flag_house = False

# флаги для работы с рыбкой
flag_fish = False
flag_fish2 = True
flag_fish_save = False

# флаги для работы с SimbirSoft
flag_simbir = False
flag_simbir2 = True
flag_simbir_save = False

count = 0

thickness = 4

while ret:
    # Read each frame from the webcam
    ret, frame = cap.read()

    x, y, c = frame.shape

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    #hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame = cv2.rectangle(frame, (40,1), (140,65), (0,0,0), 2)
    frame = cv2.rectangle(frame, (160,1), (255,65), (255,0,0), 2)
    frame = cv2.rectangle(frame, (275,1), (370,65), (255,0,0), 2)
    frame = cv2.rectangle(frame, (390,1), (485,65), (255,0,0), 2)
    frame = cv2.rectangle(frame, (505,1), (600,65), (255,0,0), 2)
    cv2.putText(frame, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "Check", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "SimbirSoft", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "Fish", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "House", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    #frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Get hand landmark prediction
    result = hands.process(framergb)
    center = None
    # post process the result
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                # # print(id, lm)
                # print(lm.x)
                # print(lm.y)
                lmx = int(lm.x * 640)
                lmy = int(lm.y * 480)

                landmarks.append([lmx, lmy])


            # Drawing landmarks on frames
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
        fore_finger = (landmarks[8][0],landmarks[8][1])
        center = fore_finger
        thumb = (landmarks[4][0],landmarks[4][1])
        cv2.circle(frame, center, 3, (0,255,0),-1)
        #print(center[1]-thumb[1])
        if (thumb[1]-center[1]<30):
            bpoints.append(deque(maxlen=512))
            blue_index += 1
            gpoints.append(deque(maxlen=512))
            green_index += 1
            rpoints.append(deque(maxlen=512))
            red_index += 1
            ypoints.append(deque(maxlen=512))
            yellow_index += 1

        elif center[1] <= 65:
            if 40 <= center[0] <= 140: # Clear Button
                flag_house = False

                flag_fish = False
                flag_fish2 = True
                flag_fish_save = False

                flag_simbir = False
                flag_simbir2 = True
                flag_simbir_save = False

                flag_save_img = False

                thickness = 4
                paintWindow = copy.deepcopy(paintWindow_first)

                bpoints = [deque(maxlen=512)]
                gpoints = [deque(maxlen=512)]
                rpoints = [deque(maxlen=512)]
                ypoints = [deque(maxlen=512)]

                blue_index = 0
                green_index = 0
                red_index = 0
                yellow_index = 0

                paintWindow[67:,:,:] = 255
            elif 160 <= center[0] <= 255:
                    colorIndex = 0 # Check
                    flag_save_img = True
            elif 275 <= center[0] <= 370:
                    colorIndex = 1 # SimbirSoft
                    flag_simbir = True
                    thickness = 9
            elif 390 <= center[0] <= 485:
                    colorIndex = 2 # Fish
                    flag_fish = True
            elif 505 <= center[0] <= 600:
                    colorIndex = 3 # House
                    flag_house = True
        else :
            # условия если навелся на кнопку
            if colorIndex == 0:
                bpoints[blue_index].appendleft(center)
            elif colorIndex == 1:
                gpoints[green_index].appendleft(center)
            elif colorIndex == 2:
                rpoints[red_index].appendleft(center)
            elif colorIndex == 3:
                ypoints[yellow_index].appendleft(center)

    # Append the next deques when nothing is detected to avois messing up
    else:
        bpoints.append(deque(maxlen=512))
        blue_index += 1
        gpoints.append(deque(maxlen=512))
        green_index += 1
        rpoints.append(deque(maxlen=512))
        red_index += 1
        ypoints.append(deque(maxlen=512))
        yellow_index += 1

    # если навелся на House
    if flag_house:
        frame = cv2.circle(frame, ((frame.shape[0] // 2) - 10, (frame.shape[1] // 2) - 90), 7, (0,0,255), -1)
        frame = cv2.circle(frame, ((frame.shape[0] // 2) - 10, (frame.shape[1] // 2) + 10), 7, (0,0,255), -1)
        frame = cv2.circle(frame, ((frame.shape[0] // 2) + 160, (frame.shape[1] // 2) - 90), 7, (0,0,255), -1)
        frame = cv2.circle(frame, ((frame.shape[0] // 2) + 160, (frame.shape[1] // 2) + 10), 7, (0,0,255), -1)

        frame = cv2.circle(frame, ((frame.shape[0] // 2) + 80, (frame.shape[1] // 2) - 185), 7, (0,0,255), -1)

        paintWindow = cv2.circle(paintWindow, ((paintWindow.shape[0] // 2) - 10, (paintWindow.shape[1] // 2) - 90), 7, (0, 0, 255), -1)
        paintWindow = cv2.circle(paintWindow, ((paintWindow.shape[0] // 2) - 10, (paintWindow.shape[1] // 2) + 10), 7, (0, 0, 255), -1)
        paintWindow = cv2.circle(paintWindow, ((paintWindow.shape[0] // 2) + 160, (paintWindow.shape[1] // 2) - 90), 7, (0, 0, 255), -1)
        paintWindow = cv2.circle(paintWindow, ((paintWindow.shape[0] // 2) + 160, (paintWindow.shape[1] // 2) + 10), 7, (0, 0, 255), -1)

        paintWindow = cv2.circle(paintWindow, ((paintWindow.shape[0] // 2) + 80, (paintWindow.shape[1] // 2) - 185), 7, (0, 0, 255), -1)

    # если навелся на рыбку
    if flag_fish and flag_fish2:
        img1 = cv2.imread('./save_images/resized_img.png', cv2.IMREAD_COLOR)
        #cv2.imwrite('test_img2.jpg', paintWindow)
        #img1 = np.asarray(img1, np.float64)
        #paintWindow = np.asarray(paintWindow, np.float64)
        print(paintWindow.shape, img1.shape)
        print(type(img1), type(paintWindow))
        alpha = 0.5
        beta = (1.0 - alpha)

        paintWindow = cv2.addWeighted(img1, alpha, paintWindow.astype(np.uint8), beta, 0.0)
        flag_fish = False
        flag_fish2 = False
        flag_fish_save = True

    # если навелся на СимбирСофт
    if flag_simbir and flag_simbir2:
        img1 = cv2.imread('./save_images/simbir_save.jpg', cv2.IMREAD_COLOR)


        print(paintWindow.shape, img1.shape)
        print(type(img1), type(paintWindow))
        alpha = 0.5
        beta = (1.0 - alpha)

        paintWindow = cv2.addWeighted(img1, alpha, paintWindow.astype(np.uint8), beta, 0.0)
        flag_simbir = False
        flag_simbir2 = False
        flag_simbir_save = True


    # Draw lines of all the colors on the canvas and frame
    points = [bpoints, gpoints, rpoints, ypoints]

    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is None or points[i][j][k] is None:
                    continue
                #cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                #cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                cv2.line(frame, points[i][j][k - 1], points[i][j][k], (255,0,0), thickness)
                cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], (255,0,0), thickness)


    # сохраняем рисунок и сравниваем его с шаблоном
    if flag_save_img:
        if flag_house:
            print('save!!')
            tmp_img = paintWindow[(frame.shape[1] // 2) - 200:(frame.shape[1] // 2) + 40, (frame.shape[0] // 2)-30:(frame.shape[0] // 2) + 180]
            cv2.imwrite(f'./save_images/{count}.jpg', tmp_img)
            flag_save_img = False

            percent = compare_images(f'./save_images/{count}.jpg', f'./save_images/template_2.jpg')

            cv2.putText(paintWindow, f"Ты художник на {percent}%", (paintWindow.shape[0] // 2, paintWindow.shape[1] - 200), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_4)
            print("!!!!!!!!!!!!!!!")
            print("Изображения похожи на", compare_images(f'./save_images/{count}.jpg', f'./save_images/template_2.jpg'), "%")
            print("!!!!!!!!!!!!!!!")
        elif flag_fish_save:
            print('save!!')
            tmp_img = paintWindow[80:390, 110:520]
            cv2.imwrite(f'./save_images/{count}.jpg', tmp_img)
            flag_save_img = False

            percent = compare_images(f'./save_images/{count}.jpg', f'./save_images/template_fish.jpg')

            cv2.putText(paintWindow, f"Ты художник на {percent}%",
                        (paintWindow.shape[0] // 2, paintWindow.shape[1] - 200), cv2.FONT_HERSHEY_COMPLEX, 0.8,
                        (255, 0, 0), 2, cv2.LINE_4)
            print("!!!!!!!!!!!!!!!")
            print("Изображения похожи на",
                  compare_images(f'./save_images/{count}.jpg', f'./save_images/template_fish.jpg'), "%")
            print("!!!!!!!!!!!!!!!")
        elif flag_simbir_save:
            print('save!!')
            tmp_img = paintWindow[150:325, 235:400]
            cv2.imwrite(f'./save_images/{count}.jpg', tmp_img)
            flag_save_img = False

            percent = compare_images(f'./save_images/{count}.jpg', f'./save_images/template_simbir.jpg')

            cv2.putText(paintWindow, f"Ты художник на {percent}%",
                        (paintWindow.shape[0] // 2, paintWindow.shape[1] - 200), cv2.FONT_HERSHEY_COMPLEX, 0.8,
                        (255, 0, 0), 2, cv2.LINE_4)
            print("!!!!!!!!!!!!!!!")
            print("Изображения похожи на",
                  compare_images(f'./save_images/{count}.jpg', f'./save_images/template_simbir.jpg'), "%")
            print("!!!!!!!!!!!!!!!")




    # логика для рисунка точки за пальцем
    paintWindow_pred = copy.deepcopy(paintWindow)
    paintWindow = cv2.circle(paintWindow, center, 7, (0, 0, 255), -1)



    cv2.imshow("Output", frame) 
    cv2.imshow("Paint", paintWindow)

    # после показа точки где находится палец обновляем холст без этой точки
    paintWindow = paintWindow_pred



    if cv2.waitKey(1) == ord('q'):
        break

# release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()
