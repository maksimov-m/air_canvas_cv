from tkinter import *
import customtkinter
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import copy
import time


def compare_images(draw_img: str, template: str) -> float:
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


def add_to_leaderboard(name, email, img, score, filename="leaderboard.txt"):
    """
    Запись информации в лидерборд
    :param name: имя
    :param email: email
    :param img: Тип рисунка (simbirsot, house, fish)
    :param score: Награда
    :param filename: путь к файлу лидерборда
    :return: None
    """
    # Открываем файл для добавления строки в конец
    with open(filename, "a") as file:
        # Записываем строку в файл в формате "имя фамилия баллы"
        file.write(f"{name} {email} {img} {score}\n")

    # Открываем файл для добавления строки в конец
    with open("info.txt", "a") as file:
        # Записываем строку в файл в формате "имя фамилия баллы"
        file.write(f"{name} {email} {img} {score}\n")


class Program:
    def __init__(self):

        # Чтобы узнать ширину и высоту надо узнать frame.shape из функции main_app
        self.width_window = 960
        self.hight_window = 720
        self.header_width = 115
        # Предоставление разных массивов для обработки цветовых точек разного цвета
        self.bpoints = [deque(maxlen=1024)]

        # Эти индексы будут использоваться для обозначения точек в определенных массивах определенного цвета
        self.blue_index = 0

        self.colorIndex = 0

        # TODO: Подогнать размер для нового разрешения окон
        # menu
        self.header = cv2.imread('./save_images/window.png')
        self.header = cv2.resize(self.header, (self.width_window, self.hight_window))

        # 84 - высота шапки-меню
        self.header = self.header[0:self.header_width, 0:self.width_window]

        # Here is code for Canvas setup
        self.paintWindow = np.zeros((self.hight_window, self.width_window, 3), dtype=np.uint8) + 255

        # Дополнительно копируем чисто окно
        self.paintWindow_pred = copy.deepcopy(self.paintWindow)
        self.paintWindow_first = copy.deepcopy(self.paintWindow)

        # initialize mediapipe
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.mpDraw = mp.solutions.drawing_utils

        self.thickness = 4
        # Initialize the webcam
        self.cap = cv2.VideoCapture(0)
        self.ret = True

        # флаг для сохранение рисунка в файл
        self.flag_save_img = False

        # флаг для вывода точек домика
        self.flag_house = False
        self.flag_house2 = True
        self.flag_house_save = False

        # флаги для работы с рыбкой
        self.flag_fish = False
        self.flag_fish2 = True
        self.flag_fish_save = False

        # флаги для работы с SimbirSoft
        self.flag_simbir = False
        self.flag_simbir2 = True
        self.flag_simbir_save = False

        # Засекаем начальное время
        self.start_time_ = False
        self.start_cur_time = False
        self.start_time = None

        # Задаем время работы программы после окончания рисования
        self.program_duration = 5

        self.leader_bord = True

        # время начала игры
        self.start_game_flag = False
        self.start_game = None
        # количество секунд на игру
        self.time_game = None
        self.elapsed_time = None

        self.count = 0

        self.block_click_button = False

    def change_size_image(self, img):
        # percent by which the image is resized
        scale_percent = 150
        # calculate the 50 percent of original dimensions
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)

        dsize = (width, height)
        img = cv2.resize(img, dsize)
        # print(img.shape)
        return img

    def clear(self):
        self.flag_house = False
        self.flag_house2 = True
        self.flag_house_save = False

        self.flag_fish = False
        self.flag_fish2 = True
        self.flag_fish_save = False

        self.flag_simbir = False
        self.flag_simbir2 = True
        self.flag_simbir_save = False

        self.flag_save_img = False

        self.paintWindow = copy.deepcopy(self.paintWindow_first)

        self.bpoints = [deque(maxlen=512)]

        self.blue_index = 0

    def main_app(self, name_user="Anonimus", email_user="None"):
        while self.ret:

            # Запуск начала игры
            if self.start_time_:
                self.start_time = time.time()
                self.start_time_ = False
                self.start_cur_time = True

            # окончание работы программы после окончания рисования (как только прошло self.program_duration секунд)
            if self.start_cur_time:
                current_time = time.time()  # Получаем текущее время

                # Если прошло больше program_duration секунд, выходим из цикла
                if current_time - self.start_time >= self.program_duration:
                    break

            # чтение кадров
            self.ret, frame = self.cap.read()

            frame = cv2.flip(frame, 1)
            frame = self.change_size_image(frame)
            framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # TODO: Подогнать размер
            self.header = self.header[0:self.header_width, 0:self.width_window]
            frame[0:self.header_width, 0:self.width_window] = self.header

            # Get hand landmark prediction
            result = self.hands.process(framergb)
            center = None
            # post process the result
            if result.multi_hand_landmarks:
                landmarks = []
                for handslms in result.multi_hand_landmarks:
                    for lm in handslms.landmark:
                        # # print(id, lm)
                        # print(lm.x)
                        # print(lm.y)
                        # TODO: Подогнать размеры
                        lmx = int(lm.x * self.width_window)
                        lmy = int(lm.y * self.hight_window)

                        landmarks.append([lmx, lmy])

                    # Drawing landmarks on frames
                    self.mpDraw.draw_landmarks(frame, handslms, self.mpHands.HAND_CONNECTIONS)

                fore_finger = (landmarks[8][0], landmarks[8][1])
                # Центр пальца, которым рисуем
                center = fore_finger
                thumb = (landmarks[4][0], landmarks[4][1])
                cv2.circle(frame, center, 3, (0, 255, 0), -1)
                if (thumb[1] - center[1] < 30):
                    self.bpoints.append(deque(maxlen=512))
                    self.blue_index += 1

                elif center[1] <= self.header_width:
                    if 50 <= center[0] <= 190:  # Clear Button
                        # print('Clear')
                        self.block_click_button = False
                        self.clear()
                    elif 230 <= center[0] <= 380:  # Check
                        # Проверяем только тогда, когда начали игру
                        # print('Check')
                        if self.block_click_button:
                            self.flag_save_img = True
                    elif 420 <= center[0] <= 570:  # SimbirSoft
                        # print('SimbirSoft')
                        if not self.block_click_button:
                            self.flag_simbir = True
                            self.start_game_flag = True
                            self.thickness = 9
                            # Флаг для блокировки других кнопок (тк начали уже игру)
                            self.block_click_button = True
                    elif 610 <= center[0] <= 750:  # Fish
                        # print("Fish")
                        if not self.block_click_button:
                            self.flag_fish = True
                            self.start_game_flag = True

                            # Флаг для блокировки других кнопок (тк начали уже игру)
                            self.block_click_button = True
                    elif 790 <= center[0] <= 900:  # House
                        # print('House')
                        if not self.block_click_button:
                            self.flag_house = True
                            self.start_game_flag = True

                            # Флаг для блокировки других кнопок (тк начали уже игру)
                            self.block_click_button = True
                else:
                    self.bpoints[self.blue_index].appendleft(center)
            else:
                self.bpoints.append(deque(maxlen=512))
                self.blue_index += 1

            if self.start_game_flag and self.block_click_button:
                self.start_game_flag = False
                self.start_game = time.time()
                # если навелся на House
                if self.flag_house and self.flag_house2:
                    self.time_game = 30
                    img1 = cv2.imread('./save_images/house_save.png', cv2.IMREAD_COLOR)
                    img1 = cv2.resize(img1, (self.width_window, self.hight_window))

                    alpha = 0.2
                    beta = (1.0 - alpha)

                    self.paintWindow = cv2.addWeighted(img1, alpha, self.paintWindow.astype(np.uint8), beta, 0.0)
                    self.flag_house = False
                    self.flag_house2 = False
                    self.flag_house_save = True

                # если навелся на рыбку
                if self.flag_fish and self.flag_fish2:
                    self.time_game = 30
                    img1 = cv2.imread('./save_images/resized_img.png', cv2.IMREAD_COLOR)
                    img1 = cv2.resize(img1, (self.width_window, self.hight_window))

                    alpha = 0.5
                    beta = (1.0 - alpha)

                    self.paintWindow = cv2.addWeighted(img1, alpha, self.paintWindow.astype(np.uint8), beta, 0.0)
                    self.flag_fish = False
                    self.flag_fish2 = False
                    self.flag_fish_save = True

                # если навелся на СимбирСофт
                if self.flag_simbir and self.flag_simbir2:
                    self.time_game = 30
                    img1 = cv2.imread('./save_images/simbir_save.jpg', cv2.IMREAD_COLOR)
                    img1 = cv2.resize(img1, (self.width_window, self.hight_window))

                    alpha = 0.5
                    beta = (1.0 - alpha)

                    self.paintWindow = cv2.addWeighted(img1, alpha, self.paintWindow.astype(np.uint8), beta, 0.0)
                    self.flag_simbir = False
                    self.flag_simbir2 = False
                    self.flag_simbir_save = True

            points = [self.bpoints]

            for i in range(len(points)):
                for j in range(len(points[i])):
                    for k in range(1, len(points[i][j])):
                        if points[i][j][k - 1] is None or points[i][j][k] is None:
                            continue
                        cv2.line(frame, points[i][j][k - 1], points[i][j][k], (255, 0, 0), self.thickness)
                        cv2.line(self.paintWindow, points[i][j][k - 1], points[i][j][k], (255, 0, 0), self.thickness)

            # сохраняем рисунок и сравниваем его с шаблоном
            if self.flag_save_img or (self.elapsed_time != None and self.elapsed_time >= self.time_game):
                self.start_time_ = True
                self.time_game = None
                self.elapsed_time = None
                self.type_img = None

                if self.flag_house_save:
                    self.type_img = "House"
                    tmp_img = self.paintWindow[(frame.shape[1] // 2) - 300:(frame.shape[1] // 2) + 20,
                              (frame.shape[0] // 2) - 40:(frame.shape[0] // 2) + 270]

                    cv2.imwrite(f'./save_images/{self.count}.jpg', tmp_img)

                    self.flag_save_img = False

                    self.percent = compare_images(f'./save_images/{self.count}.jpg', f'./save_images/template_2.jpg')

                    cv2.putText(self.paintWindow, f"{name_user}, ты художник на {self.percent}%",
                                (200, self.paintWindow.shape[1] - 300), cv2.FONT_HERSHEY_COMPLEX, 0.8,
                                (255, 0, 0), 2, cv2.LINE_4)

                elif self.flag_fish_save:
                    self.type_img = "Fish"

                    tmp_img = self.paintWindow[150:600, 100:800]
                    cv2.imwrite(f'./save_images/{self.count}.jpg', tmp_img)
                    self.flag_save_img = False

                    self.percent = compare_images(f'./save_images/{self.count}.jpg', f'./save_images/template_fish.jpg')

                    cv2.putText(self.paintWindow, f"{name_user}, ты художник на {self.percent}%",
                                (200, self.paintWindow.shape[1] - 300), cv2.FONT_HERSHEY_COMPLEX, 0.8,
                                (255, 0, 0), 2, cv2.LINE_4)

                elif self.flag_simbir_save:
                    self.type_img = "SimbirSoft"

                    tmp_img = self.paintWindow[190:505, 235:700]
                    cv2.imwrite(f'./save_images/{self.count}.jpg', tmp_img)
                    self.flag_save_img = False
                    cv2.imwrite(f'./save_images/{self.count}_full.jpg', self.paintWindow)
                    self.percent = compare_images(f'./save_images/{self.count}.jpg',
                                                  f'./save_images/template_simbir.jpg')

                    cv2.putText(self.paintWindow, f"{name_user}, ты художник на {self.percent}%",
                                (200, self.paintWindow.shape[1] - 300), cv2.FONT_HERSHEY_COMPLEX, 0.8,
                                (255, 0, 0), 2, cv2.LINE_4)

                if self.leader_bord:
                    add_to_leaderboard(name_user, email_user, self.type_img, self.percent)
                    self.leader_bord = False

            # логика для рисунка точки за пальцем
            self.paintWindow_pred = copy.deepcopy(self.paintWindow)
            self.paintWindow = cv2.circle(self.paintWindow, center, 7, (0, 0, 255), -1)

            if self.time_game != None and self.time_game >= 0:
                self.elapsed_time = time.time() - self.start_game
                remaining_time = self.time_game - int(self.elapsed_time)

                cv2.putText(self.paintWindow, f" Осталось {remaining_time} секунд",
                            (200, self.paintWindow.shape[1] - 300), cv2.FONT_HERSHEY_COMPLEX, 0.8,
                            (255, 0, 0), 2, cv2.LINE_4)

            self.paintWindow[0:self.header_width, 0:self.width_window] = self.header

            cv2.imshow("Image", frame)
            cv2.imshow("Paint", self.paintWindow)

            # после показа точки где находится палец обновляем холст без этой точки
            self.paintWindow = self.paintWindow_pred

            if cv2.waitKey(1) == ord('q'):
                break
        # release the webcam and destroy all active windows
        self.cap.release()
        cv2.destroyAllWindows()


class ToplevelWindow(customtkinter.CTkToplevel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.title("Лидерборд")

        self.geometry("500x400")
        # Читаем информацию из файла
        try:
            with open("leaderboard.txt", "r") as file:
                leaderboard_data = file.readlines()
        except FileNotFoundError:
            customtkinter.CTkLabel(self, text="Файл лидерборда не найден").pack()
            return

        # Создаем список для хранения данных лидерборда
        leaderboard = []
        for line in leaderboard_data:
            # Разбиваем строку на части по пробелу
            parts = line.split()
            if len(parts) == 4:
                name, email, category, rating = parts
                leaderboard.append((name, email, category, rating))

        # Сортируем лидерборд по рейтингу
        leaderboard.sort(key=lambda x: x[3], reverse=True)

        res = ""
        # Создаем заголовок таблицы
        header = ["Имя", "Тип рисунка", "Награда"]
        col_widths = [max(len(col), 10) for col in header]  # Находим максимальную длину для каждого столбца
        res += "+" + "-" * (col_widths[0] + 2) + "+" + "-" * (col_widths[1] + 2) + "+" + "-" * (
                    col_widths[2] + 2) + "+\n"
        res += "|" + header[0].ljust(col_widths[0]) + "|" + header[1].ljust(col_widths[1]) + "|" + header[2].ljust(
            col_widths[2]) + "|\n"
        res += "+" + "-" * (col_widths[0] + 2) + "+" + "-" * (col_widths[1] + 2) + "+" + "-" * (
                    col_widths[2] + 2) + "+\n"

        # Добавляем данные
        for name, _, img, score in leaderboard:
            res += "|" + name.ljust(col_widths[0]) + "|" + img.ljust(col_widths[1]) + "|" + score.ljust(
                col_widths[2]) + "|\n"

        # Добавляем нижнюю границу таблицы
        res += "+" + "-" * (col_widths[0] + 2) + "+" + "-" * (col_widths[1] + 2) + "+" + "-" * (
                    col_widths[2] + 2) + "+\n"
        print(res)
        # Отображаем лидерборд
        # for i, (name, email, category, rating) in enumerate(leaderboard, start=1):
        #    customtkinter.CTkLabel(self.leaderboard_window, text=f"{i}. {name} {email} ({category}): {rating}").pack()
        customtkinter.CTkLabel(self, text=res, font=("Courier New", 15)).pack()


class UIApp(customtkinter.CTk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        customtkinter.set_appearance_mode("System")  # Modes: system (default), light, dark
        customtkinter.set_default_color_theme("blue")  # Themes: blue (default), dark-blue, green

        self.geometry("400x300")
        self.grid_columnconfigure(0, weight=1)

        self.name_label = customtkinter.CTkLabel(self, text="Введите имя", fg_color="transparent")
        self.name_label.grid(row=0, column=0, padx=0, pady=10, sticky="ew")

        self.name = customtkinter.CTkEntry(self, placeholder_text="имя")
        self.name.grid(row=1, column=0, padx=20, pady=20, sticky="ew")

        self.email_label = customtkinter.CTkLabel(self, text="Введите email", fg_color="transparent")
        self.email_label.grid(row=2, column=0, padx=10, pady=10, sticky="ew")

        self.email = customtkinter.CTkEntry(self, placeholder_text="email")
        self.email.grid(row=3, column=0, padx=20, pady=20, sticky="ew")

        # Use CTkButton instead of tkinter Button
        self.button = customtkinter.CTkButton(master=self, text="Начать", command=self.run_script)
        self.button.grid(row=4, column=0, padx=20, pady=20, sticky="ew")

        self.leaderboard_window = None

    def show_leaderboard(self):
        self.leaderboard_window = customtkinter.CTkToplevel(self)  # master argument is optional
        self.leaderboard_window.title("Лидерборд")

        self.leaderboard_window.geometry("400x200")
        # Читаем информацию из файла
        try:
            with open("leaderboard.txt", "r") as file:
                leaderboard_data = file.readlines()
        except FileNotFoundError:
            customtkinter.CTkLabel(self.leaderboard_window, text="Файл лидерборда не найден").pack()
            return

        # Создаем список для хранения данных лидерборда
        leaderboard = []
        for line in leaderboard_data:
            # Разбиваем строку на части по пробелу
            parts = line.split()
            if len(parts) == 4:
                name, email, category, rating = parts
                leaderboard.append((name, email, category, rating))

        # Сортируем лидерборд по рейтингу
        leaderboard.sort(key=lambda x: x[3], reverse=True)

        res = ""
        res = res + "+" + "-" * 15 + "+" + "-" * 15 + "+" + "-" * 15 + "+" + '\n'
        res = res + "|", "Имя".ljust(10), "|", "Тип рисунка".ljust(10), "|", "Награда".ljust(10), "|" + '\n'
        res = res + "+" + "-" * 15 + "+" + "-" * 15 + "+" + "-" * 15 + "+" + '\n'
        for name, img, score in leaderboard:
            res = res + "|", name.ljust(10), "|", img.ljust(10), "|", score.ljust(10), "|" + '\n'
        res = res + "+" + "-" * 15 + "+" + "-" * 15 + "+" + "-" * 15 + "+" + '\n'

        # Отображаем лидерборд
        #for i, (name, email, category, rating) in enumerate(leaderboard, start=1):
        #    customtkinter.CTkLabel(self.leaderboard_window, text=f"{i}. {name} {email} ({category}): {rating}").pack()
        customtkinter.CTkLabel(self.leaderboard_window, text=res).pack()

    def open_toplevel(self):
        if self.leaderboard_window is None or not self.leaderboard_window.winfo_exists():
            self.leaderboard_window = ToplevelWindow(self)  # create window if its None or destroyed
        else:
            self.leaderboard_window.destroy()
            self.leaderboard_window = ToplevelWindow(self)  # create window if its None or destroyed

    def run_script(self):
        name = self.name.get()
        email = self.email.get()

        """# Ваш путь к скрипту
        script_path = "main_script.py"
        # Здесь запускается ваш скрипт, передавая имя в качестве аргумента
        process = subprocess.Popen(["python", script_path, name, email])
        # Ожидаем завершения процесса
        process.wait()"""

        app = Program()
        app.main_app(name, email)

        self.open_toplevel()


if __name__ == "__main__":
    # Program().main_app()
    main_app = UIApp()
    main_app.mainloop()
