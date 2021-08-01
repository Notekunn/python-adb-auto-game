import time

from ppadb.client import Client as AdbClient
import numpy as np
import cv2

from ocr import DigitRecognition

ocr = DigitRecognition()


class TaskConfig:
    def __init__(self):
        self.max_strength = 0.5 * 1000000  # 3M LC
        self.POINT_EXIST_HISTORY = (853.5, 82.8)
        self.POINT_FIND_FIGHT = (779.5, 356.5)
        self.POINT_REFRESH_FIGHT = (703.8, 148.8)
        self.POINT_SELECT_FIGHT = ((700, 250), (700, 335), (700, 415))
        self.POINT_START_FIGHT = (478.4, 390.3)
        self.STRENGTH_POINT_AREA = ((305, 110), ((240, 323, 408), 40))
        self.MY_TEAM_STRENGTH_AREA = (184, 68, 101, 30)
        self.SKIP_BUTTON = (914.7, 28.10)
        self.REWARD_BUTTON = (666.8, 287.3)
        self.CLOSE_RESULT = (473.6, 496.6)


def same_image(source, target):
    difference = cv2.subtract(source, target)
    is_same = not np.any(difference)
    return is_same

class TaskResult:
    def __init__(self):
        self.win = 0
        self.loss = 0
        self.start_time = time.time()

    def get_point(self):
        return self.win * 2 + self.loss

    def add_win(self):
        self.win += 1

    def add_loss(self):
        self.loss += 1

    def get_timer(self):
        time_diff = time.time() - self.start_time
        return f'Time: {time_diff}s| PerH: {self.get_point() * 60 * 60 / time_diff}'

    def get_progress(self) -> str:
        return f'Win: {self.win} | Loss: {self.loss} | Point: {self.get_point()}'

class Task:
    def __init__(self, device_name, config: TaskConfig, debug=True):
        self.is_running = True
        self.debug = debug
        self.device_name = device_name
        self.client = None
        self.device = None
        self.start_adb()
        self.config = config
        self.stats = TaskResult()
        self.log(f'Start bot with max strength: {self.config.max_strength}')

    def log(self, message, mod=0):
        if mod == 0:
            print(message)
        elif mod == 1 and self.debug:
            print(f'[DEBUG]: {message}')

    def start(self):
        while self.is_running:
            self.check_history()
            time.sleep(0.2)
            if self.is_running:
                i = self.find_fight()
                if self.is_running:
                    self.start_fight(i)
                    self.log(self.stats.get_progress())
                    self.log(self.stats.get_timer())
                    if self.stats.get_point() > 4 * 500:
                        self.stop()
                    time.sleep(2)
            # if self.is_running:
            #     self.fight()
            pass

    def stop(self):
        self.is_running = False

    def start_adb(self):
        self.client = AdbClient('127.0.0.1', port=5037)
        self.log(f'Start adb server version {self.client.version()}')
        device = self.client.device(self.device_name)
        if device is not None:
            self.device = device
            self.log(f'Connect to device {self.device_name}')
        else:
            self.log(f'Cannot found device {self.device_name}')
            self.stop()

    @staticmethod
    def crop_image(source, x, y, w, h):
        return source[y:y + h, x:x + w]

    @staticmethod
    def match(screenshot, template, threshold=0.9):
        img_rgb = np.copy(screenshot)
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        w, h = template.shape[::-1]
        res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)
        # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        # print(f'{min_val} - {max_val} - {min_loc} - {max_loc}')
        # print(cv2.threshold(res, 0.9, max_val, cv2.THRESH_BINARY))
        for pt in zip(*loc[::-1]):
            cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 2)
        is_match = len(loc[0]) > 0
        return is_match

    def get_screenshot(self):
        image = self.device.screencap()
        img = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
        # image = cv2.resize(img, *DEVICE_SIZE)
        cv2.imwrite('images/screenshot.png', img)
        return img

    def check_history(self):
        screenshot = self.get_screenshot()
        if self.debug:
            cv2.imwrite('images/screenshot/history.png', screenshot)
        history = cv2.imread('images/history.png', 0)
        is_match = self.match(screenshot, history)
        if is_match:
            print(f'Find history popup. Click {self.config.POINT_EXIST_HISTORY}')
            self.device.input_tap(*self.config.POINT_EXIST_HISTORY)
        else:
            # print()
            self.log('Not found history popup. Skip...', 1)
            pass

    def find_fight(self, refresh=False):
        if not refresh:
            self.log('Find other player to fight...')
            self.device.input_tap(*self.config.POINT_FIND_FIGHT)
            time.sleep(0.2)
        points = self.get_strange_point()
        if points is None:
            return
        self.log(f'Find 3 player with strength: {points[0]} - {points[1]} - {points[2]}')
        min_p = min(points)
        if min_p > self.config.max_strength:
            self.log('No one can fight!. Refresh player list.', 1)
            self.device.input_tap(*self.config.POINT_REFRESH_FIGHT)
            time.sleep(0.1)
            return self.find_fight(True)
        else:
            i = points.index(min_p)
            self.log(f'Select player #{i} with strength point: {min_p}')
            return i
        pass

    def get_strange_point(self):
        screenshot = self.get_screenshot()
        other_player = cv2.imread('images/other-player.png', 0)
        if self.debug:
            cv2.imwrite('images/screenshot/other-player.png', screenshot)
        if not self.match(screenshot, other_player):
            self.log('Something wrong...')
            self.stop()
            return
        points = list()
        x, w = self.config.STRENGTH_POINT_AREA[0]
        ay, h = self.config.STRENGTH_POINT_AREA[1]
        area_to_crop = map(lambda y: (x, y, w, h), ay)
        cropped_images = list(map(lambda area: self.crop_image(screenshot, *area), area_to_crop))
        for i in range(3):
            img = cropped_images[i]
            # if self.debug:
            #     cv2.imwrite(f'images/img-{i}.png', img)
            point = ocr.image_to_number(img, i == 1)
            self.log(f'SP: {point}', 1)
            points.append(point)
        return points

    def refresh_point(self):
        pass

    def start_fight(self, i: int):
        if i < 0 or i > 2:
            self.log(f'Player with index {i} does not exist!')
            return
        else:
            p_click = self.config.POINT_SELECT_FIGHT[i]
            self.log(f'Click point {p_click}', 1)
            self.device.input_tap(*p_click)
            time.sleep(0.3)
            same_sp = self.check_my_team()
            if same_sp:
                self.device.input_tap(*self.config.POINT_START_FIGHT)
                self.skip_fight()
                self.get_reward()
                return self.get_result()
                pass
            else:
                self.log(f'Change SP! Stop attack...')
                self.stop()
        pass

    def check_my_team(self):
        screenshot = self.get_screenshot()
        if self.debug:
            cv2.imwrite('images/screenshot/my-team.png', screenshot)
        my_team = cv2.imread('images/my-team.png', 0)
        if not self.match(screenshot, my_team):
            self.log('Something wrong when check team...')
            # self.stop()
            return self.check_my_team()
        current_strength = self.crop_image(screenshot, *self.config.MY_TEAM_STRENGTH_AREA)
        old_strength = cv2.imread('images/team-strength.png')
        if old_strength is None:
            self.log('Save last team strength...')
            cv2.imwrite('images/team-strength.png', current_strength)
            return True
        else:
            return True

    def skip_fight(self):
        skip_button = cv2.imread('images/skip-button.png', 0)
        self.log('Wait to skip...')
        while True:
            screenshot = self.get_screenshot()
            if self.match(screenshot, skip_button):
                self.log('Founded skip button!')
                self.device.input_tap(*self.config.SKIP_BUTTON)
                return
            time.sleep(0.1)

    def get_reward(self) -> bool:
        reward = cv2.imread('images/reward.png', 0)
        loss = cv2.imread('images/loss.png', 0)
        self.log('Wait to get reward...')
        # time.sleep(1)
        # self.log('Founded reward! Receiving...')
        # self.device.input_tap(*self.config.REWARD_BUTTON)
        # time.sleep(0.2)
        # self.device.input_tap(*self.config.SKIP_BUTTON)
        while True:
            screenshot = self.get_screenshot()
            if self.match(screenshot, reward, 0.5):
                self.log('Founded reward! Receiving...')
                self.device.input_tap(*self.config.REWARD_BUTTON)
                time.sleep(0.2)
                self.device.input_tap(*self.config.SKIP_BUTTON)
                return True
            if self.match(screenshot, loss):
                self.log('Loss match! Cannot get reward!')
                return False
            # time.sleep(0.1)

    def get_result(self):
        loss = cv2.imread('images/loss.png', 0)
        win = cv2.imread('images/win.png', 0)
        while True:
            screenshot = self.get_screenshot()
            if self.match(screenshot, win):
                self.device.input_tap(*self.config.CLOSE_RESULT)
                self.log('Win match', 1)
                self.stats.add_win()
                return True
            if self.match(screenshot, loss):
                self.device.input_tap(*self.config.CLOSE_RESULT)
                self.log('Loss match', 1)
                self.stats.add_loss()
                return False
            self.get_reward()
            time.sleep(0.1)