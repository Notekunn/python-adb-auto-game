import math

import pytesseract
import cv2

pytesseract.pytesseract.tesseract_cmd: str = 'C:\Program Files (x86)\Tesseract-OCR\\tesseract.exe'


class DigitRecognition:
    def __init__(self):
        pass

    @staticmethod
    def try_parse_int(string, base=None):
        try:
            return int(string, base) if base else int(string)
        except Exception:
            return math.inf

    def image_to_number(self, img, debug=False):

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if debug:
            cv2.imwrite('images/ocr/gray.png', img)
        # img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        if debug:
            cv2.imwrite('images/ocr/threshold.png', img)
        # img = cv2.medianBlur(img, 5)
        # if debug:
        #     cv2.imwrite('images/ocr/noise.png', img)

        config = '--psm 13 --oem 3 -c tessedit_char_whitelist=0123456789'
        text = pytesseract.image_to_string(img, config=config).splitlines()[0]
        return self.try_parse_int(str(text))
