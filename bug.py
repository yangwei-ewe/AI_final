from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from threading import Timer
import os
import datetime
import base64
from PIL import Image
from io import BytesIO
chrome_options = Options()
chrome_options.add_experimental_option("debuggerAddress", "127.0.0.1:9222")
chrome_driver = r"C:\Users\yyany\OneDrive桌面\chromedriver.exe"
driver = webdriver.Chrome(service = Service(chrome_driver), options=chrome_options)


class RepeatingTimer(Timer):
    def run(self):
        self.finished.wait(self.interval)
        while not self.finished.is_set():
            self.function(*self.args, **self.kwargs)
            self.finished.wait(self.interval)
def click():
    driver.refresh()
    try:
        driver.find_element(By.XPATH,'//*[@id="start"]').click()
    except:pass
    try:
        element = WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.ID, 'code')))
        element.click()
    except:pass
    for j in range(50):
        try:
            driver.find_element(By.ID,'code').send_keys(Keys.ENTER)
        except:pass

def spider():
    driver.maximize_window()
    current_time = datetime.datetime.now()
    timestamp = current_time.strftime("%Y%m%d%H%M%S")
    local_path = 'imgs'
    os.makedirs(local_path, exist_ok=True)
    txt_path = 'lable'
    os.makedirs(txt_path, exist_ok=True)
    for i in range(50):
        xpath = '//*[@id="finishModal"]/div/div/div[4]/div[{}]/img'.format(i)
        urlxpath = '//*[@id="finishModal"]/div/div/div[4]/div[{}]/div'.format(i)
        for element in driver.find_elements(By.XPATH,xpath):
            try:
                img_url = element.get_attribute('src')
                image_data = base64.b64decode(img_url.split(',')[1])
                image = Image.open(BytesIO(image_data))
                name = f'image_{i}_{timestamp}'
                image_path = os.path.join(local_path, '{}.png'.format(name) )
                image.save(image_path)
            except:
                print("error")


def run_click_and_spider():
    click()
    spider()


click()
spider()
t = RepeatingTimer(5, run_click_and_spider)
t.start()
