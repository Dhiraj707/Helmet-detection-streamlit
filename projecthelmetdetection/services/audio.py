import os
from playsound import playsound

def play_alert():
    alert_path = "C:\\Users\\hp\\Downloads\\censor-beep-2-372461.mp3"
    if os.path.exists(alert_path):
        print("no helmet image")
        playsound(alert_path)
    else:
        print("Alert sound not found.")