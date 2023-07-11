#!/usr/bin/env python

import os
import re
import time
import shutil
import fnmatch

import numpy as np
import sounddevice as sd
import threading
import subprocess
import requests

from pyannote.audio.pipelines import VoiceActivityDetection
from scipy.io.wavfile import write

from dotenv import load_dotenv
from faker import Faker
from flask import Flask, Response, jsonify, redirect, request
from twilio.jwt.access_token import AccessToken
from twilio.jwt.access_token.grants import VoiceGrant
from twilio.twiml.voice_response import Dial, VoiceResponse

load_dotenv()

app = Flask(__name__)
fake = Faker()
alphanumeric_only = re.compile("[\W_]+")
phone_pattern = re.compile(r"^[\d\+\-\(\) ]+$")

twilio_number = os.environ.get("TWILIO_CALLER_ID")

# Store the most recently created identity in memory for routing calls
IDENTITY = {"identity": ""}

is_call_ongoing = False

# Initialize a global counter for folder names
folder_counter = 0

rrr = None


# voice activity detection 모델 세팅 (app.py에 들어가야 할 듯)

# 1. visit hf.co/pyannote/segmentation and accept user conditions
# 2. visit hf.co/settings/tokens to create an access token
# 3. instantiate pretrained model
from pyannote.audio import Model
model = Model.from_pretrained("pyannote/segmentation",
                              use_auth_token="your_VAD_token")

HYPER_PARAMETERS = {
  # onset/offset activation thresholds
  "onset": 0.684, "offset": 0.577,
  # remove speech regions shorter than that many seconds.
  "min_duration_on": 0.181,
  # fill non-speech regions shorter than that many seconds.
  "min_duration_off": 0.037
}

pipeline = VoiceActivityDetection(segmentation=model)
pipeline.instantiate(HYPER_PARAMETERS)

def process_audio(filename):
    global folder_counter
    global rrr

    # Increase the counter
    folder_counter += 1

    # Make sure the output directory exists
    os.makedirs(f"D:/segments_{str(folder_counter).zfill(4)}", exist_ok=True)

    # Use VAD on the audio file
    vad1 = pipeline(filename)

    # Get the timeline from the vad Annotation
    timeline = vad1.get_timeline()

    # Process each segment
    for i, segment in enumerate(timeline):
        # Calculate start and end times in seconds
        start_s = segment.start
        duration_s = segment.duration

        # Construct the ffmpeg command
        command = [
            'ffmpeg',
            '-i', filename,  # input file
            '-ss', str(start_s),  # start time
            '-t', str(duration_s),  # duration
            '-vn',  # no video
            f"D:/segments_{str(folder_counter).zfill(4)}/segment_{str(folder_counter).zfill(4)}_{i}.wav"  # output file
        ]

        # Run the command
        subprocess.run(command, check=True)

    # Upload all files in the output directory
    url = 'http://dd1e-34-91-77-244.ngrok-free.app/upload'  # ngrok URL을 사용
    for file_name in os.listdir(f"D:/segments_{str(folder_counter).zfill(4)}"):
        file_path = os.path.join(f"D:/segments_{str(folder_counter).zfill(4)}", file_name)

        with open(file_path, 'rb') as f:
            files = {'file': f}
            r = requests.post(url, files=files)

    url = 'http://dd1e-34-91-77-244.ngrok-free.app/evaluate'
    rrr = requests.get(url)
    print(rrr.text)

def record_audio():
    fs = 22050  # Sample rate
    seconds = 10  # Duration of recording
    device_index = 3  # Replace with the index of your device

    # Initialize the counter
    counter = 1

    while is_call_ongoing:
        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1, device=device_index)
        sd.wait()  # Wait until recording is finished

        # Create a filename with the counter
        filename = 'D:/output_{:04d}.wav'.format(counter)
        # Increase the counter
        counter += 1

        # Save as WAV file
        write(filename, fs, myrecording)

        # Process the audio in a new thread
        thread = threading.Thread(target=process_audio, args=(filename,))
        thread.start()


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/token", methods=["GET"])
def token():
    # get credentials for environment variables
    account_sid = os.environ["TWILIO_ACCOUNT_SID"]
    application_sid = os.environ["TWILIO_TWIML_APP_SID"]
    api_key = os.environ["API_KEY"]
    api_secret = os.environ["API_SECRET"]

    # Generate a random user name and store it
    identity = alphanumeric_only.sub("", fake.user_name())
    IDENTITY["identity"] = identity

    # Create access token with credentials
    token = AccessToken(account_sid, api_key, api_secret, identity=identity)

    # Create a Voice grant and add to token
    voice_grant = VoiceGrant(
        outgoing_application_sid=application_sid,
        incoming_allow=True,
    )
    token.add_grant(voice_grant)

    # Return token info as JSON
    token = token.to_jwt()

    # Return token info as JSON
    return jsonify(identity=identity, token=token)


@app.route("/voice", methods=["POST"])
def voice():
    global is_call_ongoing

    resp = VoiceResponse()
    if request.form.get("To") == twilio_number:
        # Receiving an incoming call to our Twilio number
        dial = Dial()
        # Route to the most recently created client based on the identity stored in the session
        dial.client(IDENTITY["identity"])
        resp.append(dial)
    elif request.form.get("To"):
        # Placing an outbound call from the Twilio client
        dial = Dial(caller_id=twilio_number)
        # wrap the phone number or client name in the appropriate TwiML verb
        # by checking if the number given has only digits and format symbols
        if phone_pattern.match(request.form["To"]):
            dial.number(request.form["To"])
        else:
            dial.client(request.form["To"])
        resp.append(dial)
    else:
        resp.say("Thanks for calling!")

    # Start recording in a new thread
    is_call_ongoing = 1
    thread = threading.Thread(target=record_audio)
    thread.start()

    return Response(str(resp), mimetype="text/xml")

@app.route("/call-status", methods=["POST"])
def call_status():
    global is_call_ongoing
    call_status = request.values.get('CallStatus', None)
    if call_status == 'completed':
        is_call_ongoing = False

        time.sleep(20)

        url = 'http://dd1e-34-91-77-244.ngrok-free.app/clean'
        r = requests.get(url)
        print(r.text)  # 'Clean up complete'를 출력합니다.

        dir_pattern = 'segments_*'
        dir_path = 'D:\\'

        for dir_name in os.listdir(dir_path):
            if fnmatch.fnmatch(dir_name, dir_pattern):
                full_dir_path = os.path.join(dir_path, dir_name)
                if os.path.isdir(full_dir_path):
                    shutil.rmtree(full_dir_path)
                    print(f"'{dir_name}' folder is deleted.")

    return ('', 204)

@app.route("/get_rrr", methods=["GET"])
def get_rrr():
    global rrr
    if rrr and rrr.status_code == 200:  # rrr이 None이 아니고, 응답 코드가 200(성공)인 경우
        return jsonify(value=rrr.json())
    else:
        return jsonify(value=None)  # rrr이 None이거나, 응답 코드가 200이 아닌 경우


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
