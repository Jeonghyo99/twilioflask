import requests


url = 'http://5efa-34-136-130-86.ngrok-free.app/evaluate'
r = requests.get(url)
print(r.text)  # 'Evaluation complete'를 출력합니다.


'''
url = 'http://c5fb-34-136-130-86.ngrok-free.app/upload'  # ngrok URL을 사용
file_path = 'D:/segments_0001/segment_0.wav'  # 업로드하려는 파일의 경로

with open(file_path, 'rb') as f:
    files = {'file': f}
    r = requests.post(url, files=files)
'''