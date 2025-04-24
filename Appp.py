### app.py
from flask import Flask, render_template, Response, request, jsonify
from detect import detect_state
import cv2

app = Flask(__name__)

camera = cv2.VideoCapture(0)


def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            state, processed_frame = detect_state(frame)
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/get_state', methods=['POST'])
def get_state():
    success, frame = camera.read()
    if not success:
        return jsonify({'state': 'error'})
    state, _ = detect_state(frame)
    return jsonify({'state': state})


if __name__ == '__main__':
    app.run(debug=True)