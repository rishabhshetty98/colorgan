from flask import Flask, render_template, request, send_from_directory
import os
from deployment import predict

app = Flask(__name__)


APP_ROOT = os.path.dirname(os.path.abspath(__file__))


@app.route('/')
def home():
    return render_template('upload.html')


@app.route('/upload', methods=['POST'])
def upload():
    target = os.path.join(APP_ROOT, 'images/')
    print(target)
    if not os.path.isdir(target):
        os.mkdir(target)
    file = request.files['file']
    filename = file.filename
    destination = "/".join([target, filename])
    file.save(destination)
    outname = predict(filename)
    return render_template('colorize.html', og=filename, res_name=outname)


@app.route('/upload/<filename>')
def og_img(filename):
    return send_from_directory('images', filename)


@app.route('/res/<filename>')
def colorize(filename):
    return send_from_directory('colored', filename)


if __name__ == '__main__':
    app.run(debug=True)
