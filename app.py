from flask import Flask, request, render_template, send_from_directory
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import os
from ultralytics import YOLO

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/results'

# Load models
yolo_model = YOLO('yolov8n.pt')
vit_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
vit_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    model_choice = request.form['model']
    file = request.files['file']
    filename = file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    if model_choice == 'yolo':
        results = yolo_model(filepath, verbose=False)
        for r in results:
            im_array = r.plot()
            im = Image.fromarray(im_array[..., ::-1])
            result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'yolo_' + filename)
            im.save(result_path)
        return render_template('predict.html', prediction="Object detection using YOLO", image='yolo_' + filename)
    
    elif model_choice == 'vit':
        image = Image.open(filepath)
        inputs = vit_processor(images=image, return_tensors="pt")
        outputs = vit_model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        predicted_class = vit_model.config.id2label[predicted_class_idx]
        return render_template('predict.html', prediction=f"Predicted class: {predicted_class}", image=filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=False,port=5000)
