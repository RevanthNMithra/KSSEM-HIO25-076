from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import os

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

IMG_SIZE = 224
CLASS_NAMES = {
    0: "No_DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferate_DR"
}
ALPHA = 0.25
GAMMA = 2.0

def focal_loss(y_true, y_pred, alpha=ALPHA, gamma=GAMMA):
    """Focal Loss for addressing class imbalance"""
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    
    ce = -y_true * tf.math.log(y_pred)
    focal_weight = y_true * tf.pow(1. - y_pred, gamma)
    focal_loss_val = alpha * focal_weight * ce
    
    return tf.reduce_mean(tf.reduce_sum(focal_loss_val, axis=-1))

# Load model
MODEL_PATH = r"D:"#Update the model path
model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'focal_loss': focal_loss})

def ben_graham_preprocessing(img, img_size):
    """Ben Graham's preprocessing for fundus images"""
    img_512 = tf.image.resize(img, [512, 512])
    
    # Normalize
    img_mean = tf.reduce_mean(img_512, axis=[0, 1], keepdims=True)
    img_std = tf.math.reduce_std(img_512, axis=[0, 1], keepdims=True)
    img_normalized = (img_512 - img_mean) / (img_std + 1e-5)
    
    # Subtract green channel mean
    green = img_normalized[:, :, 1]
    green_mean = tf.reduce_mean(green)
    img_normalized = img_normalized - tf.expand_dims(tf.expand_dims(green_mean, 0), 0)
    
    # Apply circular mask
    center_x, center_y = 256, 256
    radius = 240
    yy, xx = tf.meshgrid(tf.range(512), tf.range(512), indexing='ij')
    mask = tf.cast(
        tf.sqrt(tf.cast((xx - center_x)**2 + (yy - center_y)**2, tf.float32)) <= radius,
        tf.float32
    )
    mask = tf.stack([mask, mask, mask], axis=-1)
    img_normalized = img_normalized * mask
    
    # Resize to target size
    img_final = tf.image.resize(img_normalized, [img_size, img_size])
    img_final = tf.clip_by_value(img_final, -1.0, 1.0)
    
    return img_final

def preprocess_image(img_bytes):
    """Load and preprocess image from bytes"""
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_tensor = tf.convert_to_tensor(img_array)
        img_tensor = tf.expand_dims(img_tensor, 0)
        img_preprocessed = ben_graham_preprocessing(img_tensor[0], IMG_SIZE)
        img_preprocessed = tf.expand_dims(img_preprocessed, 0)
        return img_preprocessed
    except Exception as e:
        raise ValueError(f"Error preprocessing image: {str(e)}")

@app.route('/')
def index():
    html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DR Classifier - Professional Medical Diagnosis</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }

        .main-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            max-width: 1200px;
            width: 100%;
        }

        .card {
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            backdrop-filter: blur(10px);
        }

        .header {
            text-align: center;
            margin-bottom: 40px;
        }

        .header h1 {
            font-size: 32px;
            color: #333;
            margin-bottom: 10px;
            font-weight: 700;
        }

        .header p {
            color: #666;
            font-size: 14px;
            letter-spacing: 0.5px;
        }

        .upload-section {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }

        .upload-area {
            border: 3px dashed #667eea;
            border-radius: 15px;
            padding: 50px 30px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            background: linear-gradient(135deg, #f5f7ff 0%, #e9ecff 100%);
        }

        .upload-area:hover {
            border-color: #764ba2;
            background: linear-gradient(135deg, #ede9ff 0%, #ddd6ff 100%);
            transform: translateY(-2px);
        }

        .upload-area.dragover {
            border-color: #764ba2;
            background: #e0d5ff;
            transform: scale(1.02);
        }

        .upload-icon {
            font-size: 48px;
            margin-bottom: 15px;
            display: block;
        }

        .upload-text {
            color: #333;
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 5px;
        }

        .upload-subtext {
            color: #999;
            font-size: 12px;
        }

        input[type="file"] {
            display: none;
        }

        .preview-section {
            display: none;
            flex-direction: column;
            gap: 15px;
        }

        .preview-section.active {
            display: flex;
        }

        .preview-container {
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.2);
            height: 300px;
            background: #f5f5f5;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .preview-container img {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }

        .action-buttons {
            display: flex;
            gap: 10px;
        }

        .btn {
            flex: 1;
            padding: 12px 24px;
            border: none;
            border-radius: 10px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            letter-spacing: 0.5px;
        }

        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }

        .btn-primary:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
        }

        .btn-primary:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }

        .btn-secondary {
            background: #f5f5f5;
            color: #333;
            border: 2px solid #ddd;
        }

        .btn-secondary:hover {
            background: #efefef;
            border-color: #999;
        }

        .results-section {
            display: none;
            flex-direction: column;
            gap: 20px;
        }

        .results-section.active {
            display: flex;
        }

        .result-card {
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            border-left: 5px solid;
        }

        .result-card.no-dr {
            background: #e8f5e9;
            border-left-color: #4caf50;
        }

        .result-card.mild {
            background: #fff3e0;
            border-left-color: #ff9800;
        }

        .result-card.moderate {
            background: #fff8e1;
            border-left-color: #ffc107;
        }

        .result-card.severe {
            background: #ffebee;
            border-left-color: #f44336;
        }

        .result-card.proliferate {
            background: #fce4ec;
            border-left-color: #e91e63;
        }

        .result-label {
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
        }

        .result-label.no-dr { color: #2e7d32; }
        .result-label.mild { color: #e65100; }
        .result-label.moderate { color: #f57f17; }
        .result-label.severe { color: #c62828; }
        .result-label.proliferate { color: #880e4f; }

        .result-class {
            font-size: 42px;
            font-weight: 700;
            margin: 15px 0;
        }

        .result-class.no-dr { color: #2e7d32; }
        .result-class.mild { color: #e65100; }
        .result-class.moderate { color: #f57f17; }
        .result-class.severe { color: #c62828; }
        .result-class.proliferate { color: #880e4f; }

        .result-description {
            font-size: 13px;
            color: #666;
            line-height: 1.6;
        }

        .severity-indicator {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 20px;
            padding: 15px;
            background: rgba(0, 0, 0, 0.02);
            border-radius: 10px;
            gap: 10px;
        }

        .severity-bar {
            flex: 1;
            height: 8px;
            background: #ddd;
            border-radius: 10px;
            overflow: hidden;
        }

        .severity-fill {
            height: 100%;
            border-radius: 10px;
            transition: width 0.6s ease;
        }

        .severity-level {
            font-size: 12px;
            font-weight: 600;
            white-space: nowrap;
        }

        .loading {
            display: none;
            text-align: center;
            padding: 40px 20px;
        }

        .loading.active {
            display: block;
        }

        .spinner {
            width: 50px;
            height: 50px;
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .loading-text {
            color: #667eea;
            font-weight: 600;
            margin-top: 10px;
        }

        .info-panel {
            display: none;
            flex-direction: column;
            gap: 20px;
        }

        .info-panel.active {
            display: flex;
        }

        .info-item {
            padding: 20px;
            background: #f9f9f9;
            border-radius: 12px;
            border-left: 4px solid #667eea;
        }

        .info-title {
            font-weight: 700;
            color: #333;
            margin-bottom: 8px;
            font-size: 14px;
        }

        .info-text {
            color: #666;
            font-size: 13px;
            line-height: 1.6;
        }

        .error-message {
            display: none;
            padding: 15px;
            background: #ffebee;
            color: #c62828;
            border-left: 4px solid #f44336;
            border-radius: 8px;
            font-size: 14px;
            margin-top: 15px;
        }

        .error-message.show {
            display: block;
        }

        @media (max-width: 900px) {
            .main-container {
                grid-template-columns: 1fr;
            }

            .header {
                grid-column: 1 / -1;
                margin-bottom: 20px;
            }

            .header h1 {
                font-size: 28px;
            }
        }

        @media (max-width: 600px) {
            .card {
                padding: 25px;
            }

            .header h1 {
                font-size: 24px;
            }

            .preview-container {
                height: 250px;
            }

            .result-class {
                font-size: 32px;
            }

            .action-buttons {
                flex-direction: column;
            }
        }
    </style>
</head>
<body>
    <div class="main-container">
        <div>
            <div class="card">
                <div class="header">
                    <h1>👁️ DR Classifier</h1>
                    <p>Diabetic Retinopathy Detection</p>
                </div>

                <div class="upload-section">
                    <div class="upload-area" id="uploadArea">
                        <span class="upload-icon">📤</span>
                        <div class="upload-text">Click to upload</div>
                        <div class="upload-subtext">or drag and drop your fundus image</div>
                        <input type="file" id="fileInput" accept="image/*">
                    </div>

                    <div class="preview-section" id="previewSection">
                        <div class="preview-container">
                            <img id="preview" alt="Preview">
                        </div>
                        <div class="action-buttons">
                            <button class="btn btn-primary" id="classifyBtn">Classify Image</button>
                            <button class="btn btn-secondary" id="clearBtn">Clear</button>
                        </div>
                    </div>

                    <div class="loading" id="loading">
                        <div class="spinner"></div>
                        <div class="loading-text">Analyzing image...</div>
                    </div>

                    <div class="error-message" id="errorMessage"></div>
                </div>
            </div>
        </div>

        <div>
            <div class="card">
                <div class="results-section" id="resultsSection">
                    <div class="result-card" id="resultCard">
                        <div class="result-label" id="resultLabel"></div>
                        <div class="result-class" id="resultClass"></div>
                        <div class="result-description" id="resultDescription"></div>
                        <div class="severity-indicator" id="severityIndicator">
                            <div class="severity-bar">
                                <div class="severity-fill" id="severityFill"></div>
                            </div>
                            <div class="severity-level" id="severityLevel"></div>
                        </div>
                    </div>
                </div>

                <div class="info-panel" id="infoPanel">
                    <div class="info-item">
                        <div class="info-title">📋 DR Classification</div>
                        <div class="info-text">This tool uses deep learning to classify diabetic retinopathy severity from fundus images into 5 categories.</div>
                    </div>
                    <div class="info-item">
                        <div class="info-title">🎯 Classes</div>
                        <div class="info-text"><strong>0:</strong> No DR | <strong>1:</strong> Mild | <strong>2:</strong> Moderate | <strong>3:</strong> Severe | <strong>4:</strong> Proliferative</div>
                    </div>
                    <div class="info-item">
                        <div class="info-title">⚠️ Disclaimer</div>
                        <div class="info-text">This tool is for educational purposes. Always consult with a healthcare professional for medical diagnosis and treatment.</div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const previewSection = document.getElementById('previewSection');
        const preview = document.getElementById('preview');
        const classifyBtn = document.getElementById('classifyBtn');
        const clearBtn = document.getElementById('clearBtn');
        const loading = document.getElementById('loading');
        const resultsSection = document.getElementById('resultsSection');
        const infoPanel = document.getElementById('infoPanel');
        const errorMessage = document.getElementById('errorMessage');
        const resultCard = document.getElementById('resultCard');

        const classDescriptions = {
            0: { name: 'No DR', desc: 'No signs of diabetic retinopathy detected', severity: 0, class: 'no-dr' },
            1: { name: 'Mild DR', desc: 'Mild microaneurysms present', severity: 25, class: 'mild' },
            2: { name: 'Moderate DR', desc: 'Moderate non-proliferative changes', severity: 50, class: 'moderate' },
            3: { name: 'Severe DR', desc: 'Severe non-proliferative retinopathy', severity: 75, class: 'severe' },
            4: { name: 'Proliferative DR', desc: 'Proliferative diabetic retinopathy detected', severity: 100, class: 'proliferate' }
        };

        uploadArea.addEventListener('click', () => fileInput.click());

        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                fileInput.files = files;
                handleFileSelect(files[0]);
            }
        });

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFileSelect(e.target.files[0]);
            }
        });

        function handleFileSelect(file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                preview.src = e.target.result;
                previewSection.classList.add('active');
                uploadArea.style.display = 'none';
                infoPanel.classList.remove('active');
                resultsSection.classList.remove('active');
                errorMessage.classList.remove('show');
            };
            reader.readAsDataURL(file);
        }

        classifyBtn.addEventListener('click', async () => {
            const file = fileInput.files[0];
            if (!file) return;

            loading.classList.add('active');
            classifyBtn.disabled = true;
            errorMessage.classList.remove('show');

            const formData = new FormData();
            formData.append('file', file);

            try {
                const response = await fetch('/classify', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (response.ok) {
                    displayResults(data.class, data.class_name);
                } else {
                    showError(data.error || 'Classification failed');
                }
            } catch (err) {
                showError('Error: ' + err.message);
            } finally {
                loading.classList.remove('active');
                classifyBtn.disabled = false;
            }
        });

        function displayResults(classNum, className) {
            const info = classDescriptions[classNum];
            const resultLabel = document.getElementById('resultLabel');
            const resultClass = document.getElementById('resultClass');
            const resultDescription = document.getElementById('resultDescription');
            const severityFill = document.getElementById('severityFill');
            const severityLevel = document.getElementById('severityLevel');

            resultCard.className = `result-card ${info.class}`;
            resultLabel.className = `result-label ${info.class}`;
            resultClass.className = `result-class ${info.class}`;

            resultLabel.textContent = info.name;
            resultClass.textContent = `Class ${classNum}`;
            resultDescription.textContent = info.desc;

            severityFill.style.background = getSeverityColor(info.severity);
            severityFill.style.width = info.severity + '%';
            severityLevel.textContent = `Severity: ${info.severity}%`;

            resultsSection.classList.add('active');
            infoPanel.classList.remove('active');
        }

        function getSeverityColor(severity) {
            if (severity === 0) return '#4caf50';
            if (severity <= 25) return '#ff9800';
            if (severity <= 50) return '#ffc107';
            if (severity <= 75) return '#f44336';
            return '#e91e63';
        }

        function showError(message) {
            errorMessage.textContent = message;
            errorMessage.classList.add('show');
        }

        clearBtn.addEventListener('click', () => {
            fileInput.value = '';
            preview.src = '';
            previewSection.classList.remove('active');
            uploadArea.style.display = 'block';
            resultsSection.classList.remove('active');
            infoPanel.classList.add('active');
            errorMessage.classList.remove('show');
        });

        // Show info panel on load
        infoPanel.classList.add('active');
    </script>
</body>
</html>'''
    return html_content

@app.route('/classify', methods=['POST'])
def classify():
    """Classify DR from uploaded image"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read and preprocess image
        img_bytes = file.read()
        img_preprocessed = preprocess_image(img_bytes)
        
        # Make prediction
        prediction = model.predict(img_preprocessed, verbose=0)
        predicted_class = int(np.argmax(prediction[0]))
        confidence = float(np.max(prediction[0]))
        
        # Get class name
        class_name = CLASS_NAMES.get(predicted_class, "Unknown")
        
        return jsonify({
            'class': predicted_class,
            'class_name': class_name,
            'confidence': round(confidence * 100, 2)
        }), 200
    
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Classification error: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'Model running successfully'}), 200

if __name__ == '__main__':
    print("=" * 60)
    print("🏥 Diabetic Retinopathy Classifier")
    print("=" * 60)
    print("✅ Model loaded successfully")
    print("📊 Classes: 0=No_DR, 1=Mild, 2=Moderate, 3=Severe, 4=Proliferate_DR")
    print("🌐 Starting server at http://localhost:5000")
    print("=" * 60)

    app.run(debug=True, host='0.0.0.0', port=5000)
