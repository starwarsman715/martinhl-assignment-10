# app.py
import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import pandas as pd
from flask import Flask, request, render_template, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import open_clip
import glob
from sklearn.decomposition import PCA

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
device = "cuda" if torch.cuda.is_available() else "cpu"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load model and embeddings
print("Loading model and embeddings...")
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B/32', pretrained='openai')
model = model.to(device)
model.eval()
df = pd.read_pickle('image_embeddings.pickle')
tokenizer = open_clip.get_tokenizer('ViT-B-32')

# Prepare PCA with first 10000 images
print("Computing PCA components for first 10000 images...")
embeddings_array = np.stack(df['embedding'].iloc[:10000].apply(lambda x: x.cpu().numpy() if isinstance(x, torch.Tensor) else x))
pca = PCA()
pca.fit(embeddings_array)

# Pre-compute PCA embeddings for different k values
max_components = min(512, len(embeddings_array) - 1)  # Maximum possible components
pca_embeddings = {}
print(f"Computing PCA transformations for k=1 to {max_components}...")
for k in range(1, max_components + 1):
    pca_k = PCA(n_components=k)
    pca_k.fit(embeddings_array)
    transformed = pca_k.transform(embeddings_array)
    pca_embeddings[k] = transformed

print("Model, embeddings, and PCA components loaded successfully!")

def get_pca_embedding(query_embedding, k):
    """Transform a query embedding using PCA with k components"""
    query_np = query_embedding.cpu().numpy()
    if len(query_np.shape) == 3:
        query_np = query_np.squeeze(0)
    return pca.transform(query_np.reshape(1, -1))[:, :k]

def get_image_embedding(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_embedding = F.normalize(model.encode_image(image_tensor))
        return image_embedding
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def get_text_embedding(text):
    try:
        with torch.no_grad():
            text_tokens = tokenizer([text]).to(device)
            text_embedding = F.normalize(model.encode_text(text_tokens))
        return text_embedding
    except Exception as e:
        print(f"Error processing text: {e}")
        return None

def find_similar_images(query_embedding, use_pca=False, k_components=None, top_k=5):
    try:
        similarities = []
        
        if use_pca and k_components:
            # Transform query embedding using PCA
            query_pca = get_pca_embedding(query_embedding, k_components)
            # Use pre-computed PCA embeddings
            db_embeddings = pca_embeddings[k_components]
            
            # Only search through the first 10000 images when using PCA
            for i, row in df.iloc[:10000].iterrows():
                db_embedding = db_embeddings[i]
                similarity = np.dot(query_pca.flatten(), db_embedding) / (
                    np.linalg.norm(query_pca) * np.linalg.norm(db_embedding))
                filename = os.path.join('coco_images_resized', row['file_name'])
                similarities.append((filename, float(similarity)))
        else:
            # Use original CLIP embeddings for all images
            query_embedding = query_embedding.cpu().numpy()
            if len(query_embedding.shape) == 3:
                query_embedding = query_embedding.squeeze(0)
            
            for _, row in df.iterrows():
                db_embedding = row['embedding']
                if isinstance(db_embedding, torch.Tensor):
                    db_embedding = db_embedding.cpu().numpy()
                
                if len(db_embedding.shape) == 1:
                    db_embedding = db_embedding.reshape(1, -1)
                if len(query_embedding.shape) == 1:
                    query_embedding = query_embedding.reshape(1, -1)
                
                similarity = np.dot(query_embedding, db_embedding.T)[0][0]
                filename = os.path.join('coco_images_resized', row['file_name'])
                similarities.append((filename, float(similarity)))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    except Exception as e:
        print(f"Error finding similar images: {e}")
        return []

@app.route('/')
def home():
    return render_template('index.html', max_components=max_components)

@app.route('/static/images/<path:filename>')
def serve_image(filename):
    directory = os.path.dirname(filename)
    file = os.path.basename(filename)
    return send_from_directory(os.path.join('static/images', directory), file)

@app.route('/search', methods=['POST'])
def search():
    try:
        query_type = request.form.get('query_type', 'image')
        text_query = request.form.get('text_query', '')
        weight = float(request.form.get('weight', 0.8))
        use_pca = request.form.get('use_pca') == 'true'
        k_components = int(request.form.get('k_components', 0)) if use_pca else None
        
        # Initialize embeddings
        text_embedding = None
        image_embedding = None
        
        # Process based on query type
        if query_type in ['text', 'hybrid']:
            if text_query:
                text_embedding = get_text_embedding(text_query)
                if text_embedding is None:
                    return jsonify({'error': 'Failed to process text query'})
        
        if query_type in ['image', 'hybrid']:
            if 'image_query' in request.files:
                image_file = request.files['image_query']
                if image_file and allowed_file(image_file.filename):
                    filename = secure_filename(image_file.filename)
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    image_file.save(filepath)
                    image_embedding = get_image_embedding(filepath)
                    os.remove(filepath)  # Clean up
                    if image_embedding is None:
                        return jsonify({'error': 'Failed to process image query'})
        
        # Combine embeddings based on query type
        if query_type == 'hybrid' and text_embedding is not None and image_embedding is not None:
            query_embedding = F.normalize(weight * text_embedding + (1.0 - weight) * image_embedding)
        elif text_embedding is not None:
            query_embedding = text_embedding
        elif image_embedding is not None:
            query_embedding = image_embedding
        else:
            return jsonify({'error': 'No valid query provided'})
        
        # Find similar images
        results = find_similar_images(
            query_embedding, 
            use_pca=use_pca and query_type in ['image', 'hybrid'],
            k_components=k_components,
            top_k=5
        )
        
        if not results:
            return jsonify({'error': 'No results found'})
            
        return jsonify({'results': results})
        
    except Exception as e:
        print(f"Search error: {e}")
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)