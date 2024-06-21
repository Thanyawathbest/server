from flask import Flask, jsonify, request, send_from_directory
from flask_pymongo import PyMongo
from bson.objectid import ObjectId
import bson.binary
import numpy as np
from PIL import Image
from img2vec_pytorch import Img2Vec
from sklearn.metrics.pairwise import cosine_similarity
import io
import logging
import base64
from flask_cors import CORS
import os
from dotenv import load_dotenv

load_dotenv()  # โหลด Environment Variables จากไฟล์ .env

app = Flask(__name__)
CORS(app)
# ตั้งค่าเชื่อมต่อกับ MongoDB
app.config['MONGO_URI'] = os.getenv('MONGO_URI')
mongo = PyMongo(app)

# เลือกหรือสร้างคอลเลคชัน
collection = mongo.db.mycollection
collectionform = mongo.db.evaluate_satisfaction
image_features_collection = mongo.db.image_features

# โหลดโมเดล densenet จาก img2vec
img2vec = Img2Vec(model='efficientnet_b5')

# ตั้งค่าการบันทึก log
logging.basicConfig(level=logging.INFO)

def extract_features(img):
    try:
        # Convert image to RGB if it's RGBA or grayscale
        if img.mode != 'RGB':
            img = img.convert('RGB')
        features = img2vec.get_vec(img, tensor=False)
        return features
    except Exception as e:
        logging.error(f"เกิดข้อผิดพลาดในการดึงฟีเจอร์: {e}")
        return None

@app.route('/upload_image', methods=['POST'])
def upload_image():
    try:
        if 'image' not in request.files:
            return jsonify(message="ไม่ได้ส่งไฟล์รูปภาพมา"), 400
        if 'description' not in request.form:
            return jsonify(message="ไม่ได้ส่งคำอธิบายมา"), 400
        if 'topic' not in request.form:
            return jsonify(message="ไม่ได้ส่งหัวข้อมา"), 400

        image_file = request.files['image']
        description = request.form['description']
        topic = request.form['topic']

        img = Image.open(image_file)
        features = extract_features(img)
        if features is None:
            return jsonify(message="เกิดข้อผิดพลาดในการดึงฟีเจอร์"), 500

        # บันทึกรูปภาพใน MongoDB พร้อมฟีเจอร์เวกเตอร์และคำอธิบาย
        image_binary = io.BytesIO()
        img.save(image_binary, format=img.format)
        image_binary = image_binary.getvalue()

        document = {
            'features': features.tolist(),  # เก็บฟีเจอร์เวกเตอร์เป็นอาร์เรย์
            'filename': image_file.filename,
            'description': description,
            'topic': topic,
            'image': bson.binary.Binary(image_binary)
        }

        logging.info(f"Document to be inserted: {document}")
        image_features_collection.insert_one(document)
        return jsonify(message="เพิ่มรูปภาพ คำอธิบาย และฟีเจอร์เรียบร้อยแล้ว"), 201
    except Exception as e:
        logging.error(f"เกิดข้อผิดพลาดในการอัปโหลดรูปภาพ: {e}")
        return jsonify(message="เกิดข้อผิดพลาดในการอัปโหลดรูปภาพ"), 500


@app.route('/search_image', methods=['POST'])
def search_image():
    try:
        if 'image' not in request.files:
            return jsonify(message="ไม่ได้ส่งไฟล์รูปภาพมา"), 400

        image_file = request.files['image']
        img = Image.open(image_file)
        query_features = extract_features(img)
        if query_features is None:
            return jsonify(message="เกิดข้อผิดพลาดในการดึงฟีเจอร์"), 500

        # สร้าง query สำหรับ MongoDB Atlas Vector Search
        search_query = {
            "$vectorSearch": {
                "index": "vector_index",  # ชื่อของ index ที่ใช้ค้นหา
                "path": "features",  # path ที่เก็บฟีเจอร์ในเอกสาร
                "queryVector": query_features.tolist(),  # เวกเตอร์คำค้นหา
                "numCandidates": 10,  # จำนวน candidates ที่จะพิจารณาในการค้นหา
                "limit": 1  # จำนวนผลลัพธ์ที่ต้องการ
            }
        }

        # เรียกใช้ aggregate กับ search query
        pipeline = [
            search_query,
            {"$limit": 5}  # จำกัดจำนวนผลลัพธ์ที่ต้องการ
        ]

        results = list(image_features_collection.aggregate(pipeline))
        if not results:
            return jsonify(message="ไม่พบรูปภาพที่คล้ายกัน"), 404

        result_images = []
        for result in results:
            if 'image' in result:
                image_data = result['image']
                # แปลง image_data เป็น base64 เพื่อส่งกลับไปที่ frontend
                image_base64 = base64.b64encode(image_data).decode('utf-8')
                result_images.append({
                    'filename': result['filename'],
                    'topic': result.get('topic', 'N/A'),  # เพิ่ม topic ด้วย
                    'description': result['description'],
                    'image': image_base64
                })
            else:
                logging.error(f"No 'image' field in result with ID {result['_id']}")

        return jsonify(results=result_images)
    except Exception as e:
        logging.error(f"เกิดข้อผิดพลาดในการค้นหารูปภาพ: {e}")
        return jsonify(message=f"เกิดข้อผิดพลาดในการค้นหารูปภาพ: {e}"), 500


@app.route('/form', methods=['POST'])
def postform():
    if 'rate' not in request.form:
        return jsonify(message="Rate not provided"), 400

    rate = request.form['rate']
    comment = request.form.get('comment', '')
    
    document = {
        'rate': rate,
        'comment': comment
    }

    try:
        collectionform.insert_one(document)
    except Exception as e:
        return jsonify(message=f"Error occurred while storing form data: {e}"), 500

    return jsonify(message="Form data received and stored"), 200

@app.route('/')
def index():
    return send_from_directory('', 'index.html')

if __name__ == '__main__':
    app.run(debug=False)