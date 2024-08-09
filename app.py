from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import io
import base64
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
import tkinter as tk
from tkinter import filedialog
from sklearn.cluster import KMeans
from scipy import ndimage

app = Flask(__name__)

upload_folder = os.path.join('static', 'uploads')

# Membuat direktori 'static/uploads' jika belum ada
if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)

app.config['UPLOAD'] = upload_folder

@app.route('/')
def landing_page():
    return render_template('index.html')

@app.route('/histogram', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['img']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD'], filename))
        img_path = os.path.join(app.config['UPLOAD'], filename)

        # Membaca gambar dengan OpenCV
        img = cv2.imread(img_path)

        # Menghitung histogram untuk masing-masing saluran (R, G, B)
        hist_r = cv2.calcHist([img], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([img], [1], None, [256], [0, 256])
        hist_b = cv2.calcHist([img], [2], None, [256], [0, 256])

        # Normalisasi histogram
        hist_r /= hist_r.sum()
        hist_g /= hist_g.sum()
        hist_b /= hist_b.sum()

        # Simpan histogram sebagai gambar PNG
        hist_image_path = os.path.join(app.config['UPLOAD'], 'histogram.png')
        plt.figure()
        plt.title("RGB Histogram")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.plot(hist_r, color='red', label='Red')
        plt.plot(hist_g, color='green', label='Green')
        plt.plot(hist_b, color='blue', label='Blue')
        plt.legend()
        plt.savefig(hist_image_path)

        # Hasil equalisasi
        img_equalized = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)  # Ubah ke ruang warna YCrCb
        img_equalized[:, :, 0] = cv2.equalizeHist(img_equalized[:, :, 0])  # Equalisasi komponen Y (luminance)
        img_equalized = cv2.cvtColor(img_equalized, cv2.COLOR_YCrCb2BGR)  # Kembalikan ke ruang warna BGR

        # Menyimpan gambar hasil equalisasi ke folder "static/uploads"
        equalized_image_path = os.path.join('static', 'uploads', 'img-equalized.jpg')
        cv2.imwrite(equalized_image_path, img_equalized)

        # Menghitung histogram untuk gambar yang sudah diequalisasi
        hist_equalized_r = cv2.calcHist([img_equalized], [0], None, [256], [0, 256])
        hist_equalized_g = cv2.calcHist([img_equalized], [1], None, [256], [0, 256])
        hist_equalized_b = cv2.calcHist([img_equalized], [2], None, [256], [0, 256])

        # Normalisasi histogram
        hist_equalized_r /= hist_equalized_r.sum()
        hist_equalized_g /= hist_equalized_g.sum()
        hist_equalized_b /= hist_equalized_b.sum()

        # Simpan histogram hasil equalisasi sebagai gambar PNG        
        hist_equalized_image_path = os.path.join(app.config['UPLOAD'], 'histogram_equalized.png')
        plt.figure()
        plt.title("RGB Histogram (Equalized)")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.plot(hist_equalized_r, color='red', label='Red')
        plt.plot(hist_equalized_g, color='green', label='Green')
        plt.plot(hist_equalized_b, color='blue', label='Blue')
        plt.legend()
        plt.savefig(hist_equalized_image_path)

        return render_template('histogram_equalization.html', img=img_path, img2=equalized_image_path, histogram=hist_image_path, histogram2=hist_equalized_image_path)
    
    return render_template('histogram_equalization.html')

# Load a pre-trained face detection model from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
def apply_gaussian_blur_to_face(image, blur_level):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_roi = image[y:y+h, x:x+w]
        if blur_level > 0:
            face_roi = cv2.GaussianBlur(face_roi, (0, 0), blur_level)
        image[y:y+h, x:x+w] = face_roi

    return image

@app.route("/blurring", methods=["GET", "POST"])
def blur():
    original_image_path = None
    blurred_image_path = None

    if request.method == "POST":
        if "file" not in request.files:
            return render_template("blurring.html")

        file = request.files["file"]
        
        # check if file empty
        if file.filename == "":
            return render_template("blurring.html")

        if file:
            # Save the uploaded image to the uploads folder
            file_path = os.path.join(app.config['UPLOAD'], file.filename)
            file.save(file_path)

            image = cv2.imread(file_path)
            blur_level = int(request.form["blur_level"])

            # Save the original image path for display
            original_image_path = file_path

            # Apply Gaussian blur to the detected face
            image = apply_gaussian_blur_to_face(image, blur_level)

             # Menyimpan gambar dengan wajah-wajah yang telah di-blur
            blurred_image_path = os.path.join(app.config['UPLOAD'], 'blurred_image.jpg')
            cv2.imwrite(blurred_image_path, image)


        return render_template("blurring.html", original_image=original_image_path, blurred_image=blurred_image_path)
    return render_template("blurring.html")

def apply_edge_detection(image_path):
    # Baca gambar dari path
    image = cv2.imread(image_path)
    
    # Konversi gambar ke grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Lakukan deteksi tepi menggunakan Canny edge detector
    edges = cv2.Canny(gray, 100, 200)  # nilai threshold dapat diubah sesuai kebutuhan
    
    # Menyimpan gambar hasil deteksi tepi ke folder "static/uploads"
    edge_image_path = os.path.join(app.config['UPLOAD'], 'edge_detected.jpg')
    cv2.imwrite(edge_image_path, edges)
    
    return edge_image_path

@app.route("/edge", methods=["GET", "POST"])
def edge_detection():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("edge_detection.html")

        file = request.files["file"]

        if file.filename == "":
            return render_template("edge_detection.html")

        if file:
            # Simpan gambar yang diunggah ke folder uploads
            file_path = os.path.join(app.config['UPLOAD'], file.filename)
            file.save(file_path)

            # Save the original image path for display
            original_image_path = file_path
            
            # Proses deteksi tepi pada gambar
            edge_image_path = apply_edge_detection(file_path)

            return render_template("edge_detection.html", original_image=original_image_path, edge_image=edge_image_path)

    return render_template("edge_detection.html")

def grabcut_segmentation(img_path, k=2):
    # Membaca gambar dengan OpenCV
    img = cv2.imread(img_path)

    # Ubah gambar ke ruang warna LAB
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Bentuk matriks piksel sebagai vektor piksel
    pixels = lab_img.reshape((-1, 3))

    # Terapkan k-means clustering untuk segmentasi warna
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)

    # Dapatkan label dari setiap piksel
    labels = kmeans.labels_

    # Dapatkan pusat cluster
    centers = kmeans.cluster_centers_

    # Inisialisasi masker
    mask = np.zeros_like(labels)

    # Temukan label yang mewakili latar belakang (cluster dengan intensitas rendah)
    background_label = np.argmin(np.linalg.norm(centers - [0, 128, 128], axis=1))

    # Isi masker dengan 1 untuk label yang mewakili objek
    mask[labels != background_label] = 1

    # Bentuk kembali masker ke bentuk gambar
    mask = mask.reshape(img.shape[:2])

    # Gunakan masker untuk menghapus latar belakang
    result_img = img.copy()
    result_img[mask == 0] = [0, 0, 0]  # Set piksel latar belakang menjadi hitam

    # Menyimpan gambar tanpa latar belakang
    segmentation_image_path = os.path.join(app.config['UPLOAD'], 'apply_grabcut.jpg')
    cv2.imwrite(segmentation_image_path, result_img)
    

    return segmentation_image_path
        

@app.route("/segmentasi", methods=["GET", "POST"])
def segmentasi():
  
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("segmentasi.html")

        file = request.files["file"]

        # Check if the file name is empty
        if file.filename == "":
            return render_template("segmentasi.html")
        
        if file:
            # Simpan gambar yang diunggah ke folder uploads
            img_path = os.path.join(app.config["UPLOAD"], file.filename)
            file.save(img_path)

        # Call the function to remove the background
        segmentation_image_path = grabcut_segmentation(img_path)

        return render_template("segmentasi.html", original_image=img_path, segmentation_image=segmentation_image_path)

    return render_template("segmentasi.html")

import cv2

def cartoonify(Image_Path):
  """
  Mengkonversi gambar menjadi kartun.

  Args:
    Image_Path: Path ke gambar yang akan dikonversi.

  Returns:
    Gambar kartun yang telah dikonversi.
  """

  # Membaca gambar
  original_image = cv2.imread(Image_Path)

  # Memeriksa apakah gambar dapat ditemukan
  if original_image is None:
    print("Can not find any image. Choose appropriate file")
    return None

  # Mengubah ukuran gambar menjadi 4x6
  resized_image = cv2.resize(original_image, (472, 709))

  # Mengkonversi gambar resized_image ke mode warna BGR
  #resized_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR)
  
  # Melakukan median blur untuk menghaluskan gambar
  gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

  # Melakukan median blur untuk menghaluskan gambar
  smoothed_grayscale_image = cv2.medianBlur(gray, 5)

  # Mendapatkan tepi gambar untuk efek kartun
  edgess = cv2.adaptiveThreshold(smoothed_grayscale_image, 255,
                                 cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY, 9, 9)
  
  #edgess = cv2.Canny(resized_image, 100, 150)
  # Menerapkan bilateral filter untuk menghilangkan noise dan menjaga tepi tetap tajam
  color = cv2.bilateralFilter(resized_image, 9, 300, 300)

  # Melakukan masking gambar tepi dengan gambar warna
  cartoon_image = cv2.bitwise_and(color, color, mask=edgess)
  #cartoon_image = cv2.addWeighted(color, 0.9, edgess, 0.1, 0)

  # Menyimpan gambar hasil deteksi tepi ke folder "static/uploads"
  cartoon_image_path = os.path.join(app.config['UPLOAD'], 'kartunisasi.jpg')
  cv2.imwrite(cartoon_image_path, cartoon_image)

  return cartoon_image_path 

@app.route("/cartoon", methods=["GET", "POST"])
def kartun():
  """
  Mengupload dan mengkonversi gambar menjadi kartun.

  Returns:
    Template HTML yang menampilkan gambar asli dan gambar kartun yang telah dikonversi.
  """

  if request.method == "POST":
    # Mendapatkan file yang diunggah
    file = request.files["file"]

    # Memeriksa apakah file diunggah
    if file is None:
      return render_template("cartoonify.html")

    # Menyimpan file yang diunggah ke folder uploads
    ImagePath = os.path.join(app.config["UPLOAD"], file.filename)
    file.save(ImagePath)

    # Save the original image path for display
    original_image_path = ImagePath

    # Proses deteksi tepi untuk membuat sketsa pada gambar
    cartoon_image_path = cartoonify(ImagePath)

    # Mengembalikan template HTML yang menampilkan gambar asli dan gambar kartun yang telah dikonversi
    return render_template("cartoonify.html", original_image=original_image_path, cartoon_image=cartoon_image_path)

  else:

    # Mengembalikan template HTML kosong
    return render_template("cartoonify.html")

def median_filter(data, filter_size):
    temp = []
    indexer = filter_size // 2
    for i in range(len(data)):

        for j in range(len(data[0])):

            for z in range(filter_size):
                if i + z - indexer < 0 or i + z - indexer > len(data) - 1:
                    for c in range(filter_size):
                        temp.append(0)
                else:
                    if j + z - indexer < 0 or j + indexer > len(data[0]) - 1:
                        temp.append(0)
                    else:
                        for k in range(filter_size):
                            temp.append(data[i + z - indexer][j + k - indexer])

            temp.sort()
            data[i][j] = temp[len(temp) // 2]
            temp = []
    return data

def median():
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window

    # Open a file dialog for the user to select an image file
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff")])

    if not file_path:
        print("No image selected. Exiting.")
        return

    img = Image.open(file_path).convert("L")
    arr = np.array(img)

    # Apply median filter with a filter size of 3
    removed_noise = median_filter(arr, 3)

    # Create an image from the filtered data and display it
    filtered_img = Image.fromarray(removed_noise)
    filtered_img.show()
    median()


@app.route("/medianfilt", methods=["GET", "POST"])
def mediann():
    if request.method == "POST":
        # Check if the file exists.
        if "file" not in request.files:
            return render_template("medianfilter.html")

        file = request.files["file"]

        # Check if the file name is empty.
        if file.filename == "":
            return render_template("medianfilter.html")

        # Save the uploaded image to the uploads folder.
        imgg_path = os.path.join(app.config["UPLOAD"], file.filename)
        file.save(imgg_path)

        # Load the uploaded image.
        image = cv2.imread(imgg_path, cv2.IMREAD_GRAYSCALE)

        # Apply the median filter with a filter size of 3.
        image = median_filter(image, 3)
    
        # Menyimpan gambar tanpa latar belakang
        filtered_image_path = os.path.join(app.config['UPLOAD'], 'apply_medianfilt.jpg')
        cv2.imwrite(filtered_image_path, image)

        # Return the rendered template with the original and filtered image paths.
        return render_template("medianfilter.html", og_image=imgg_path, filtered_image=filtered_image_path)

    # Return the rendered template without the original and filtered image paths.
    return render_template("medianfilter.html")


def dilation_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Create a kernel
    kernel = np.ones((7, 7), np.uint8)

    # Perform dilation
    dilation = cv2.dilate(img, kernel, iterations=1)
    
    return dilation

@app.route('/dilation', methods=['GET', 'POST'])
def dilation():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            img_path = os.path.join(app.config['UPLOAD'], filename)
            file.save(img_path)
                
            dilated_image = dilation_image(img_path)
            dilation_image_path = os.path.join(app.config['UPLOAD'], 'eroded_' + filename)
            cv2.imwrite(dilation_image_path, dilated_image)
            return render_template('dilation.html', original_image=img_path, dilation_image=dilation_image_path)
    
    return render_template('dilation.html')

def erode_image(image_path):
    img = cv2.imread(image_path, 0)

    # Membuat kernel
    kernel = np.ones((5, 5), np.uint8)

    # Erosi gambar
    erosion = cv2.erode(img, kernel, iterations=1) #Untuk erosi dan dilasi diubah pada cv2.erode dan cv2.dilate
    
    return erosion

@app.route('/erosion', methods=['GET', 'POST'])
def erosion():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            img_path = os.path.join(app.config['UPLOAD'], filename)
            file.save(img_path)
                
            eroded_image = erode_image(img_path)
            eroded_image_path = os.path.join(app.config['UPLOAD'], 'eroded_' + filename)
            cv2.imwrite(eroded_image_path, eroded_image)
            
        return render_template('erosion.html', original_image=img_path, erosion_image=eroded_image_path)
    
    return render_template('erosion.html')

@app.route('/opening', methods=['GET', 'POST'])
def opening_image():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            img_path = os.path.join(app.config['UPLOAD'], filename)
            
            file.save(img_path)

        # Perform morphological opening
        img = cv2.imread(img_path)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binarized_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        kernel = np.ones((5, 5), np.uint8)
        opening = cv2.morphologyEx(binarized_img, cv2.MORPH_OPEN, kernel, iterations=1)

        # Save the result to a file
        opening_image_path = os.path.join(app.config['UPLOAD'], 'opening_image.jpg')
        cv2.imwrite(opening_image_path, opening)

        return render_template('opening.html', original_image=img_path, opening_image=opening_image_path)
    return render_template('opening.html')

@app.route('/closing', methods=['GET', 'POST'])
def closing():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            img_path = os.path.join(app.config['UPLOAD'], filename)
            file.save(img_path)

        # Perform morphological closing
        img = cv2.imread(img_path)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binarized_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        kernel = np.ones((3, 3), np.uint8)
        closing = cv2.morphologyEx(binarized_img, cv2.MORPH_CLOSE, kernel, iterations=4)

        # Save the result to a file
        closing_image_path = os.path.join(app.config['UPLOAD'], 'closing_image.jpg')
        cv2.imwrite(closing_image_path, closing)

        return render_template('closing.html', original_image=img_path, closing_image=closing_image_path)
    return render_template('closing.html')


@app.route('/nearest', methods=['GET', 'POST'])
def nearest():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            img_path = os.path.join(app.config['UPLOAD'], filename)
            file.save(img_path)

        img = cv2.imread(img_path)

         # Mendefinisikan scale
        scale_x = float(request.form['scale_x'])  
        scale_y = float(request.form['scale_y'])  

        # scaling menggunakan bicubic interpolation
        nearest_image = cv2.resize(img, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_NEAREST)

        # Menyimpan gambar
        nearest_image_path = os.path.join(app.config['UPLOAD'], 'scaled_image.jpg')
        cv2.imwrite(nearest_image_path, nearest_image)

        img_matrix = cv2.imread(nearest_image_path, 0)  # Mode 0 untuk grayscale, 1 untuk RGB
        img_matrix_list = img_matrix.tolist()

        return render_template('nearest.html', original_image=img_path, nearest_image=nearest_image_path, nearest_image_data=img_matrix_list)
    return render_template('nearest.html')


@app.route('/bilinear', methods=['GET', 'POST'])
def bilinear():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            img_path = os.path.join(app.config['UPLOAD'], filename)
            file.save(img_path)

        img = cv2.imread(img_path)

         # Mendefinisikan scale
        scale_x = float(request.form['scale_x'])  
        scale_y = float(request.form['scale_y'])  
        
        # scaling menggunakan bilinear interpolation
        scaled_img = cv2.resize(img, None, fx=scale_x, fy=scale_y, 
                                interpolation=cv2.INTER_LINEAR)

        # Menyimpan gambar
        scaled_image_path = os.path.join(app.config['UPLOAD'], 'scaled_image.jpg')
        cv2.imwrite(scaled_image_path, scaled_img)

        img_matrix = cv2.imread(scaled_image_path, 0)  # Mode 0 untuk grayscale, 1 untuk RGB
        img_matrix_list = img_matrix.tolist()

        return render_template('bilinear.html', original_image=img_path, 
                               scaled_image=scaled_image_path, 
                               scaled_image_data=img_matrix_list)
    return render_template('bilinear.html')

@app.route('/bicubic', methods=['GET', 'POST'])
def bicubic():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            img_path = os.path.join(app.config['UPLOAD'], filename)
            file.save(img_path)

        img = cv2.imread(img_path)

         # Mendefinisikan scale
        scale_x = float(request.form['scale_x'])  
        scale_y = float(request.form['scale_y'])  

        # scaling menggunakan bicubic interpolation
        bicubic_image = cv2.resize(img, None, fx=scale_x, fy=scale_y, 
                                   interpolation=cv2.INTER_CUBIC)

        # # Menyimpan gambar
        bicubic_image_path = os.path.join(app.config['UPLOAD'], 'scaled_image.jpg')
        cv2.imwrite(bicubic_image_path, bicubic_image)

        img_matrix = cv2.imread(bicubic_image_path, 0)  # Mode 0 untuk grayscale, 1 untuk RGB
        img_matrix_list = img_matrix.tolist()

        return render_template('bicubic.html', original_image=img_path, 
                               bicubic_image=bicubic_image_path, 
                               bicubic_image_data=img_matrix_list)
    return render_template('bicubic.html')

codeList = [5, 6, 7, 4, -1, 0, 3, 2, 1]

def getChainCode(dx, dy):
    hashKey = (3 * dy + dx + 4) % len(codeList)
    chainCode = codeList[hashKey]

    # Ensure the chain code is not negative
    if chainCode < 0:
        chainCode += len(codeList)

    return chainCode

def generate_chain_code(ListOfPoints):
    chainCode = []
    for i in range(len(ListOfPoints) - 1):
        a = ListOfPoints[i]
        b = ListOfPoints[i + 1]
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        chainCode.append(getChainCode(dx, dy))
    return chainCode

def calculate_chain_code_from_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:  
        largest_contour = max(contours, key=cv2.contourArea)
        
        if largest_contour.size > 0:  
            chain_code = generate_chain_code([point[0] for point in largest_contour])
            return chain_code
        else:
            return None  
    else:
        return None  

@app.route('/chain_code', methods=['GET', 'POST'])
def calculate_chain_code():
    if request.method == 'POST':
        file = request.files['img']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD'], filename))
        image_path = os.path.join(app.config['UPLOAD'], filename)

        image = cv2.imread(image_path)
        
        chain_code = calculate_chain_code_from_image(image)
        return render_template('chain_code.html', original_image=image_path, 
                               chain_code=chain_code)
    return render_template('chain_code.html')

def rank_order_filter(img, footprint):
    # Convert to grayscale as the rank order filter example seems designed for single-channel
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    
    # Apply the rank order median filter
    filtered_img = ndimage.median_filter(gray_img, footprint=footprint)
    
    # Convert back to BGR for consistency with input
    filtered_img_bgr = cv2.cvtColor(filtered_img, cv2.COLOR_GRAY2BGR)
    
    return filtered_img_bgr

def outlier_method(img, D=0.2):
    # Convert to grayscale as the algorithm seems designed for single-channel
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Kernel for neighborhood averaging
    av = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]) / 8.0
    
    # Apply neighborhood averaging
    mean_neighbors = ndimage.convolve(gray_img, av, mode='nearest')

    # Detecting outliers
    outliers = np.abs(gray_img - mean_neighbors) > D

    # Replacing outlier pixel values with the mean of their neighbors
    gray_img[outliers] = mean_neighbors[outliers]

    return gray_img

@app.route('/remove_noise', methods=['GET', 'POST'])
def remove_noise():
    if request.method == 'POST':
        file = request.files['img']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD'], filename))
        img_path = os.path.join(app.config['UPLOAD'], filename)

        img = cv2.imread(img_path)
        
        if request.form.get('filter_type') == 'median':
            # Membersihkan salt and pepper noise menggunakan filter median
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            median_img = cv2.medianBlur(gray_img, 5)  

            # Menyimpan gambar yang telah dibersihkan
            median_clean_img = os.path.join(app.config['UPLOAD'], 
                                            'cleaned_image.jpg')
            cv2.imwrite(median_clean_img, median_img)

            filter_type = request.form.get('filter_type', None)

            return render_template('remove_noise.html', img=img_path, 
                                   img2=median_clean_img, 
                                   filter_type=filter_type)
        
        if request.form.get('filter_type') == 'lowpass':
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Membersihkan gambar menggunakan filter low-pass (Gaussian blur)
            lowpass_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

            # Menyimpan gambar yang telah dibersihkan
            lowpass_clean_img = os.path.join(app.config['UPLOAD'], 
                                             'cleaned_image_lowpass.jpg')
            cv2.imwrite(lowpass_clean_img, lowpass_img)

            filter_type = request.form.get('filter_type', None)

            return render_template('remove_noise.html', img=img_path, 
                                   img2=lowpass_clean_img, 
                                   filter_type=filter_type)
        
        if request.form.get('filter_type') == 'rank-order':
            # Define the non-rectangular mask (cross-shaped)
            cross = np.array([[0,1,0],[1,1,1],[0,1,0]])

            # Apply the rank order filter
            filtered_img = rank_order_filter(img, cross)

            # Save the filtered image
            filtered_img_path = os.path.join(app.config['UPLOAD'], 
                                             'filtered_image.jpg')
            cv2.imwrite(filtered_img_path, filtered_img)

            filter_type = 'rank-order'

            return render_template('remove_noise.html', img=img_path, 
                                   img2=filtered_img_path, 
                                   filter_type=filter_type)
        
        if request.form.get('filter_type') == 'outlier':
            # Apply the outlier method
            cleaned_img = outlier_method(img)

            # Save the cleaned image
            cleaned_img_path = os.path.join(app.config['UPLOAD'], 
                                            'cleaned_image.jpg')
            cv2.imwrite(cleaned_img_path, cleaned_img)

            filter_type = 'outlier'

            return render_template('remove_noise.html', img=img_path, 
                                   img2=cleaned_img_path, 
                                   filter_type=filter_type)
    
    return render_template('remove_noise.html')

def add_salt_and_pepper_noise(image, salt_prob, pepper_prob):
    noisy_image = np.copy(image)

    # Add salt noise
    salt_mask = np.random.rand(*image.shape)
    noisy_image[salt_mask < salt_prob] = 255  # white in grayscale

    # Add pepper noise
    pepper_mask = np.random.rand(*image.shape)
    noisy_image[pepper_mask < pepper_prob] = 0  # black in grayscale

    return noisy_image

def add_gaussian_noise(image, mean=0, std=25):
    # Check if the image is grayscale
    if len(image.shape) == 2:
        gauss = np.random.normal(mean, std, image.shape)
        noisy = image + gauss
        noisy = np.clip(noisy, 0, 255)
        return noisy.astype(np.uint8)
    # If the image is RGB, apply noise to each channel separately
    elif len(image.shape) == 3:
        noisy_image = np.copy(image)
        for channel in range(image.shape[2]):
            gauss = np.random.normal(mean, std, image.shape[:2])
            noisy_image[:, :, channel] = image[:, :, channel] + gauss
        noisy_image = np.clip(noisy_image, 0, 255)
        return noisy_image.astype(np.uint8)
    else:
        raise ValueError("Unsupported image shape")

def add_speckle_noise(image, mean=0, std=0.5):
    row, col = image.shape[:2] # Dapatkan dimensi gambar
    speckle = np.random.normal(mean, std, (row, col)) # Generate speckle noise dengan distribusi normal
    noisy = image + image * speckle # Apply speckle noise pada gambar
    noisy = np.clip(noisy, 0, 255)  # Clip nilai ke rentang yang valid [0, 255]
    return noisy.astype(np.uint8)

def add_periodic_noise(image, frequency=10, amplitude=50):
    row, col = image.shape[:2]
    x, y = np.meshgrid(np.arange(col), np.arange(row))  # Switched col and row here
    noise = amplitude * np.sin(2 * np.pi * frequency * (x + y) / (row + col))

    if len(image.shape) == 2:  # Grayscale image
        noisy_image = np.clip(image + noise[:row, :col], 0, 255).astype(np.uint8)
    elif len(image.shape) == 3:  # RGB image
        noisy_image = np.clip(image + noise[:row, :col, np.newaxis], 0, 255).astype(np.uint8)
    else:
        raise ValueError("Unsupported image shape")

    return noisy_image

@app.route('/adding_noise', methods=['GET', 'POST'])
def adding_noise():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            img_path = os.path.join(app.config['UPLOAD'], filename)
            file.save(img_path)
            
            # Read image
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            noise_type = request.form.get('noise_type', None)

            if noise_type == 'salt_and_pepper':
                # Add noise salt-and-pepper
                noisy_image = add_salt_and_pepper_noise(image, salt_prob=0.02, pepper_prob=0.02)
            elif noise_type == 'gaussian':
                # Add Gaussian noise
                noisy_image = add_gaussian_noise(image)
            elif noise_type == 'speckle':
                # Add Speckle noise
                noisy_image = add_speckle_noise(image)
            elif noise_type == 'periodic':
                # Add Periodic noise
                noisy_image = add_periodic_noise(image)
            
            # Saving image
            noisy_image_path = f'static/noisy_image_{noise_type}.png'
            cv2.imwrite(noisy_image_path, noisy_image)
            
            return render_template('addnoise.html', original_image=img_path, 
                                   noisy_image=noisy_image_path, noise_type=noise_type)

    return render_template('addnoise.html')

# Huffman Coding Algorithm Implementation
class Node:
    left = None
    right = None
    item = None
    weight = 0

    def __init__(self, symbol, weight, l=None, r=None):
        self.symbol = symbol
        self.weight = weight
        self.left = l
        self.right = r

    def __repr__(self):
        return '("%s", %s, %s, %s)' % (self.symbol, self.weight, 
                                       self.left, self.right)

def sort_by_weight(node):
    return (node.weight * 1000000 + ord(node.symbol[0]))

class HuffmanEncoder:
    def __init__(self):
        self.symbols = {}
        self.codes = {}
        self.tree = []
        self.message = ""

    def frequency_analysis(self):
        self.symbols = {}
        for symbol in self.message:
            self.symbols[symbol] = self.symbols.get(symbol, 0) + 1

    def preorder_traverse(self, node, path=""):
        if node.left is None:
            self.codes[node.symbol] = path
        else:
            self.preorder_traverse(node.left, path + "0")
            self.preorder_traverse(node.right, path + "1")

    def encode(self, message):
        self.message = message
        self.frequency_analysis()
        self.tree = []
        for symbol in self.symbols.keys():
            self.tree.append(Node(symbol, self.symbols[symbol], None, None))

        self.tree.sort(key=sort_by_weight)

        while len(self.tree) > 1:
            left_node = self.tree.pop(0)
            right_node = self.tree.pop(0)
            new_node = Node(left_node.symbol + right_node.symbol, 
                            left_node.weight + right_node.weight, left_node,
                            right_node)
            self.tree.append(new_node)
            self.tree.sort(key=sort_by_weight)

        self.codes = {}
        self.preorder_traverse(self.tree[0])

        encoded_message = ""
        for symbol in message:
            encoded_message = encoded_message + self.codes[symbol]

        return encoded_message

@app.route('/compress', methods=['GET', 'POST'])
def compress():
    if request.method == 'POST':
        message = request.form['message']
        encoder = HuffmanEncoder()
        compressed_message = encoder.encode(message)
        result = {"message": message, "compressed_message": compressed_message}

        huffman_codes = encoder.codes
        return render_template('compress.html', compressed_message=compressed_message, 
                               huffman_codes=huffman_codes)

    return render_template('compress.html')


if __name__ == '__main__': 
    app.run(debug=True,port=8001)