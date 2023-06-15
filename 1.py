import numpy as np
import skfuzzy as fuzz

# Fuzzifikasi variabel masukan
suhu = np.arange(20, 41, 1)
kelembaban = np.arange(40, 101, 1)
ph_tanah = np.arange(4, 9, 1)
diagnosis = np.arange(0, 101, 1)

# Pembentukan fungsi keanggotaan suhu
suhu_rendah = fuzz.trimf(suhu, [20, 20, 30])
suhu_normal = fuzz.trimf(suhu, [20, 30, 40])
suhu_tinggi = fuzz.trimf(suhu, [30, 40, 40])

# Pembentukan fungsi keanggotaan kelembaban
kelembaban_rendah = fuzz.trimf(kelembaban, [0, 0, 50])
kelembaban_normal = fuzz.trimf(kelembaban, [0, 50, 100])
kelembaban_tinggi = fuzz.trimf(kelembaban, [50, 100, 100])

# Pembentukan fungsi keanggotaan pH tanah
ph_asam = fuzz.trimf(ph_tanah, [4, 4, 6])
ph_netral = fuzz.trimf(ph_tanah, [4, 6, 8])
ph_basa = fuzz.trimf(ph_tanah, [6, 8, 8])

# Pembentukan fungsi keanggotaan tingkat penyakit
diagnosis_rendah = fuzz.trimf(diagnosis, [0, 0, 50])
diagnosis_sedang = fuzz.trimf(diagnosis, [0, 50, 100])
diagnosis_tinggi = fuzz.trimf(diagnosis, [50, 100, 100])

# Dataset
dataset = [
    [28, 75, 6.5, 55.73],
    [32, 85, 7.2, 61.62],
    [26, 65, 5.8, 50.00],
    [30, 80, 6.9, 58.01],
    [27, 70, 6.2, 52.99],
    [31, 90, 7.5, 66.97],
    [29, 75, 6.7, 55.73],
    [33, 80, 7.0, 55.04],
    [25, 60, 6.0, 50.00],
    [30, 85, 7.3, 61.62],
    [25, 70, 6.5, 50.00],
    [30, 60, 7.2, 58.87],
    [28, 75, 6.8, 55.73],
    [32, 80, 6.2, 58.87],
    [26, 65, 7.0, 55.73],
    [29, 85, 6.5, 61.62],
    [27, 72, 6.9, 50.00],
    [31, 78, 6.4, 61.62],
    [24, 68, 7.1, 50.00],
    [33, 76, 6.3, 61.62]
]

# Menghitung jumlah data pada dataset
n_data = len(dataset)

# Inisialisasi variabel hasil
hasil_diagnosis = []

# Proses inferensi dan defuzzifikasi
for data in dataset:
    suhu_input = data[0]
    kelembaban_input = data[1]
    ph_tanah_input = data[2]

    # Inferensi
    suhu_level_rendah = fuzz.interp_membership(suhu, suhu_rendah, suhu_input)
    suhu_level_normal = fuzz.interp_membership(suhu, suhu_normal, suhu_input)
    suhu_level_tinggi = fuzz.interp_membership(suhu, suhu_tinggi, suhu_input)

    kelembaban_level_rendah = fuzz.interp_membership(kelembaban, kelembaban_rendah, kelembaban_input)
    kelembaban_level_normal = fuzz.interp_membership(kelembaban, kelembaban_normal, kelembaban_input)
    kelembaban_level_tinggi = fuzz.interp_membership(kelembaban, kelembaban_tinggi, kelembaban_input)

    ph_level_asam = fuzz.interp_membership(ph_tanah, ph_asam, ph_tanah_input)
    ph_level_netral = fuzz.interp_membership(ph_tanah, ph_netral, ph_tanah_input)
    ph_level_basa = fuzz.interp_membership(ph_tanah, ph_basa, ph_tanah_input)

    # Rules
    # Rule 1: Jika suhu rendah DAN kelembaban rendah DAN pH tanah asam, maka diagnosis tinggi
    rule1 = np.fmin(np.fmin(suhu_level_rendah, kelembaban_level_rendah), ph_level_asam)
    diagnosis_activation1 = np.fmin(rule1, diagnosis_tinggi)

    # Rule 2: Jika suhu tinggi DAN kelembaban tinggi DAN pH tanah basa, maka diagnosis rendah
    rule2 = np.fmin(np.fmin(suhu_level_tinggi, kelembaban_level_tinggi), ph_level_basa)
    diagnosis_activation2 = np.fmin(rule2, diagnosis_rendah)

    # Rule 3: Jika suhu normal DAN kelembaban normal DAN pH tanah netral, maka diagnosis sedang
    rule3 = np.fmin(np.fmin(suhu_level_normal, kelembaban_level_normal), ph_level_netral)
    diagnosis_activation3 = np.fmin(rule3, diagnosis_sedang)

    # Agregasi hasil inferensi
    aggregated = np.fmax(diagnosis_activation1, np.fmax(diagnosis_activation2, diagnosis_activation3))

    # Defuzzifikasi menggunakan metode Tsukamoto
    diagnosis_defuzz = fuzz.defuzz(diagnosis, aggregated, 'centroid')
    diagnosis_result = round(diagnosis_defuzz, 2)

    # Menambahkan hasil diagnosis ke dalam variabel hasil
    hasil_diagnosis.append(diagnosis_result)

# Menampilkan hasil diagnosis
for i in range(n_data):
    print("Data", i+1, ": Tingkat Penyakit Blast pada Padi =", hasil_diagnosis[i])
