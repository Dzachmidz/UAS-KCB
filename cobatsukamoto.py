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
kelembaban_normal = fuzz.trimf(kelembaban, [0, 50, 50])
kelembaban_tinggi = fuzz.trimf(kelembaban, [50, 100, 100])

# Pembentukan fungsi keanggotaan pH tanah
ph_asam = fuzz.trimf(ph_tanah, [4, 4, 6])
ph_netral = fuzz.trimf(ph_tanah, [4, 6, 8])
ph_basa = fuzz.trimf(ph_tanah, [6, 8, 8])

# Pembentukan fungsi keanggotaan tingkat penyakit
diagnosis_rendah = fuzz.trimf(diagnosis, [0, 0, 50])
diagnosis_sedang = fuzz.trimf(diagnosis, [0, 50, 100])
diagnosis_tinggi = fuzz.trimf(diagnosis, [50, 100, 100])

# Definisi dataset
dataset = [
    [28, 55, 6.5],
    [30, 65, 7],
    [26, 70, 6.8],
    [32, 85, 6.2],
    [27, 80, 7.5],
    [29, 75, 6.7],
    [31, 90, 6.9],
    [25, 60, 6],
    [30, 80, 7.3],
    [27, 70, 6.5],
    [31, 75, 7.2],
    [28, 75, 6.8],
    [32, 80, 6.2],
    [26, 65, 7],
    [29, 85, 6.5],
    [27, 72, 6.9],
    [31, 78, 6.4],
    [24, 68, 7.1],
    [33, 76, 6.3]
]


# Melakukan perhitungan fuzzy dan mencetak hasil diagnosis untuk setiap data
for data in dataset:
    suhu_input = data[0]
    kelembaban_input = data[1]
    ph_tanah_input = data[2]
    # Proses fuzzifikasi
    suhu_rendah_degree = fuzz.interp_membership(suhu, suhu_rendah, suhu_input)
    suhu_normal_degree = fuzz.interp_membership(suhu, suhu_normal, suhu_input)
    suhu_tinggi_degree = fuzz.interp_membership(suhu, suhu_tinggi, suhu_input)

    kelembaban_rendah_degree = fuzz.interp_membership(kelembaban, kelembaban_rendah, kelembaban_input)
    kelembaban_normal_degree = fuzz.interp_membership(kelembaban, kelembaban_normal, kelembaban_input)
    kelembaban_tinggi_degree = fuzz.interp_membership(kelembaban, kelembaban_tinggi, kelembaban_input)

    ph_asam_degree = fuzz.interp_membership(ph_tanah, ph_asam, ph_tanah_input)
    ph_netral_degree = fuzz.interp_membership(ph_tanah, ph_netral, ph_tanah_input)
    ph_basa_degree = fuzz.interp_membership(ph_tanah, ph_basa, ph_tanah_input)

    # Inferensi dengan menggunakan metode Tsukamoto
    # Rule 1: IF suhu rendah AND kelembaban rendah AND pH asam THEN tingkat penyakit rendah
    rule1 = np.fmin(suhu_rendah_degree, np.fmin(kelembaban_rendah_degree, ph_asam_degree))
    diagnosis_output1 = np.fmin(rule1, diagnosis_rendah)

    # Rule 2: IF suhu normal AND kelembaban normal AND pH netral THEN tingkat penyakit rendah
    rule2 = np.fmin(suhu_normal_degree, np.fmin(kelembaban_normal_degree, ph_netral_degree))
    diagnosis_output2 = np.fmin(rule2, diagnosis_sedang)

    # Rule 3: IF suhu rendahsuhu_rendah AND kelembaban tinggi AND pH basa THEN tingkat penyakit tinggi
    rule3 = np.fmin(suhu_tinggi_degree, np.fmin(kelembaban_tinggi_degree, ph_basa_degree))
    diagnosis_output3 = np.fmin(rule3, diagnosis_tinggi)

    rule4 = np.fmin(suhu_rendah_degree, np.fmin(kelembaban_normal_degree, ph_asam_degree))
    diagnosis_output4 = np.fmin(rule4, diagnosis_rendah)

    rule5 = np.fmin(suhu_normal_degree, np.fmin(kelembaban_tinggi_degree, ph_netral_degree))
    diagnosis_output5 = np.fmin(rule5, diagnosis_tinggi)

    rule6 = np.fmin(suhu_rendah_degree, np.fmin(kelembaban_rendah_degree, ph_netral_degree))
    diagnosis_output6 = np.fmin(rule6, diagnosis_rendah)

    rule7 = np.fmin(suhu_rendah_degree, np.fmin(kelembaban_tinggi_degree, ph_asam_degree))
    diagnosis_output7 = np.fmin(rule7, diagnosis_sedang)

    rule8 = np.fmin(suhu_rendah_degree, np.fmin(kelembaban_normal_degree, ph_basa_degree))
    diagnosis_output8 = np.fmin(rule8, diagnosis_sedang)

    rule9 = np.fmin(suhu_rendah_degree, np.fmin(kelembaban_tinggi_degree, ph_basa_degree))
    diagnosis_output9 = np.fmin(rule9, diagnosis_tinggi)

    rule10 = np.fmin(suhu_normal_degree, np.fmin(kelembaban_rendah_degree, ph_asam_degree))
    diagnosis_output10 = np.fmin(rule10, diagnosis_rendah)

    rule11 = np.fmin(suhu_normal_degree, np.fmin(kelembaban_normal_degree, ph_asam_degree))
    diagnosis_output11 = np.fmin(rule11, diagnosis_rendah)

    rule12 = np.fmin(suhu_normal_degree, np.fmin(kelembaban_tinggi_degree, ph_asam_degree))
    diagnosis_output12 = np.fmin(rule12, diagnosis_sedang)

    rule13 = np.fmin(suhu_normal_degree, np.fmin(kelembaban_rendah_degree, ph_netral_degree))
    diagnosis_output13 = np.fmin(rule13, diagnosis_rendah)

    rule14 = np.fmin(suhu_normal_degree, np.fmin(kelembaban_rendah_degree, ph_basa_degree))
    diagnosis_output14 = np.fmin(rule14, diagnosis_rendah)

    rule15 = np.fmin(suhu_normal_degree, np.fmin(kelembaban_normal_degree, ph_basa_degree))
    diagnosis_output15 = np.fmin(rule15, diagnosis_tinggi)

    rule16 = np.fmin(suhu_normal_degree, np.fmin(kelembaban_tinggi_degree, ph_basa_degree))
    diagnosis_output16 = np.fmin(rule16, diagnosis_tinggi)

    rule17 = np.fmin(suhu_tinggi_degree, np.fmin(kelembaban_rendah_degree, ph_asam_degree))
    diagnosis_output17 = np.fmin(rule17, diagnosis_rendah)

    rule18 = np.fmin(suhu_tinggi_degree, np.fmin(kelembaban_normal_degree, ph_asam_degree))
    diagnosis_output18 = np.fmin(rule18, diagnosis_sedang)

    rule19 = np.fmin(suhu_tinggi_degree, np.fmin(kelembaban_tinggi_degree, ph_asam_degree))
    diagnosis_output19 = np.fmin(rule19, diagnosis_tinggi)

    rule20 = np.fmin(suhu_tinggi_degree, np.fmin(kelembaban_rendah_degree, ph_netral_degree))
    diagnosis_output20 = np.fmin(rule20, diagnosis_rendah)

    rule21 = np.fmin(suhu_tinggi_degree, np.fmin(kelembaban_normal_degree, ph_netral_degree))
    diagnosis_output21 = np.fmin(rule21, diagnosis_sedang)

    rule22 = np.fmin(suhu_tinggi_degree, np.fmin(kelembaban_tinggi_degree, ph_netral_degree))
    diagnosis_output22 = np.fmin(rule22, diagnosis_tinggi)

    rule23 = np.fmin(suhu_tinggi_degree, np.fmin(kelembaban_rendah_degree, ph_basa_degree))
    diagnosis_output23 = np.fmin(rule23, diagnosis_sedang)

    rule24 = np.fmin(suhu_tinggi_degree, np.fmin(kelembaban_normal_degree, ph_basa_degree))
    diagnosis_output24 = np.fmin(rule24, diagnosis_tinggi)

    rule25 = np.fmin(suhu_rendah_degree, np.fmin(kelembaban_normal_degree, ph_netral_degree))
    diagnosis_output25 = np.fmin(rule25, diagnosis_sedang)

    rule26 = np.fmin(suhu_rendah_degree, np.fmin(kelembaban_tinggi_degree, ph_netral_degree))
    diagnosis_output26 = np.fmin(rule26, diagnosis_tinggi)

    rule27 = np.fmin(suhu_rendah_degree, np.fmin(kelembaban_rendah_degree, ph_basa_degree))
    diagnosis_output27 = np.fmin(rule27, diagnosis_rendah)

    # Penggabungan output dari semua aturan
    diagnosis_agg = np.fmax(diagnosis_output1,
                        np.fmax(diagnosis_output2,
                                np.fmax(diagnosis_output3,
                                        np.fmax(diagnosis_output4,
                                                np.fmax(diagnosis_output5,
                                                        np.fmax(diagnosis_output6,
                                                                np.fmax(diagnosis_output7,
                                                                        np.fmax(diagnosis_output8,
                                                                                np.fmax(diagnosis_output9,
                                                                                        np.fmax(diagnosis_output10,
                                                                                                np.fmax(diagnosis_output11,
                                                                                                        np.fmax(diagnosis_output12,
                                                                                                                np.fmax(diagnosis_output13,
                                                                                                                        np.fmax(diagnosis_output14,
                                                                                                                                np.fmax(diagnosis_output15,
                                                                                                                                        np.fmax(diagnosis_output16,
                                                                                                                                                np.fmax(diagnosis_output17,
                                                                                                                                                        np.fmax(diagnosis_output18,
                                                                                                                                                                np.fmax(diagnosis_output19,
                                                                                                                                                                        np.fmax(diagnosis_output20,
                                                                                                                                                                                np.fmax(diagnosis_output21,
                                                                                                                                                                                        np.fmax(diagnosis_output22,
                                                                                                                                                                                                np.fmax(diagnosis_output23,
                                                                                                                                                                                                        np.fmax(diagnosis_output24, diagnosis_output25,
                                                                                                                                                                                                                np.fmax(diagnosis_output26, diagnosis_output27)))))))))))))))))))))))))
                                                                                                                                                                                                        
                                                                                                                                                                                                                        
# Defuzzifikasi menggunakan metode Tsukamoto
diagnosis_result = fuzz.defuzz(diagnosis, diagnosis_agg, 'centroid')
diagnosis_result = round(diagnosis_result, 2)

# Menampilkan hasil diagnosis
for i in range(n_data):
    print("Data", i+1, ":", data)
    print("Hasil Diagnosis Menggunakan Metode Tsukamoto =", hasil_diagnosis[i])
