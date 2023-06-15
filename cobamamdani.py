import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# Mendefinisikan variabel linguistik
suhu = ctrl.Antecedent(np.arange(20, 41, 1), 'suhu')
ph_tanah = ctrl.Antecedent(np.arange(4, 9, 1), 'ph_tanah')
kelembaban = ctrl.Antecedent(np.arange(0, 101, 1), 'kelembaban')
diagnosis = ctrl.Consequent(np.arange(0, 101, 1), 'diagnosis')

# Mendefinisikan fungsi keanggotaan untuk variabel suhu
suhu['rendah'] = fuzz.trimf(suhu.universe, [20, 20, 30])
suhu['normal'] = fuzz.trimf(suhu.universe, [20, 30, 40])
suhu['tinggi'] = fuzz.trimf(suhu.universe, [30, 40, 40])

# Mendefinisikan fungsi keanggotaan untuk variabel pH tanah
ph_tanah['asam'] = fuzz.trimf(ph_tanah.universe, [4, 4, 6])
ph_tanah['netral'] = fuzz.trimf(ph_tanah.universe, [4, 6, 8])
ph_tanah['basah'] = fuzz.trimf(ph_tanah.universe, [6, 8, 8])

# Mendefinisikan fungsi keanggotaan untuk variabel kelembaban
kelembaban['rendah'] = fuzz.trimf(kelembaban.universe, [0, 0, 50])
kelembaban['normal'] = fuzz.trimf(kelembaban.universe, [0, 50, 100])
kelembaban['tinggi'] = fuzz.trimf(kelembaban.universe, [50, 100, 100])

# Mendefinisikan fungsi keanggotaan untuk variabel diagnosis
diagnosis['rendah'] = fuzz.trimf(diagnosis.universe, [0, 0, 50])
diagnosis['sedang'] = fuzz.trimf(diagnosis.universe, [0, 50, 100])
diagnosis['tinggi'] = fuzz.trimf(diagnosis.universe, [50, 100, 100])

# Membuat aturan-aturan
rule1 = ctrl.Rule(suhu['rendah'] & ph_tanah['asam'] & kelembaban['rendah'], diagnosis['rendah'])
rule2 = ctrl.Rule(suhu['rendah'] & ph_tanah['asam'] & kelembaban['normal'], diagnosis['rendah'])
rule3 = ctrl.Rule(suhu['rendah'] & ph_tanah['asam'] & kelembaban['tinggi'], diagnosis['sedang'])

rule4 = ctrl.Rule(suhu['rendah'] & ph_tanah['netral'] & kelembaban['rendah'], diagnosis['rendah'])
rule5 = ctrl.Rule(suhu['rendah'] & ph_tanah['netral'] & kelembaban['normal'], diagnosis['sedang'])
rule6 = ctrl.Rule(suhu['rendah'] & ph_tanah['netral'] & kelembaban['tinggi'], diagnosis['tinggi'])

rule7 = ctrl.Rule(suhu['rendah'] & ph_tanah['basah'] & kelembaban['rendah'], diagnosis['rendah'])
rule8 = ctrl.Rule(suhu['rendah'] & ph_tanah['basah'] & kelembaban['normal'], diagnosis['sedang'])
rule9 = ctrl.Rule(suhu['rendah'] & ph_tanah['basah'] & kelembaban['tinggi'], diagnosis['tinggi'])

rule10 = ctrl.Rule(suhu['normal'] & ph_tanah['asam'] & kelembaban['rendah'], diagnosis['rendah'])
rule11 = ctrl.Rule(suhu['normal'] & ph_tanah['asam'] & kelembaban['normal'], diagnosis['rendah'])
rule12 = ctrl.Rule(suhu['normal'] & ph_tanah['asam'] & kelembaban['tinggi'], diagnosis['sedang'])

rule13 = ctrl.Rule(suhu['normal'] & ph_tanah['netral'] & kelembaban['rendah'], diagnosis['rendah'])
rule14 = ctrl.Rule(suhu['normal'] & ph_tanah['netral'] & kelembaban['normal'], diagnosis['sedang'])
rule15 = ctrl.Rule(suhu['normal'] & ph_tanah['netral'] & kelembaban['tinggi'], diagnosis['tinggi'])

rule16 = ctrl.Rule(suhu['normal'] & ph_tanah['basah'] & kelembaban['rendah'], diagnosis['rendah'])
rule17 = ctrl.Rule(suhu['normal'] & ph_tanah['basah'] & kelembaban['normal'], diagnosis['tinggi'])
rule18 = ctrl.Rule(suhu['normal'] & ph_tanah['basah'] & kelembaban['tinggi'], diagnosis['tinggi'])

rule19 = ctrl.Rule(suhu['tinggi'] & ph_tanah['asam'] & kelembaban['rendah'], diagnosis['rendah'])
rule20 = ctrl.Rule(suhu['tinggi'] & ph_tanah['asam'] & kelembaban['normal'], diagnosis['sedang'])
rule21 = ctrl.Rule(suhu['tinggi'] & ph_tanah['asam'] & kelembaban['tinggi'], diagnosis['tinggi'])

rule22 = ctrl.Rule(suhu['tinggi'] & ph_tanah['netral'] & kelembaban['rendah'], diagnosis['rendah'])
rule23 = ctrl.Rule(suhu['tinggi'] & ph_tanah['netral'] & kelembaban['normal'], diagnosis['sedang'])
rule24 = ctrl.Rule(suhu['tinggi'] & ph_tanah['netral'] & kelembaban['tinggi'], diagnosis['tinggi'])

rule25 = ctrl.Rule(suhu['tinggi'] & ph_tanah['basah'] & kelembaban['rendah'], diagnosis['sedang'])
rule26 = ctrl.Rule(suhu['tinggi'] & ph_tanah['basah'] & kelembaban['normal'], diagnosis['tinggi'])
rule27 = ctrl.Rule(suhu['tinggi'] & ph_tanah['basah'] & kelembaban['tinggi'], diagnosis['tinggi'])

# Membuat sistem fuzzy
diagnosis_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11, rule12, rule13, rule14, rule15, rule16, rule17, rule18, rule19, rule20, rule21, rule22, rule23, rule24, rule25, rule26, rule27])
diagnosa = ctrl.ControlSystemSimulation(diagnosis_ctrl)

dataset = [
    [28, 75, 6.5],
    [32, 85, 7.2],
    [26, 65, 5.8],
    [30, 80, 6.9],
    [27, 70, 6.2],
    [31, 90, 7.5],
    [29, 75, 6.7],
    [33, 80, 7],
    [25, 60, 6],
    [30, 85, 7.3],
    [25, 70, 6.5],
    [30, 60, 7.2],
    [28, 75, 6.8],
    [32, 80, 6.2],
    [26, 65, 7],
    [29, 85, 6.5],
    [27, 72, 6.9],
    [31, 78, 6.4],
    [24, 68, 7.1],
    [33, 76, 6.3]
]

for data in dataset:
    suhu_input = data[0]
    kelembaban_input = data[1]
    ph_input = data[2]

    diagnosa.input['suhu'] = suhu_input
    diagnosa.input['ph_tanah'] = ph_input
    diagnosa.input['kelembaban'] = kelembaban_input
    diagnosa.compute()

    hasil_diagnosis = diagnosa.output['diagnosis']
    print("Hasil Diagnosis:", hasil_diagnosis)

# Menampilkan kurva keanggotaan untuk variabel diagnosis
diagnosis.view(sim=diagnosa)

# Menunggu tampilan plot ditutup sebelum keluar
plt.show()