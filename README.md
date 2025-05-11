
# Proyek Akhir: Menyelesaikan Permasalahan Jaya Jaya Institut

## Business Understanding

Jaya Jaya Institut menghadapi tantangan dalam menurunkan tingkat dropout mahasiswa yang cukup tinggi. Banyak mahasiswa tidak menyelesaikan studinya, yang berimplikasi pada kerugian reputasi dan efisiensi institusi. Untuk itu, diperlukan pendekatan berbasis data yang mampu mengenali mahasiswa yang berisiko dropout agar intervensi bisa dilakukan lebih awal.

### Permasalahan Bisnis

-   Tingginya angka mahasiswa yang dropout sebelum menyelesaikan studi.
    
-   Keterbatasan dalam mengidentifikasi faktor-faktor penyebab dropout secara sistematis.
    
-   Belum adanya sistem prediksi yang terintegrasi untuk mendeteksi mahasiswa berisiko tinggi.
    

### Cakupan Proyek

-   Eksplorasi dan analisis data mahasiswa untuk memahami pola dropout.
    
-   Pengembangan model machine learning berbasis Random Forest untuk memprediksi status mahasiswa (Dropout/Graduate).
    
-   Implementasi aplikasi prediksi menggunakan Streamlit.
    
-   Penyusunan dashboard visual interaktif berbasis Metabase untuk mendukung pemantauan performa mahasiswa.
    

### Persiapan

Sumber data: [Dataset mahasiswa dari Dicoding GitHub](https://github.com/dicodingacademy/dicoding_dataset/tree/main/students_performance)

Setup environment:

```
pip install -r requirements.txt
```

## Business Dashboard

Dashboard dikembangkan dengan Metabase dan berisi visualisasi yang menggambarkan:

-   Distribusi dropout berdasarkan gender, usia, status pernikahan, jalur masuk, dan program studi.
    
-   Pola performa akademik semester 1 dan 2 (jumlah mata kuliah lulus dan nilai rata-rata).
    
-   Hubungan antara kondisi ekonomi (status pembayaran kuliah, beasiswa, status utang) dengan risiko dropout.
    

> Link dashboard: (akan ditambahkan setelah deployment selesai)

## Menjalankan Sistem Machine Learning

Sistem prediktif dibangun dengan Streamlit dan didukung oleh model Random Forest. Aplikasi `app.py` mencakup:

-   **Form input** yang mencakup: data demografis (umur, gender, displaced), latar belakang pendidikan (nilai kualifikasi dan nilai admission), performa akademik (nilai dan evaluasi semester 1 & 2), serta kondisi ekonomi (status pembayaran dan utang).
    
-   **Fungsi encoding** yang menggunakan `StandardScaler` dan `OneHotEncoder` untuk preprocessing data.
    
-   **Fungsi predict** yang hanya menggunakan 10 fitur penting berdasarkan hasil feature importance dari `notebook.ipynb`.
    
-   **Pemanggilan model**  `joblib_model.pkl` untuk prediksi status mahasiswa.
    
-   **Tampilan hasil prediksi** secara langsung di aplikasi (Graduate atau Dropout).
    

Cara menjalankan:

```
streamlit run app.py
```

> Link aplikasi online: (akan ditambahkan setelah deploy ke Streamlit Community Cloud)

## Conclusion

Model Random Forest yang dibangun dari notebook berhasil mencapai akurasi sebesar **84.07%** dan recall **90.51%** dalam memprediksi status mahasiswa. Fitur-fitur dengan korelasi tertinggi terhadap dropout mencakup:

-   `Curricular_units_2nd_sem_grade`
    
-   `Curricular_units_2nd_sem_approved`
    
-   `Curricular_units_1st_sem_grade`
    
-   `Tuition_fees_up_to_date`
    
-   `Age_at_enrollment`
    

Aplikasi Streamlit yang dibangun memberikan kemudahan bagi pihak kampus untuk menguji status risiko dropout berdasarkan data input mahasiswa baru. Sementara dashboard Metabase mendukung analisis dan pemantauan tren dropout secara keseluruhan.

### Rekomendasi Action Items

-   Prioritaskan pendampingan akademik bagi mahasiswa yang memiliki nilai akademik rendah sejak semester pertama.
    
-   Evaluasi sistem pembayaran untuk memberikan fleksibilitas kepada mahasiswa dengan keterbatasan ekonomi.
    
-   Integrasikan model prediksi ke dalam sistem informasi akademik kampus.
    
-   Lakukan sosialisasi penggunaan sistem kepada staf akademik dan pembimbing agar intervensi lebih proaktif.
    
-   Lakukan retraining model secara berkala untuk menyesuaikan dinamika dan tren baru pada data mahasiswa.
