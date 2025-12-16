import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import gdown
import os

MODEL_PATH = "best_batiklens_model_final.h5"
MODEL_URL = "https://drive.google.com/uc?id=1Bke9qFhYmLDbD1C8a_IJQoSPl5q4razz"

IMG_WIDTH, IMG_HEIGHT = 224, 224

CLASS_NAMES = [
    'batik-bali', 'batik-betawi', 'batik-celup', 'batik-cendrawasih', 'batik-ceplok',
    'batik-ciamis', 'batik-garutan', 'batik-gentongan', 'batik-kawung', 'batik-keraton',
    'batik-lasem', 'batik-megamendung', 'batik-parang', 'batik-pekalongan',
    'batik-priangan', 'batik-sekar', 'batik-sidoluhur', 'batik-sidomukti',
    'batik-sogan', 'batik-tambal'
]

@st.cache_resource # Cache agar model hanya dimuat sekali
def load_batik_model():
    model = load_model(MODEL_PATH)
    return model

model = load_batik_model()

#data isi teks
FILOSOFI_BATIK = {
    'batik-bali': {
        'asal' : "Bali",
        'sejarah' : "Batik Bali adalah hasil dari penyebaran Batik dari Pulau Jawa yang dibawa ke Pulau Bali.",
        'motif' : "Motif batik Bali terinspirasi dari lingkungan alam dan budaya bali serta pengaruh dari luar daerah, yang divisualisasikan sebagai motif naturalis, dekoratif, dan abstrak. Batik Bali juga berwarna-warni, cerah, tidak monoton dan menggambarkan kearifan lokal khas Bali seperti Barong, bunga Kamboja dan burung jalak Bali.",
        'makna': "Motif-motif batik Bali mengandung makna nilai-nilai solidaritas dengan warna yang cerah dan unsur agama Hindu yang cukup kuat.",
        'sumber' : "Berita TEMPO yang bisa diakses melalui link berikut ini https://www.tempo.co/hiburan/mengenal-batik-khas-bali-warna-motif-dan-cara-pembuatannya-115981 dan wikipedia https://id.wikipedia.org/wiki/Batik_Bali"
    },
    'batik-betawi': {
        'asal' : "Jakarta, Jawa Barat",
        'sejarah' : "Batik Betawi mencerminkan identitas masyarakat Jakarta, sehingga menjadikannya salah satu simbol kebanggaan bagi warga Ibu Kota Jakarta.",
        'motif' : "Motif Batik Betawi menggambarkan kearifan lokal seperti Ondel-ondel, Monas, Sungai Ciliwung dan Nusa Kelapa.",
        'makna': "Motif batik Betawi menyimpan filosofi mendalam tentang kehidupan dan kebersamaan.",
        'sumber' : "Website Wonderful Indonesia yang bisa diakses melalui link berikut ini https://www.indonesia.travel/id/id/travel-ideas/culture/batik-betawi/"
    },
   'batik-celup': {
        'asal' : "Jawa",
        'sejarah' : "Teknik seni jumput/celup berasal dari Timur jauh sejak sekitar 3000 tahun sebelum Masehi dan mulai dikenal di berbagai belahan dunia karena perdagangan tekstil, termasuk di India, Cina, dan Jepang. Orang Romawi, misalnya, mulai mengenal teknik pewarnaan kain ini dari India dan Cina melalui jalur perdagangan kuno. Seiring waktu, teknik ini menyebar dan berkembang hingga masuk ke Nusantara sehingga dikenal sebagai bagian dari kerajinan tekstil di Indonesia.",
        'motif' :"Motif Batik Celup dibuat dengan cara batik yang dibuat dengan cara ikat celup, yaitu diikat dengan tali yang dicelupkan ke dalam warna.",
        'makna': "Motif yang terbentuk dari Batik Celup cukup beragam dan berbeda-beda, Batik Celup mengekspresikan kreativitas dan inovasi.",
        'sumber': "Berita Detik yang bisa diakses melalui link berikut ini https://www.detik.com/jabar/berita/d-6280041/mengenal-batik-jumputan-pengertian-motif-dan-cara-membuat/"
    },
    'batik-cendrawasih': {
        'asal' : "Papua",
        'sejarah' : "Perkembangan batik di Indonesia timur bermula ketika pemerintah memperoleh program bantuan dari salah satu organisasi PBB, UNDP di tahun 1985. Bantuan tersebut dipergunakan untuk pemberdayaan kebudayaan di Papua.",
        'motif' : "Motif yang tertuang dalam Batik Cendrawasih merupakan kearifan lokal khas Papua yaitu Burung Cendrawasih. Dimana Burung Cendrawasih adalah salah satu spesies burung langka, dilindungi oleh pemerintah Indonesia.",
        'makna': "Motif Batik Cendrawasih dianggap sebagai motif sakral dan mewakili identitas masyarakat Papua, baik di provinsi Papua maupun Papua Barat. Burung Cendrawasih juga dipercaya sebagai burung surga yang menghubungkan kehidupan di bumi dengan surga.",
        'sumber': "Artikel yang bisa diakses melalui link berikut ini https://www.iwarebatik.org/burung-cendrawasih-agr/?lang=id dan https://validnews.id/kultura/identitas-papua-dalam-motif-batik-cendrawasih"
    },
    'batik-ceplok': {
        'asal' : "Yogyakarta, Jawa Tengah",
        'sejarah' : "Batik Ceplok telah ada sejak zaman Kerajaan Mataram, dengan pusat produksi utama di Kotagede, Yogyakarta.",
        'motif' : "Motif Batik Ceplokan atau yang dikenal sebagai Ceplokan, berasal dari Bahasa Jawa, di mana kata ceplok memiliki arti sekuntum. Istilah ini digunakan untuk menggambarkan pola hiasan batik yang terdiri dari satuan-satuan terpisah.",
        'makna': "Motif Batik Ceplokan mencerminkan keteraturan dan keseimbangan yang dianggap penting dalam menjalani kehidupan. Pola hiasannya yang berulang, dalam jumlah yang melimpah, dapat diartikan sebagai simbol kumpulan. Dalam konteks filosofi Jawa, hal ini dikenal dengan istilah grampol yang merujuk pada kumpulan segala hal yang baik.",
        'sumber' : "Website Telusur Kultur yang bisa diakses melalui link berikut ini https://telusurkultur.com/blogs/news/mengenal-asal-usul-batik-ceplok/"
    },
    'batik-ciamis': {
        'asal' : "Ciamis, Jawa Barat",
        'sejarah' : "Batik Ciamis sudah ada sejak abad ke 20, pengaruh dari Kerajaan Galuh juga turut membentuk keunikan batik ini. Pada 1960-1980, batik khas Ciamis mengalami masa kejayaannya.",
        'motif' : "Motif Batik Ciamis menggambarkan kearifan lokal seperti Lepan Kembang, Ciungwanara, Onom, dan Lepan Kukubu.",
        'makna': "Motif batik Ciamis sederhana tapi penuh wibawa. Kesederhaan ini tak lepas dari sejarah keberadaannya yang banyak dipengaruhi daerah lain, sehingga menghasilkan motif yang sesuai dengan gaya dan selera masyarakat setempat yang bersahaja dan elegan.",
        'sumber' : "Website Visit Ciamis https://visitciamis.id/batik-ciamis-warisan-budaya-yang-otentik/ dan Perpustakaan Digital Budaya Indonesia https://budaya-indonesia.org/Batik-Ciamis/"
    },
    'batik-garutan': {
        'asal' : "Garut, Jawa Barat",
        'sejarah' : "Berasal dari warisan nenek moyang yang berlangsung secara turun-temurun dan telah berkembang cukup lama sebelum masa kemerdekaan Indonesia. Hingga tahun 1945, batik Garut semakin populer dengan sebutan batik tulis Garutan yang mengalami masa kejayaan antara tahun 1967 sampai 1985.",
        'motif' : "Motif-motif Batik Garut biasanya berbentuk geometrik sebagai ciri khas ragam hiasnya, dan juga ada yang bermotif flora dan fauna.",
        'makna': "Bentuk motif Batik Garut merupakan cerminan dari kehidupan sosial budaya, falsafah hidup, dan adat istiadat orang Sunda.",
        'sumber': "Artikel Kompas yang bisa diakses melalui link berikut ini https://www.kompas.com/skola/read/2023/02/21/140000169/mengenal-sejarah-batik-garutan-batik-tulis-warisan-leluhur"
    },
    'batik-gentongan': {
        'asal' : "Madura",
        'sejarah' : "Batik Gentongan tercipta karena budaya di wilayah pesisir. Di mana ketika para suami bekerja sebagai nelayan, para istri akan membatik sembari menunggu kepulangan suaminya selama berbulan-bulan. Pada saat itu batik tidak digunakan sebagai mata pencaharian masyarakat setempat, melainkan sebagai hadiah yang akan diberikan kepada suami untuk menyambut kepulangannya.",
        'motif' : "Corak dan motif Batik Gentongan umumnya menggambarkan kegiatan nelayan dan hewan-hewan yang dijumpai ketika pergi melaut, karena sebagian besar masyarakat bermata pencaharian sebagai nelayan.",
        'makna': "Dari segi warna, karakteristik warna yang digunakan cenderung berani dan tegas seperti warna merah (melambangkan karakter masyarakat Madura yang kuat dan keras), hijau (melambangkan beberapa kerajaan Islam yang didirikan dan berkembang di Madura), kuning (melambangkan bulir-bulir padi) dan biru (melambangkan laut yang mengelilingi sekitar Pulau Madura).",
        'sumber': "Artikel DetikJatim yang bisa diakses melalui link berikut ini https://www.detik.com/jatim/budaya/d-6960972/batik-gentongan-madura-asal-usul-filosofi-hingga-teknik-pembuatan"
    },
    'batik-kawung': {
        'asal' : "Yogyakarta, Jawa Tengah",
        'sejarah' : "Motif batik ini pertama kali dikenal pada abad ke 13 tepatnya di pulau Jawa. Pada awalnya motif ini muncul pada ukiran dinding di beberapa candi di Jawa seperti Prambanan.",
        'motif' : "Motif Batik Kawung merupakan motif batik yang bentuknya berupa bulatan mirip buah kawung (sejenis kelapa atau kadang juga dianggap sebagai aren atau kolang-kaling) yang ditata rapi secara geometris.",
        'makna': "Motif kawung bermakna kesempurnaan, kemurnian dan kesucian. Motif ini juga diyakini diciptakan oleh salah satu Sultan kerajaan Mataram.",
        'sumber': "Website Dinas Kebudayaan (Kundha Kabudayan) Daerah Istimewa Yogyakarta https://budaya.jogjaprov.go.id/berita/detail/1152-batik-kawung"
    },
    'batik-keraton': {
        'asal' : "Jawa Tengah",
        'sejarah' : "Berasal dari Kerajaan Mataram dimana membatik di lingkungan keraton awalnya merupakan kegiatan spiritual yang dilakukan oleh para putri keraton dan abdi dalem sebagai bentuk latihan kesabaran dan olah rasa.",
        'motif' : "Motif batik keraton ada yang berfungsi sebagai simbol status sosial, untuk upacara adat dan juga sebagai media untuk meditasi.",
        'makna': "Beberapa motif batik keraton berfungsi sebagai simbol status sosial untuk menunjukkan hierarki pemakainya, apakah ia seorang Raja, Pangeran, atau kerabat jauh. Lalu juga sebagai sarana ritual dimana batik tersebut Digunakan dalam upacara adat, mulai dari kelahiran (mitoni), pernikahan, hingga upacara kematian. Dan yang terakhir sebagai media meditasi karena proses pembuatan batik yang rumit dan repetitif (seperti membuat titik-titik cecek) dianggap sebagai bentuk meditasi untuk mendekatkan diri pada Sang Pencipta.",
        'sumber': "Website Mandalas https://wearemandalas.com/en-id/blogs/articles/motif-batik-keraton-simbolisme-fungsi-dan-nilai-budaya/"
    },
    'batik-lasem': {
        'asal' : "Rembang, Jawa Tengah",
        'sejarah' : "Kemunculan motif Batik Lasem berasal dari kedatangan bangsa Tiongkok ke wilayah Rembang, Jawa Tengah yang lalu disebut sebagai Tiongkok kecil. Pada abad ke-15, seorang pendatang dari Tiongkok bernama Na Li Ni atau Si Putri Campa memperkenalkan teknik membatik.",
        'motif': "Motif Batik Lasem didominasi oleh perpaduan dari motif hewan dengan motif tumbuhan khas Jawa.",
        'makna': "Makna motif Batik Lasem bermacam-macam tergantung gambar yang tertuang di dalam batik adalah motif flora atau motif fauna. Batik Lasem Naga bermakna berbagai harapan mulia dan simbolisasi perjalanan spiritualisme. Motif gunung ringgit merupakan manifestasi harapan supaya penggunanya selalu dilimpahi dengan kemakmuran.",
        'sumber': "Website Gramedia Blog https://www.gramedia.com/best-seller/batik-lasem-mengenal-sejarah-dan-makna-motifnya/."
    },
    'batik-megamendung': {
        'asal' : "Cirebon, Jawa Barat",
        'sejarah' : "Kemunculan motif batik megamendung dari kedatangan bangsa Tiongkok ke wilayah Keraton Cirebon dimana Pelabuhan Muara Jati merupakan tempat persinggahan para pendatang yang berasal dari dalam maupun luar negeri. Pada abad ke-16, terjadi pernikahan Sunan Gunung Jati dengan Ratu Ong Tien dari China sehingga akulturasi budaya.",
        'motif' : "Nama “Mega Mendung” berasal dari kata “Mega” yang berarti awan dan “Mendung” yang menggambarkan cuaca yang sejuk. ",
        'makna': "Filosofi dari motif ini adalah kesabaran dan ketenangan, seperti langit yang mendung menandakan hujan yang akan membawa kesejukan. Awan yang berlapis-lapis juga melambangkan ketenangan dan keuletan.",
        'sumber': "Website Gramedia Blog https://www.gramedia.com/literasi/motif-batik-megamendung/ dan Telkom Blog https://bcaf.telkomuniversity.ac.id/batik-mega-mendung-di-mata-dunia/"
    },
    'batik-parang': {
        'asal' : "Jawa",
        'sejarah' : "Pada masa Kesultanan Mataram pada abad ke-16, motif ini digunakan secara eksklusif oleh bangsawan dan keluarga kerajaan sebagai simbol status dan kekuasaan. Batik Parang juga tidak hanya dianggap sebagai pakaian, tetapi juga sebagai lambang sosial yang menunjukkan kedudukan seseorang dalam masyarakat.",
        'motif' : "Motif Batik Parang berkaitan dengan simbolisme pedang sebagai sumber inspirasi desainnya.",
        'makna': "Motif Batik Parang menggambarkan kekuatan, keberanian, dan semangat juang. Pola yang teratur dan berulang pada Batik Parang juga melambangkan keseimbangan dan harmoni dalam kehidupan.",
        'sumber': "Website Museum Sonobudoyo Yogyakarta https://sonobudoyo.jogjaprov.go.id/id/tulisan/read/sejarah-batik-parang"
    },
    'batik-pekalongan': {
        'asal' : "Pekalongan, Jawa  Tengah",
        'sejarah' : "Batik Pekalongan diperkirakan muncul tahun 1800-an dan mengalami perkembangan pesat setelah Perang Jawa atau Perang Diponegoro. Motif-motifnya pun terinspirasi dari kebudayaan yang dibawa oleh pedagang dari berbagai negara.",
        'motif' : "Motif batik Pekalongan umumnya mengambil inspirasi dari flora dan fauna. Warna-warna yang digunakan antara lain gradasi merah muda, merah tua, kuning terang, jingga, cokelat, biru muda, hijau muda, hijau tua, dan ungu.",
        'makna': "Motif-motif Batik Pekalongan mencerminkan keindahan flora dan fauna yang ada di sekitar daerah Pekalongan serta nilai-nilai budaya yang berkembang secara dinamis mengikuti zaman.",
        'sumber': "Website Indonesia Kaya https://indonesiakaya.com/pustaka-indonesia/batik-pekalongan/"
    },
    'batik-priangan': {
        'asal' : "Tasikmalaya, Jawa Barat",
        'sejarah' : "Pada masa Kerajaan Tarumanegara, salah satu kerajaan tertua di Nusantara, batik mulai dikenal oleh masyarakat setempat berkat melimpahnya populasi pohon tarum, yang menjadi bahan utama dalam proses pembuatan batik. Desa-desa seperti Mangunreja, Sukapura, Maronjaya, Wurug, dan Tasikmalaya Kota memiliki jejak sejarah yang kuat terkait dengan tradisi batik ini, karena wilayah-wilayah ini dulunya pernah menjadi pusat pemerintahan Tarumanegara.",
        'motif' : "Batik Priangan memiliki motif hias non-geometris yang menggabungkan unsur flora dan fauna dalam bentuk abstrak maupun realistik.",
        'makna': "Motif Batik Priangan yang umumnya menunjukkan bentuk tumbuhan dan hewan yang digambarkan secara abstrak maupun realistik, mencerminkan identitas budaya Priangan yang sederhana, terbuka, dan penuh estetik.",
        'sumber': "Website  Batik Prabuseno https://www.batikprabuseno.com/artikel/edukasi/batik-priangan-menggali-kekayaan-warisan-tasikmalaya/"
    },
    'batik-sekar': {
        'asal' : "Yogyakarta & Solo, Jawa Tengah",
        'sejarah' : "Saat kejayaan kerajaan Majapahit masih berkibar di tanah Jawa, Batik Sekar Jagad menjadi karya adiluhung leluhur pribumi yang membuka diri terhadap adanya akulturasi asing. Dalam perkembangannya, bukan hanya Belanda yang mempengaruhi namun etnis Tiongkok juga turut memberi pengaruh terhadap budaya yang ada.",
        'motif' : "Motif Batik Sekar Jagad berasal dari kata “kar jagad”. Kar diambil dari kata kaart bahasa Belanda yang artinya peta dan jagad dari Bahasa Jawa yang artinya dunia. Sehingga motif batik sekar jagad juga bisa melambangkan keanekaragaman dunia.",
        'makna': "Gambaran pengulangan motif geometris dengan nuansa bunga, dapat dimaknai sebagai keindahan dan keluhuran kehidupan dunia yang penuh dengan ragam perbedaan dan saling berdampingan.",
        'sumber': "Website Hamzah Batik https://hamzahbatik.co.id/mengenal-motif-batik-sekar-jagad-dan-makna-simbol-keindahannya/"
    },
    'batik-sidoluhur': {
        'asal' : "Yogyakarta & Solo, Jawa Tengah",
        'sejarah' : "Batik Sido Luhur diciptakan oleh Ki Ageng Henis untuk anak keturunannya. Ki Ageng Selo berharap agar pemakai batik Sido Luhur mempunyai hati bersih serta berfikir luhur sehingga kedepannya dapat berguna bagi masyarakat secara luas. Secara historis sendiri, batik Sido Luhur sudah ada sejak Kesultanan Mataram berdiri. Ki Ageng Henis sendiri merupakan kakek dari Panembahan Senopati serta merupakan cucu dari Ki Ageng Selo.",
        'motif' : "Motif Batik Sido Luhur terdiri dari beberapa ornamen utama yang disusun secara geometris dan simetris sehingga membentuk pola yang khas dan teratur. Ornamen-ornamen tersebut antara lain: adalah ornamen bangunan/ tahta, ornamen garuda/ lar, ornamen burung dan ornamen bunga.",
        'makna': "Secara umum batik ini bermakna keluhuran, dalam artian orang Jawa dalam menjalani kehidupannya akan selalu mewujudkan keluhuran baik secara materi maupun non materi. Ornamen bangunan/tahta menggambarkan bentuk singgasana yang merepresentasikan visual kedudukan seseorang yang terhormat. Ornamen garuda/lar berbentuk seperti sayap burung menggambarkan sifat ketabahan. Ornamen burung berupa merak atau kupu-kupu menciptakan kesan ruang udara. Ornamen bunga mengekspresikan keindahan alam.",
        'sumber': "Website  Batik Prabuseno https://www.batikprabuseno.com/artikel/edukasi/batik-sidoluhur/"
    },
    'batik-sidomukti': {
        'asal' : "Solo, Jawa Tengah",
        'sejarah' : " Batik Sidomukti merupakan pengembangan dari motif batik Sidomulyo yang pertama kali muncul pada masa Kesultanan Mataram. Motif ini memiliki makna harapan agar seseorang mendapatkan kebahagiaan dan kesejahteraan dalam hidupnya.",
        'motif' : "Nama “Sidomukti” sendiri berasal dari dua kata dalam bahasa Jawa, yaitu “sido” atau “sida” yang berarti terlaksana atau menjadi kenyataan, dan “mukti” yang berarti kebahagiaan dan kesejahteraan atau tidak kurang satu apapun. Oleh karena itu, motif batik Sidomukti juga menjadi simbol harapan untuk kehidupan yang lebih baik.",
        'makna': "Motif Batik Sidomukti ada motif bunga, motif kupu-kupu dan motif garuda. Motif bunga melambangkan kehidupan, kecantikan dan kesuburan dan simbol pertumbuhan dan kemakmuran dalam hidup. Motif kupu-kupu menjadi simbol metamorfosis dan perubahan yang indah. motif garuda sendiri menjadi simbol kegagahan dan kewibawaan yang meyiratkan keindahan dan kekuatan, dimana motif ini juga sering dipakai untuk menggambarkan kemuliaan dan kehormatan.",
        'sumber': "Website Hassa Batik https://hassa.co.id/motif-batik-sidomukti/"
    },
    'batik-sogan': {
        'asal' : "Yogyakarta dan Solo, Jawa Tengah",
        'sejarah' : "Batik Sogan berasal dari sejarah pewarnaan batik yang menggunakan pewarna alami dari batang kayu pohon soga. Karena menggunakan pewarna alami, batik Sogan memiliki ciri khas warna yang didominasi warna gelap seperti hitam dan coklat.",
        'motif' : "Batik Sogan memiliki lima warna penting yang merupakan simbol nafsu manusia, yaitu hitam, merah, kuning, putih dan coklat.",
        'makna': "Warna pada Batik Sogan memiliki arti yang berbeda-beda. Hitam adalah simbol nafsu dunia, merah adalah simbol nafsu amarah, kuning adalah simbol nafsu sufiyah, dan putih adalah simbol nafsu kebaikan. Sementara warna coklat atau kecoklatan adalah simbol pribadi yang hangat, rendah hati, bersahabat, kebersamaan, tenang. Penggambaran ini sesuai dengan kepribadian masyarakat Jawa yang mengutamakan rasa dalam setiap tindak tanduknya.",
        'sumber': "Artikel Kompas yang bisa diakses melalui link berikut ini https://yogyakarta.kompas.com/read/2024/02/04/211611278/mengenal-batik-sogan-dari-asal-nama-hingga-perbedaan-gaya-solo-dan/ "
    },
    'batik-tambal': {
        'asal' : "Yogyakarta, Jawa Tengah",
        'sejarah' : "Penyebutan batik Tambal mempunyai makna menambal atau memperbaiki hal-hal yang rusak. Kain batik bermotif tambal dipercaya bisa membantu kesembuhan orang sakit. Caranya dengan menyelimuti orang sakit tersebut dengan kain motif tambal. Kepercayaan ini muncul karena orang yang sakit dianggap ada sesuatu yang kurang baik, sehingga untuk mengobatinya perlu ditambal. Di keraton, motif batik tambah digunakan para abdi dalem yang berpangkat Panewu/Mantri dari golongan Juru Tulis.",
        'motif' : "Batik Tambal memiliki ciri khas motif bidang segitiga, dimana motifnya merupakan pengembangan dari motif-motif yang sudah ada sebelumnya. Motif ini adalah bagian dari pola geometri yang terdiri dari bentuk-bentuk segi empat, bujur sangkar, dan belah ketupat.",
        'makna': "Batik Tambal memiliki makna filosofi tentang cerita kehidupan dan kesatuan/keutuhan.",
        'sumber': "Artikel Kumparan yang bisa diakses melalui link berikut ini https://kumparan.com/sejarah-dan-sosial/filosofi-batik-tambal-yang-erat-kaitannya-dengan-kehidupan-manusia-231HGPgOCsb"
    },
}

# prediksi
def predict_batik(img_file):
    img = Image.open(img_file).convert('RGB')
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0 # Normalisasi

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_motif = CLASS_NAMES[predicted_class_index]
    confidence = predictions[0][predicted_class_index] * 100

    return predicted_motif, confidence

# ui streamlit
st.set_page_config(page_title="BatikLens App", layout="centered")
st.title("🌸 BatikLens 🌸")
st.markdown("Aplikasi berbasis AI yang dapat mendeteksi motif batik dan memberikan informasi terkait makna filosofis dan asal wilayah dari batik tersebut.")

uploaded_file = st.file_uploader("Pilih dan unggah citra batik Anda", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_display = Image.open(uploaded_file)
    st.image(image_display, caption='Citra Batik yang Diunggah', use_column_width=True)
    st.markdown("---")

    with st.spinner('Sedang menganalisis motif batik...'):
        predicted_motif, confidence = predict_batik(uploaded_file)

    st.success("✅ Analisis Selesai!")

    # show hasil
    st.markdown("### Hasil Klasifikasi BatikLens")
    st.write(f"Motif Batik Teridentifikasi: **{predicted_motif.replace('-', ' ').title()}**")
    st.write(f"Tingkat Keyakinan Model: **{confidence:.2f}%**")

    # show konten edukasi
    if predicted_motif in FILOSOFI_BATIK:
        data = FILOSOFI_BATIK[predicted_motif]
        st.markdown(f"### ✨Filosofi dan Makna {predicted_motif.replace('-', ' ').title()}")
        st.info(f"**Asal Daerah:** {data['asal']}")
        st.info(f"**Sejarah:** {data['sejarah']}")
        st.info(f"**Penjelasan Motif:** {data['motif']}")
        st.info(f"**Makna Filosofis:** {data['makna']}")
        st.info(f"**Sumber:** {data['sumber']}")
