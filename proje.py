#XGBoost kütüphanesi yüklenmelidir.
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import norm,skew

from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.base import clone

import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

"""
BURADAKİ TÜM KOLONLARIMIZI BÜTÜNLÜĞÜ SAĞLAMAK AMACIYLA VERİ SETİNDE BULUNDUĞU İSİMLE OLUŞTURDUM.
KAYNAK KODLARDAN DEĞİŞTİRİLEBİLİR.

"""

#Kolon isimlerimizi burada tanımlıyoruz
column_name=["MPG","Cylinders","Displacement","Horsepower","Weight","Acceleration","Model Year","Origin"]

#burada veri setimizi tanıtıp içerisindeki boş verileri ? olarak yorumları \t biçimi olarak gösteriyoruz.
data=pd.read_csv("auto-mpg.data",names=column_name,na_values="?",comment="\t",sep=" ",skipinitialspace=True);
          
#mil başına ne kadar yakacağını hedef olarak belirledik. Burayı hesaplayacağız.
data=data.rename(columns={"MPG":"target"})

#satır bilgisini alalım
print(data.shape)

#veri hakkında genel bilgiler ve kayıp veri bilgileri
print(data.info)

#verinin ortalaması, standart sapması bilgilerini gösterir
describe=data.describe()

# %% Kayıp değerler
print(data.isna().sum())

#burada horsepower boş verileri yerine horsepower ortalamaları alınarak veri seti bozulması engellendi
data["Horsepower"]=data["Horsepower"].fillna(data["Horsepower"].mean())

#kayıp değer var mı?
print(data.isna().sum())

#boş veri yoksa grafiği çizebiliriz.
sns.distplot(data.Horsepower)

# %% EDA > Veri analizi

# Verinin korelasyonu - yani ilgi dağılımı
corr_matris=data.corr()

# Harita oluştur
sns.clustermap(corr_matris,annot=True ,fmt=".2f")

# Harita title
plt.title("Korelasyon değerleri")

# Grafiği göster
plt.show()


"""
# YORUM : Burada yüksek değerlere sahip olan kısım yüksek korelasyon ilişkisine sahip demektir.
# Burada target ile ilişkisi yüksek olan kısımlar negatif olarak görünüyor fakat kendi aralarında pozitif
# ilişkiye sahipler. 
"""


# 0.75 ve üstü ilişkiye sahip olan kısımları filtreleyerek gösteriyoruz.
sinirlama=0.75

# Bu kısımda matris in target ile ilişkisi sinirlama dan büyük olanları filtre içine attık
filtre=np.abs(corr_matris["target"])>sinirlama

# Burada değerleri kolonlara uyguladık
corr_degerler=corr_matris.columns[filtre].tolist()

# Harita oluştur
sns.clustermap(data[corr_degerler].corr(),annot=True ,fmt=".2f")

# Harita title
plt.title("Korelasyon değerleri")

# Grafiği göster
plt.show()

# %% Eş düzlemlilik : Yani bir örneği gösteren birden çok kısım varsa bu modelin doğruluğunu olumsuz etkiler.

# Eşdüzlemliliği ve grafiğini görmek için
sns.pairplot(data, diag_kind="kde", markers="+")

# Burada grafiği çiziyoruz
plt.show()


"""
# YORUM : Burada silindir sayısı kategorik veri olarak değerlendirilebilir. 4-6-8 olarak gösterilen verilerde az hata ile toplanma söz konusu
# Hızlanma, hp ve ağırlık ile target yani yakıt tüketimi arasında ters bir orantı var
# Grafiklerine bakıldığında da benzerlikleri var birbirleriyle alakalı yapılardır
# Diğer verilerde belli bir dağılım yok ve Originde de belirleyici bir şey yok.

# Ağırlık-Hızlanma grafiğinde aykırı veri bulunmakta
# Grafik sağa kuyruklu ve aykırı veri olabilir

# Silindirler(Cylinders) ve Menşei(Origin) kategorik olabilir.

# Çaprazdan grafiklerin üst ve alt kısımları aynı
"""
# %% Silindir ile Menşei kısımlarını kategorik gösteriyoruz.
plt.figure()
sns.countplot(data["Cylinders"])
print(data["Cylinders"].value_counts())

plt.figure()
sns.countplot(data["Origin"])
print(data["Origin"].value_counts())

# Box : Burada aykırı verilerin olup olmadığına bakıyoruz
# Grafiklerde üstte ve altta kalan noktalar aykırı veriler olarak değerlendirilir.
for c in data.columns:
    plt.figure
    sns.boxplot(x=c, data=data, orient="v")
    

# İvmelenme ve Beygir gücünde aykırı veriler bulunmakta.


"""
Burada aykırı değerleri çıkartıp verilerin tam olarak uyumlu olmasını sağlamaya çalışacağız.

Bunun matematiksel karşılığı ;
Grafikteki üst aykırı değerler: sondeger(q3)-ilkdeger(q1)=sonuc ve sondeger + 1.5 x sonuc olarak yapabiliriz.
Grafikteki alt aykırı değerler: sondeger(q3)-ilkdeger(q1)=sonuc ve ilkdeger - 1.5 x sonuc olarak yapabiliriz.

"""

sinir_aykiri=2
horsepower_desc=describe["Horsepower"]

# Burada üst sınır 75% ve alt sınır 25% olarak verilir.

# q3 değeri alt sınır q1 değeri üst sınır değerler alınıp çıkarılır.
q3_hp=horsepower_desc[6]
q1_hp=horsepower_desc[4]

iqr_hp= q3_hp-q1_hp

# Daha sonra fark ile aykırı sınır çarpılıp q3 ile toplanır bu üst çizgidir
top_limit_hp=q3_hp+sinir_aykiri*iqr_hp

# Sonra fark ile aykırı sınır çarpılıp q1 ile çıkarılır bu alt çizgidir
bottom_limit_hp=q1_hp-sinir_aykiri*iqr_hp

# Aykırı değer filtresi uygulanır (Alt çizgi altına)
filter_hp_bottom=bottom_limit_hp<data["Horsepower"]

# Aykırı değer filtresi uygulanır (Üst çizgi üstüne)
filter_hp_top=data["Horsepower"]<top_limit_hp

# İki filtrenin mantıksal ve ' si alınır.
filter_hp =filter_hp_top & filter_hp_bottom

# Data ya filtre uygulanır.
data=data[filter_hp]

# Hızlanma için aynısı yapılır
acceleration_desc=describe["Acceleration"]

# Burada üst sınır 75% ve alt sınır 25% olarak verilir.

q3_acc=acceleration_desc[6]
q1_acc=acceleration_desc[4]

iqr_acc= q3_acc-q1_acc

top_limit_acc=q3_acc + sinir_aykiri * iqr_acc
bottom_limit_acc=q1_acc - sinir_aykiri * iqr_acc

filter_acc_bottom=bottom_limit_acc<data["Acceleration"]
filter_acc_top=data["Acceleration"]<top_limit_acc
filter_acc =filter_acc_top & filter_acc_bottom

data=data[filter_acc]

# 3 tane aykırı değer çıkarıldı.

# %% Şimdi ise çarpıklıkların analizini yapacağız (Bağımlı veriler)
# Grafiklerin pozitif veya negatif çarpık durumu

# Çarpıklık > 1 ise pozitif çarpık yani sola yatkın sağa kuyruklu bir yapısı var demektir.
# Çarpıklık < -1 ise negatif çarpık yani sağa yatkın sola kuyruklu bir yapısı var demektir.

"""
Peki bu çarpıklık neyi etkiler
Kuyrukları aykırı verileri gösterir yani kuyruk ne kadar uzunsa o kadar aykırı değer vardır demektir.
Kuyruğun fazla olması çarpıklık değerinin de ya çok yüksek yada çok düşük olması demektir.
Buna göre biz bunları normalleştirmek için grafiği gauss dağılımına uygun hale getirmeliyiz.
Bunu sağlayarak modelimizi daha iyi çalıştırırız

Log dönüşümünü kullanarak bu çarpıklığı azaltabiliriz.
"""

# target ile başlayalım

sns.distplot(data.target, fit=norm)
# normale göre pozitif çarpık

(mu,sigma)=norm.fit(data["target"])
print("mu: {}, sigma: {}".format(mu, sigma))

# qq plot çizelim
plt.figure()
stats.probplot(data["target"],plot=plt)
plt.show()

# y ekseni bizim verilerin dağılımı 
# x ekseni teorik verilerin dağılımı 
# kırmızı çizgiye oturması gereken verilerimiz tam oturmamış ve bunu halletmemiz lazım

data["target"]=np.log1p(data["target"])

plt.figure()
sns.distplot(data.target, fit=norm)

# yeni mü ve sigma değerleri
(mu,sigma)=norm.fit(data["target"])
print("mu: {}, sigma: {}".format(mu, sigma))

# yeni qq plot çizelim
plt.figure()
stats.probplot(data["target"],plot=plt)
plt.show()

# Burada uçlarda hala küçük problemler var.
# Küçük dememin sebebi ise burada y ekseni değerleri aralarındaki fark azalınca hata daha azdır.

# %% Bağımsız değerlerin çarpıklık analizini yapalım
carpik_bagimsiz_degerler=data.apply(lambda x:skew(x.dropna())).sort_values(ascending=False)

carpik_bagimsiz=pd.DataFrame(carpik_bagimsiz_degerler,columns=["carpiklik_bmz"])
# Burada çarpıklık sadece Horsepower yani beygir gücünde var oda göz ardı edilebilecek düzeyde onun için göz ardı edebiliriz.

# %% Burada Menşei kısımlarındaki değerlerinin hataları 1 2 3 olası yerine sadece 1 ise hata diğer türlü değil gibi olmalı ki hata türü sadece 1 tane olsun

# onun için one hot encoding uygulanır

# Burada silindir ve menşei yapılacak kategorik olduğu için

# Kategorik hale getirelim
data["Cylinders"] =data["Cylinders"].astype(str)  
data["Origin"] =data["Origin"].astype(str)  


# Kategorik değerleri One hot encoding yapalım ki 1 tür hata olsun artık
data=pd.get_dummies(data)

# %% Train test split ile en doğru makine öğrenmesi algoritması seçilmeye çalışılacaktır ve Öğrenme verileri ile test verileri alınacaktır.

# Burada veri setinin verilerini aldık ama target hariç
x = data.drop(["target"],axis=1)

# Burada ise sadece target kısmını aldık
y = data.target

# Burada verinin 90% kısmı test verisi olarak kullanılacak
test_size=0.9

X_train, X_test, Y_train, Y_test=train_test_split(x,y, test_size=test_size, random_state=42)

# %% Burada verileri standardize ediyoruz ki birbirleri arasındaki değerlerin farkları çok fazla olup sonucumuzu etkilemesin
"""
[#####]Standart - Robust
"""
# Standart scaler
scaler=RobustScaler() 

#RobustScaler
#sonra yapılacak.

X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test) # Burada sadece transform ettik çünkü zaten X_train fit yapılı

# Buradan sonra ortalama(mean)=0 standart sapma=1 oldu.

# %% Buradan sonra artık modelimizin EĞİTİM kısmı başlıyor....

# Doğrusal Regrasyon (Linear regression)
# Burada veri seti içerisinde bulunan verilerin bir grafiğin üzerinde doğrusal bir çizgiye oturtulması ve arada
# oluşan gerçek veriler ile yeni yani çizgi üzerine oturmuş veriler arasındaki hata payını minimize etmek.

"""
Regresyon bir bağımlı değişken ile diğer birkaç bağımsız değişken arasındaki ilişkiyi belirler.
Regresyon analizi, bağımsız değişkenlerin bazıları değiştiğinde bağımlı değişkenin nasıl değiştiğini
anlamaya yardımcı olmaktadır.
"""

lr=LinearRegression()
lr.fit(X_train,Y_train)

#Lineer katsayılar
print("Lineer katsayılar: ",lr.coef_)

# Burada y tahmini değeri üretip X_test ile test veri setini yükleyip sonuçları gözlemliyoruz.
y_tahmini_deger=lr.predict(X_test)

# Ortalama Mutlak Hata(Mean Square Error)
mse=mean_squared_error(Y_test, y_tahmini_deger)

print("Doğrusal regrasyon ortalama mutlak hatası : ",mse)

# Burada 0.02 hata payıyla tüm verileri incelememiz mümkün oluyor. İşlemlerimize devam edelim.

# Hata : 0.020

# %% Sırt(Ridge) regrasyonu

"""
Bu ridge regrasyon u lineer regrasyon gibi bir çizgiye oturtma diyebiliriz fakat burada şöyle bir durum vardır
Linear regrasyonda dışta kalan veriler (test verileri) olabiliyor ve ortalama mutlak hata çok fazla çıkıyor
ve tam olarak oturmayan veriler olabiliyorken bu yöntem o durumu engelleyebilmektedir.
"""

# Burada ridge regrasyon için random değerini ve en fazla iterasyon sayısını belirledik
ridge=Ridge(random_state=42, max_iter=10000)

# Alfa değeri için -4 ve -0.5 belirleyip iterasyon 30 belirledik
alphas=np.logspace(-4,-0.5,30)

# Sınıflandırma algoritması sözlük tipini desteklediği için sözlüğe dönüşüm yaptık
ayarli_parametre=[{'alpha':alphas}]

# Katman değerini verdik 
n_folds=5

# Sınıflandırma için ridge, parametre ve skor tipini belirtip refit=True ile başka yerlerde kullanılmasını sağladık
clf=GridSearchCV(ridge, ayarli_parametre, cv=n_folds,scoring="neg_mean_squared_error", refit=True)

# Burada sınıflandırmak için örnek veri setlerimizi verdik ve tabloya sığdırdık
clf.fit(X_train, Y_train)

# Burada Test verilerinin skorlarının ortalamasını bulduk
scores=clf.cv_results_["mean_test_score"]

# Burada Test verilerinin skorlarının standart sapmasını bulduk
scores_std=clf.cv_results_["std_test_score"]

# Burada Sırt(ridge) regrasyonu katsayısını yazdırdık
print("Sırt(Ridge) regrasyonu katsayısı : ",clf.best_estimator_.coef_)

# En iyi tahmini bulduk
ridge=clf.best_estimator_

# En iyi tahmini yazdırdık 
print("Sırt(Ridge) regrasyonu en iyi tahmini : ",ridge)

# Tahmini değer hesaplaması
y_tahmini_deger= clf.predict(X_test)

# Ortalama karesel hatayı bulduk
mse=mean_squared_error(Y_test, y_tahmini_deger)

# Ortalama karesel hatayı yazdırdık
print("Sırt(Ridge) Ortalama karesel hata : ",mse)

print("-----------------------------------------------------------")

# Burada ise plot ile grafiği çizdirdik
plt.figure()

plt.semilogx(alphas, scores)

plt.xlabel("alpha")

plt.xlabel("score")

plt.title("Ridge")

# Hata : 0.019

# %%Lasso regrasyonu uygulayarak burada daha iyi bir sonuç ve daha az bir hata oranı elde etmeye çalıştık

# Burada lasso Sırt(ridge) regrasyonu ile aynıdır tek farkı b1 kısmının karesi değil mutlak degeri alınarak
# işlem görür.

# Bir diğer farkı katsayı olarak değeri direk 0 olarak alınabilir.

# Burada ridge regrasyon için random değerini ve en fazla iterasyon sayısını belirledik
lasso=Lasso(random_state=42, max_iter=10000)

# Alfa değeri için -4 ve -0.5 belirleyip iterasyon 30 belirledik
alphas=np.logspace(-4,-0.5,30)

# Sınıflandırma algoritması sözlük tipini desteklediği için sözlüğe dönüşüm yaptık
ayarli_parametre=[{'alpha':alphas}]

# Katman değerini verdik 
n_folds=5

# Sınıflandırma için ridge, parametre ve skor tipini belirtip refit=True ile başka yerlerde kullanılmasını sağladık
clf=GridSearchCV(lasso, ayarli_parametre, cv=n_folds,scoring="neg_mean_squared_error", refit=True)

# Burada sınıflandırmak için örnek veri setlerimizi verdik ve tabloya sığdırdık
clf.fit(X_train, Y_train)

# Burada Test verilerinin skorlarının ortalamasını bulduk
scores=clf.cv_results_["mean_test_score"]

# Burada Test verilerinin skorlarının standart sapmasını bulduk
scores_std=clf.cv_results_["std_test_score"]

# Burada Sırt(ridge) regrasyonu katsayısını yazdırdık
print("Lasso regrasyonu katsayısı : ",clf.best_estimator_.coef_)

# En iyi tahmini bulduk
lasso=clf.best_estimator_

# En iyi tahmini yazdırdık 
print("Lasso regrasyonu en iyi tahmini : ",lasso)

# Tahmini değer hesaplaması
y_tahmini_deger= clf.predict(X_test)

# Ortalama karesel hatayı bulduk
mse=mean_squared_error(Y_test, y_tahmini_deger)

# Ortalama karesel hatayı yazdırdık
print("Lasso Ortalama karesel hata : ",mse)

print("-----------------------------------------------------------")

# Burada ise plot ile grafiği çizdirdik
plt.figure()

plt.semilogx(alphas, scores)

plt.xlabel("alpha")

plt.xlabel("score")

plt.title("Lasso")

# Hata : 0.017

# %%Elastic Net ile regrasyon

"""
Amaç ridge ve lasso regresyon ile aynıdır ama elastic net, ridge ve lasso regresyonu birleştirir. 
Ridge regresyon tarzı cezalandırma ve lasso regresyon tarzında değişken seçimi yapar.

Diğerlerinden ayrılan özelliği ise yüksek bağlantılı verilerin çözümlenmesinde çok işe yarayan yöntemdir.

Not : Lambda değerleri farklıdır. Formülde hem sırt(ridge) hemde lasso da olan birinde karesel olarak alınan b1
diğerinde mutlak olarak alınan b1 hesaplanır ve lambda değerleri ile çarpılır ve toplanırlar.
"""
# Parametre olarak tabloda olacak 2 değerden birisi olan alfa ve l1 ratio bildiren bölüm denklemi

paramatre_tablosu={"alpha":alphas,
                   "l1_ratio":np.arange(0.0,1.0,0.05)}

eNet=ElasticNet(random_state=42,max_iter=10000)

clf=GridSearchCV(eNet, paramatre_tablosu, cv=n_folds,scoring='neg_mean_squared_error',refit=True)

clf.fit(X_train, Y_train)

print("ElasticNet katsayısı : ",clf.best_estimator_.coef_)

eNet=clf.best_estimator_

print("ElasticNet en iyi tahmin : ",eNet)

y_tahmini_deger=clf.predict(X_test)

mse=mean_squared_error(Y_test, y_tahmini_deger)

print("ElasticNet ortalama karesel değeri : ",mse)


# %% XGBoost ile analiz

# Büyük karmaşık veri setleri için kullanılan bir veri analizi yöntemidir.
# Bu uygulama için biraz fazla karmaşık olsada örnek olarak gösterilecektir.

"""

Algortimanın en önemli özellikleri yüksek tahmin gücü elde edebilmesi, aşırı öğrenmenin önüne geçebilmesi, 
boş verileri yönetebilmesi ve bunları hızlı yapabilmesidir.

Daha az kaynak kullanarak üstün sonuçlar elde etmek için yazılım ve donanım optimizasyon tekniklerini
uygulanmıştır. Karar ağacı tabanlı algoritmaların en iyisi olarak gösterilir

"""

model_xgb=xgb.XGBRegressor(objective='reg:linear',max_depth=5,min_child_weight=4,subsample=0.7,n_estimators=1000,learning_rate=0.07)

model_xgb.fit(X_train,Y_train)

y_tahmini_deger=model_xgb.predict(X_test)

mse=mean_squared_error(Y_test, y_tahmini_deger)

print("---------------")

print("XGB regrasyon ağacı ortalama karesel hatası : ",mse)

print("---------------")

# Hata oranı 0.01932020789212713 (Çok yüksek)

# Olmadı sonucun hata oranı yüksek bunun için parametreler vererek tekrar deneyelim.

parametre_tablosu={
    'nthread':[4],
    'objective':['reg:linear'],
    'learning_rate':[.03,0.05,.07],
    'max_depth':[5,6,7],
    'min_child_weight':[4],
    'silent':[1],
    'subsample':[0.7],
    'colsample_bytree':[0.7],
    'n_estimators':[500,1000]
    }

model_xgb=xgb.XGBRegressor()

clf=GridSearchCV(model_xgb, parametre_tablosu,cv=n_folds,scoring='neg_mean_squared_error',refit=True,n_jobs=5,verbose=True)

clf.fit(X_train, Y_train)

y_tahmini_deger=clf.predict(X_test)

mse=mean_squared_error(Y_test,y_tahmini_deger)

print("---------------")

print("XGB regrasyon ağacı ortalama karesel hatası : ",mse)

print("---------------")

# Hata oranı 0.017444718427058307 (Daha iyi)

# %% En iyi modellerin ortalaması 

"""
Kullandığımız modellerden bazılarının ortalamasını alarak en iyi sonucu elde etmeye çalışacağız

Burada en iyileri XGBoost ve Lasso değerleri en iyi değerlerdir.

"""

class ModelORtalama():
    def __init__(self,models):
        self.models=models
        
    def fit(self,X,y):
        self.models_=[clone(x) for x in self.models]
        
        for model in self.models_:
            model.fit(X,y)
            
        return self
    
    def predict(self,X):
        predictions=np.column_stack([model.predict(X) for model in self.models_])
        return np.mean(predictions,axis=1)


model_ort=ModelORtalama(models=(model_xgb,lasso))
model_ort.fit(X_train, Y_train)

y_tahmini_deger=model_ort.predict(X_test)

mse=mean_squared_error(Y_test, y_tahmini_deger)

print("---------------------")

print("En iyi modellerin ortalaması ile ortalama karesel hata : ",mse)

print("---------------------")

# Hata 0.017415759476209068 (En az hata oranlarından)


# %% SONUÇLAR


"""

-> [StandartScaler] ile elde edilen sonuçlar

Doğrusal regrasyon sonucuna göre hata oranı : 0.020632204780133088
Sırt(ridge) regrasyon sonucuna göre hata oranı : 0.01972533801080122
Lasso regrasyon sonucuna göre hata oranı : 0.017521594770822504
ElasticNet regrasyon sonucuna göre hata oranı : 0.01749609249317252

-> [RobustScaler] ile elde edilen sonuçlar (Aykırı değerleri veriden uzaklaştırır)

Doğrusal regrasyon sonucuna göre hata oranı : 0.02098471106586962
Sırt(ridge) regrasyon sonucuna göre hata oranı : 0.018839299330570537
Lasso regrasyon sonucuna göre hata oranı : 0.016597127172690823
ElasticNet regrasyon sonucuna göre hata oranı : 0.017234676963922283
XGBoost parametresiz regrasyon sonucuna göre hata oranı : 0.01932020789212713
XGBoost parametreli regrasyon sonucuna göre hata oranı : 0.017444718427058307
En iyi modellerin ortalaması sonucuna göre hata oranı : 0.017415759476209068


### Yorum

Burada ;

> StandartScaler kullanıldığında yani aykırı veriler normal verilerden uzaktaştırılmadan veri seti 
değerlendirildiğinde sonuçlar yukarıda ki gibidir ve en iyi başarımı ElasticNet ile elde ediyoruz.

> RobustScaler kullanıldığında ise aykırı veriler veri setinden uzaklaştırılarak değerlendirme yapılır.
Bununla birlikte Doğrusal regrasyonun başarımı düşerken diğerlerinde az bir iyileşme de olsa en iyi
başarımı Lasso regrasyonu sağlamaktadır. Lasso regrasyonunun başarımı ciddi oranda artmıştır ve diğer
tüm sonuçlardan daha iyi bir başarıma sahip olmuştur. 


BURADAKİ TÜM KOLONLARIMIZI BÜTÜNLÜĞÜ SAĞLAMAK AMACIYLA VERİ SETİNDE BULUNDUĞU İSİMLE OLUŞTURDUM.
KAYNAK KODLARDAN DEĞİŞTİRİLEBİLİR.

"""






















