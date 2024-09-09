# -*- coding: utf-8 -*-
"""
Imamo dataset koji sadrži informacije o određenom broju osoba koje mogu biti relevantne u predikciji da li je ta osoba sklona tome da ima srčani udar ili ne. Moj zadatak je da na osnovu datih podataka naučim, istreniram i  predvidim da li je osoba sklona tome da ima srčani udar ili ne.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
import sklearn.metrics as metrics

data = pd.read_csv("data.csv")
# prikaz prvih 10 redova
data.head(10)

"""Eksploracija podataka i priprema

"""

# Lista varijabli za Label Encoding
label_encode_vars = ['ever_married', 'Residence_type']

# Funkcije za Label Encoding
def label_encode(df, columns):
    for col in columns:
        df[col] = df[col].map({'Yes': 1, 'No': 0}) if col == 'ever_married' else df[col].map({'Urban': 1, 'Rural': 0})
    return df

data = label_encode(data, label_encode_vars)

# Lista varijabli za Ordinal Encoding
ordinal_encode_vars = ['work_type']
ordinal_map = {
    'Private': 4,
    'Self-employed': 3,
    'Govt_job': 2,
    'children': 1,
    'Never_worked': 0
}

# Funkcija za Ordinal Encoding
def ordinal_encode(df, columns, mapping):
    for col in columns:
        df[col] = df[col].map(mapping)
    return df

data = ordinal_encode(data, ordinal_encode_vars, ordinal_map)

# Lista varijabli za One-Hot Encoding
one_hot_encode_vars = ['smoking_status', 'gender']

# One-Hot Encoding
data = pd.get_dummies(data, columns=one_hot_encode_vars)

# Brisanje NaN vrijednosti, inače će se pogrešno trenirati i dati veoma loše preciznosti i rezultate evaluacije
data = data.dropna(subset=['bmi'])
print(data.head())

# Funkcija za mješanje čitavog dataseta
def shuffle_df(df):
    return df.sample(frac=1, random_state=0).reset_index(drop=True)

data_shuffled = shuffle_df(data)

from sklearn.model_selection import train_test_split
# Podjela na train i test skup
X = data_shuffled.drop('stroke', axis=1)
y = data_shuffled['stroke']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler


# Sačuvajmo originalne DataFrame-ove za analize
df_train_original = pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
df_test_original = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)

# Normaliziranje podataka
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Učenje i transformacija obučavajućih podataka
X_test_scaled = scaler.transform(X_test)        # Transformacija testnih podataka na osnovu učenja iz X_train

print("X_train_scaled shape:", X_train_scaled.shape)
print("X_test_scaled shape:", X_test_scaled.shape)

# Matrica korelacije koja nam daje informacije o tome koje su kolone povezane, bitne i imaju uticaj na predviđanje
# Vidimo da je to age i ever_married jer starost je čest uzrok bolesti
corr_matrix = df_train_original.corr()

plt.figure(figsize=(16, 10))
sns.heatmap(corr_matrix, annot=True, cmap="YlGnBu")
plt.show()

# ISPRAVNA VERZIJA

# Pošto su podaci nebalansirani za klasu 0, uradićemo oversamplovanje da bi dobili bolje rezultate treniranja i testiranja.
# Postoji šansa, ukoliko se primjeni oversampling, da dođe do overfittovanja podataka. Oversampla se samo trening skup

from imblearn.over_sampling import SMOTE

# Primjena SMOTE oversamplinga na trening skup
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)


# Pretvaranje y_train_resampled u pandas seriju
y_train_resampled_series = pd.Series(y_train_resampled)

# Prikaz rezultata
print("Broj pojavljivanja klasa nakon oversamplinga:")
print(y_train_resampled_series.value_counts())

# Brojanje pojavljivanja vrijednosti u određenoj koloni
count_values = data['stroke'].value_counts()

# Ispis broja pojavljivanja svake vrijednosti
print(count_values)
#Zbog ovoga smo radili overfitting jer podaci su nebalansirani i super će se istrenirati i evaluirati za vrijednost izlazne varijable kada je 0 ali problem nastaje kada je vrijednost 1
# Tako smo dobili vještačke podatke i poboljšali naš model

# Ukoliko se svm model obuči sa SMOTE i ne primjeni parametar C  ili zadaju neke vrijednosti, tada se dešava sljedeće :
# Precision je za klasu 0 0.99 a za klasu 1 0.12 dok recall je približno (0.72 i 0.80 respektivno), f1-score (0.84 i 0.21)
# Međutim ako se primjeni regularizacijski faktor C = 10e-6, tada se podaci mjenjaju.
# Podaci se lošije predviđaju za klasu 1 nego za klasu 0 tj razlika za recall između klasa je veća kao i za f1-score dok je precision ostao prilično isti
# Ovo je posljedica nebalansiranog dataseta


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Inicijalizacija SVM modela s regularizacijskim parametrom
svm_weighted = SVC(kernel='linear', C=1e-6, class_weight='balanced')  # Postavite class_weight na 'balanced'

# Treniranje modela na oversampled podacima
svm_weighted.fit(X_train_resampled, y_train_resampled)

# Predikcija na testnom skupu s originalnim podacima
y_pred_test = svm_weighted.predict(X_test_scaled)

# Evaluacija modela na testnom skupu
accuracy_test = accuracy_score(y_test, y_pred_test)
print("Accuracy sa originalnim test podacima:", accuracy_test)

# Ispis detaljnijeg izvještaja o klasifikaciji sa originalnim test podacima
print(classification_report(y_test, y_pred_test))

"""Grafici za SVM pri čemu je primjenjen SMOTE

"""

# Matrica konfuzije je osnovni alat za evaluaciju performansi modela klasifikacije. Ona prikazuje stvarne i predviđene klase za skup podataka.
# Matrica konfuzije omogućava procjenu gdje model griješi i može biti korisna za razumijevanje performansi modela.
# Na primjer, možete vidjeti koliko često model zbunjuje jednu klasu s drugom, ili koliko često neuspješno prepoznaje određenu klasu.
# Model je dobro istreniran jer 842 pravih(tačnih) je prepoznao kao prave, kao i 5 lažnih je ispravno klasificirao kao lažne
# Model nema mnogo grešaka zato su vrijednosti u žutim poljima niske
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Plot the Confusion Matrix as a HeatMap
class_names=["prava", "lazna"] # Name  of classes
fig, ax = plt.subplots()

# Generiranje matrice konfuzije
conf_matrix_resampled = confusion_matrix(y_test,y_pred_test)

# create heatmap
sns.heatmap(pd.DataFrame(conf_matrix_resampled), annot=True, cmap="YlGnBu" ,fmt='g')
plt.tight_layout()
ax.invert_yaxis()
ax.set_xticklabels(class_names)
ax.set_yticklabels(class_names)

plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

plt.show()

# ROC (Receiver Operating Characteristic) krivulja je graf koji prikazuje performanse binarnog klasifikatora na svim pragovima odluke.
# ROC kriva prikazuje odnos između stope lažno pozitivnih rezultata (False Positive Rate - FPR) i stope istinito pozitivnih rezultata (True Positive Rate - TPR) za različite pragove odluke.
# Bitna je površina ispod te krive i ona treba da teži ka 1 što bi značilo da naš model tada bi pravio ili male ili nikakve greške
# U našem slučaju, površina je 0.83 što je zadovoljavajuće jer će samo u 20% slučajeva pogriješiti, kriva stabilno se približava 1

#Threshold je prag koji model koristi za klasifikaciju. Na primjer, ako je prag 0.5, sve instance sa vjerovatnoćom
# većom od 0.5 biće klasifikovane kao pozitivne, dok će one sa vjerovatnoćom manjom od 0.5 biti klasifikovane kao negativne.
# ROC kriva prikazuje performanse modela za sve moguće pragove, od 0 do 1, i kako se mijenja odnos između TPR i FPR pri promjeni praga.

from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Pretpostavljamo da je svm_weighted model obučen na oversampled trening podacima

# Izračunajte vjerojatnosti pripadnosti pozitivnoj klasi za neoversamplovane test podatke
y_pred_proba = svm_weighted.decision_function(X_test_scaled)

# Izračunajte FPR i TPR
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

# Izračunajte AUC
auc = roc_auc_score(y_test, y_pred_proba)

# Crtanje ROC krivulje
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label="SVM, AUC=" + str(round(auc, 2)))
plt.legend(loc="lower right")
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)
plt.show()

# Ispis AUC-a
print('AUC Score:', round(auc, 2))
print('The AUC Score provides an aggregate measure of performance across all possible classification thresholds.')
print('AUC ranges in value from 0 to 1. A model whose predictions are 100% wrong has an AUC of 0.0; one whose predictions are 100% correct has an AUC of 1.0.')

# Površina ispod krive je niska jer je dataset nebalansiran
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

# Izračunaj vjerojatnosti pripadanja klasama
y_scores = svm_weighted.decision_function(X_test_scaled)

# Izračunaj preciznost i odziv
precision, recall, _ = precision_recall_curve(y_test, y_scores)

# Nacrtaj Precision-Recall krivulju
plt.figure()
plt.step(recall, precision, color='b', alpha=0.2, where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall Curve')
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Izdvojite potporne vektore
support_vectors = svm_weighted.support_vectors_

# Prikaz podataka i potpornih vektora u grafikonu
plt.scatter(X_train_resampled[:, 0], X_train_resampled[:, 1], c=y_train_resampled, cmap='winter', label='Podaci')
plt.scatter(support_vectors[:, 0], support_vectors[:, 1], color='red', marker='x', label='Potporni vektori')
plt.xlabel('Prva značajka')
plt.ylabel('Druga značajka')
plt.title('Podaci i potporni vektori')
plt.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Izdvojite potporne vektore
support_vectors = svm_weighted.support_vectors_

# Prikaz podataka i potpornih vektora u 3D grafikonu
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Prikaz podataka
ax.scatter(X_train_resampled[:, 0], X_train_resampled[:, 1], X_train_resampled[:, 2], c=y_train_resampled, cmap='winter', label='Podaci')

# Prikaz potpornih vektora
ax.scatter(support_vectors[:, 0], support_vectors[:, 1], support_vectors[:, 2], color='red', marker='x', label='Potporni vektori')

ax.set_xlabel('Prva značajka')
ax.set_ylabel('Druga značajka')
ax.set_zlabel('Treća značajka')
ax.set_title('Podaci i potporni vektori u 3D prostoru')
ax.legend()

plt.show()

# Dobivanje apsolutnih vrijednosti težina atributa
# Ovdje vidimo da najveću težinu nosi atribut age što smo vidjeli sa linearne korelacije i kao sa selection feature
absolute_weights = np.abs(svm_weighted.coef_)

# Prikaz težina atributa u bar grafikonu
plt.bar(range(len(absolute_weights[0])), absolute_weights[0])
plt.xlabel('Atribut')
plt.ylabel('Težina')
plt.title('Težine atributa')
plt.show()

"""SVM oversampled preko SVMSMOTE

"""

from imblearn.over_sampling import SVMSMOTE

# Primjena SVMSMOTE oversamplinga na trening skup
svmsmote = SVMSMOTE(random_state=42)
X_train_resampled, y_train_resampled = svmsmote.fit_resample(X_train_scaled, y_train)

# Ukoliko se svm model obuči sa SVMSMOTE i ne primjeni parametar C:
# Omjer precision za 0 i 1 : (0.98, 0.16), recall je blizu, f1-score (0.90 i 0.26)
# Međutim ako se primjeni regularizacijski faktor C = 10e-6, tada se podaci previše reguliraju pa za klasu 1 svi podacu budu 0, dok je za klasu 0 recall 100%.
# Za C = 10e-5, neprimjetna promjena za precision,  recall je sada jako daleko (0.95 i 0.24)
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Inicijalizacija SVM modela s težinama klasa
svm_weighted = SVC(kernel='linear',C=10e-5)

# Treniranje modela na oversampled podacima
svm_weighted.fit(X_train_resampled, y_train_resampled)  # Koristimo podatke nakon oversamplinga

# Predikcija na testnom skupu s oversampled podacima
y_pred_test= svm_weighted.predict(X_test_scaled)

# Evaluacija modela na testnom skupu s oversampled podacima
accuracy_test = accuracy_score(y_test, y_pred_test)
print("Accuracy sa oversamplovanim test podacima:", accuracy_test)

# Ispis detaljnijeg izvještaja o klasifikaciji s oversampled test podacima
# Ispis izvještaja o klasifikaciji s zero_division parametrom
print(classification_report(y_test, y_pred_test, zero_division=0))

# Model nije ni overfittovan ni underfittovan. Promjenom hiperparametara, predviđanje se može poboljšati. Takođe može se primjeniti i undersampling ali onda može doći do gubitka važnih informacija.
# Možemo primjeniti umjesto oversamplinga i undersamplinga, svm sa težinama na način da parametar class_weight postavimo na "balanced"
# Automatski se računaju težine kako bi bile obrnuto proporcionalne broju uzoraka u svakoj klasi. Na taj način, SVM će uzeti u obzir neuravnoteženost klasa prilikom učenja.

# Matrica konfuzije je osnovni alat za evaluaciju performansi modela klasifikacije. Ona prikazuje stvarne i predviđene klase za skup podataka.
# Matrica konfuzije omogućava procjenu gdje model griješi i može biti korisna za razumijevanje performansi modela.
# Na primjer, možete vidjeti koliko često model zbunjuje jednu klasu s drugom, ili koliko često neuspješno prepoznaje određenu klasu.
# Model nije toliko dobro istreniran jer 837 pravih(tačnih) je prepoznao kao prave, kao i 10 lažnih je ispravno klasificirao kao lažne
# Model nema mnogo grešaka zbog toga su "niske" vrijednosti u žutim poljima
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Plot the Confusion Matrix as a HeatMap
class_names=["prava", "lazna"] # Name  of classes
fig, ax = plt.subplots()

# Generiranje matrice konfuzije
conf_matrix= confusion_matrix(y_test, y_pred_test)

# create heatmap
sns.heatmap(pd.DataFrame(conf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
plt.tight_layout()
ax.invert_yaxis()
ax.set_xticklabels(class_names)
ax.set_yticklabels(class_names)

plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

plt.show()

# ROC (Receiver Operating Characteristic) krivulja je graf koji prikazuje performanse binarnog klasifikatora na svim pragovima odluke.
# ROC kriva prikazuje odnos između stope lažno pozitivnih rezultata (False Positive Rate - FPR) i stope istinito pozitivnih rezultata (True Positive Rate - TPR) za različite pragove odluke.
# Bitna je površina ispod te krive i ona treba da teži ka 1 što bi značilo da naš model tada bi pravio ili male ili nikakve greške
# U našem slučaju, površina je 0.81 što je zadovoljavajuće jer će samo u 20% slučajeva pogriješiti, kriva stabilno se približava 1

from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Izračunajte vjerojatnosti pripadnosti pozitivnoj klasi
y_pred_proba = svm_weighted.decision_function(X_test_scaled)

# Izračunajte FPR i TPR
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

# Izračunajte AUC
auc = roc_auc_score(y_test, y_pred_proba)

# Crtanje ROC krivulje
plt.plot(fpr, tpr, label="SVM, AUC=" + str(auc))
plt.legend(loc=4)
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

# Ispis AUC-a
print('AUC Score:', auc)
print('The AUC Score provides an aggregate measure of performance across all possible classification thresholds.')
print('AUC ranges in value from 0 to 1. A model whose predictions are 100% wrong has an AUC of 0.0; one whose predictions are 100% correct')

# Odavdje vidimo da smo dobro izbalansirali model jer su preciznost i recall usko bliski i površina ispod krive je dosta visoka ali su rezultati bolji za SMOTE

from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

# Izračunaj vjerojatnosti pripadanja klasama
y_scores = svm_weighted.decision_function(X_test_scaled)

# Izračunaj preciznost i odziv
precision, recall, _ = precision_recall_curve(y_test, y_scores)

# Nacrtaj Precision-Recall krivulju
plt.figure()
plt.step(recall, precision, color='b', alpha=0.2, where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall Curve')
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Izdvojite potporne vektore
support_vectors = svm_weighted.support_vectors_

# Prikaz podataka i potpornih vektora u grafikonu
plt.scatter(X_train_resampled[:, 0], X_train_resampled[:, 1], c=y_train_resampled, cmap='winter', label='Podaci')
plt.scatter(support_vectors[:, 0], support_vectors[:, 1], color='red', marker='x', label='Potporni vektori')
plt.xlabel('Prva značajka')
plt.ylabel('Druga značajka')
plt.title('Podaci i potporni vektori')
plt.legend()
plt.show()



import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Izdvojite potporne vektore
support_vectors = svm_weighted.support_vectors_

# Prikaz podataka i potpornih vektora u 3D grafikonu
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Prikaz podataka
ax.scatter(X_train_resampled[:, 0], X_train_resampled[:, 1], X_train_resampled[:, 2], c=y_train_resampled, cmap='winter', label='Podaci')

# Prikaz potpornih vektora
ax.scatter(support_vectors[:, 0], support_vectors[:, 1], support_vectors[:, 2], color='red', marker='x', label='Potporni vektori')

ax.set_xlabel('Prva značajka')
ax.set_ylabel('Druga značajka')
ax.set_zlabel('Treća značajka')
ax.set_title('Podaci i potporni vektori u 3D prostoru')
ax.legend()

plt.show()

"""Selekcija features(RFE)

"""

# Preko linearne korelacije, preko težine atributa a i RFE vidimo da age je zaista najbitnija značajka
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

num_features = X_train_scaled.shape[1]  # Broj značajki u tvojim podacima
selected_features_indices = None

for i in range(num_features):
    logistic_reg = LogisticRegression()
    rfe = RFE(logistic_reg, n_features_to_select=num_features-i)  # running RFE
    rfe = rfe.fit(X_train_scaled, y_train)
    selected_features_indices = np.where(rfe.support_)[0]
    second_column_name = data.columns[selected_features_indices]
    print(f"\nNumber of retained features: {num_features - i}")
    print("Ime featurea:", second_column_name)

"""NEURALNE MREŽE

"""

import tensorflow as tf
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

# Normalizacija podataka
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Učenje i transformacija obučavajućih podataka

# Primjena SMOTE oversamplinga na trening skup
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# Transformacija test podataka koristeći istu normalizaciju kao za obučavajuće podatke
X_test_scaled = scaler.transform(X_test)  # Transformacija testnih podataka na osnovu učenja iz X_train

# Definiranje modela s dodatnim slojevima i dropout-om
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_resampled.shape[1],)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

# Callback za praćenje performansi i ranog zaustavljanja
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
]

# Treniranje modela
history = model.fit(X_train_resampled, y_train_resampled,
                    epochs=90,
                    batch_size=80,
                    validation_data=(X_test_scaled, y_test),
                    callbacks=callbacks)

# Ispis arhitekture modela
print("Model Architecture:")
model.summary()

# Evaluacija modela
loss, accuracy, precision, recall = model.evaluate(X_test_scaled, y_test)

print("Loss:", loss)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)