import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# import seaborn as sns
# import seaborn as sns
import os

# <<<<<<< Tabnine <<<<<<<
def olah_data(filename):
    # Load the dataset
    df = pd.read_excel(filename)


    # mengisi missing value dengan mean masing maisng kolom
    for column in df.columns:
        if df[column].dtype in ['float64', 'int64']:  # Hanya untuk kolom numerik
            df[column].fillna(df[column].mean(), inplace=True)

    # Menentukan kolom terakhir
    last_column = df.columns[-1]

# Mengisi missing value di kolom terakhir dengan modus kolom tersebut
    df[last_column].fillna(df[last_column].mode()[0], inplace=True)

    # Inisialisasi LabelEncoder
    encoding = LabelEncoder()

    # Identifikasi kolom bertipe object
    object_cols = df.select_dtypes(include='object').columns

    # Transformasi kolom bertipe object
    for col in object_cols:
        df[col] = encoding.fit_transform(df[col])

    # Menentukan label dan feature , label diatur hanya untuk kolom terakhir.
    # Definisikan feature dan label
    feature = df.iloc[:, :-1].values  # Semua kolom kecuali kolom terakhir
    label = df.iloc[:, -1].values    # Kolom terakhir

    # Normalisasi data
    scaller = MinMaxScaler()
    feature = scaller.fit_transform(feature)

    # Split data menjadi train dan test
    X_train, X_test, y_train, y_test = train_test_split(feature, label, test_size=0.2, random_state=42)
    # x_train, x_test, y_train, y_test = train_test_split(feature, label, test_size=0.2, random_state=50)
    print(f"Jumlah data train: {len(X_train)}")
    print(f"Jumlah data test: {len(X_test)}")

    while True:
        print("Pilih model yang ingin digunakan:")
        print("1. Gaussian")
        print("2. KNN")
        print("3. Keluar")
        input_user = input("Masukkan pilihan Anda (1/2): ")
        if input_user == "1":
            GNB = GaussianNB()
            GNB.fit(X_train, y_train)
            pred = GNB.predict(X_test)
            accuracy = accuracy_score(y_test, pred)
            print(f"Akurasi model Gaussian Naive Bayes: {accuracy}")
            os.system("pause")
            os.system("cls")

        elif input_user == "2":
            KNN = KNeighborsClassifier(n_neighbors=5)
            KNN.fit(X_train, y_train)
            pred = KNN.predict(X_test)
            accuracy = accuracy_score(y_test, pred)
            print(f"Akurasi model KNN: {accuracy}")
            os.system("pause")
            os.system("cls")

        elif input_user == "3":
            break



    
def main():
        while True:
            print("1. MULAI PROGRAM")
            print("2. Keluar")
            inputPilihan = input("Masukkan pilihan Anda (1/2): ")
            if inputPilihan == "1":
                input_file = input("Masukkan nama file Excel: ")
                olah_data(input_file)
            else:
                break
        
    
if __name__ == "__main__":
    main()


