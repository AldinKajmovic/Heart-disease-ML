# Predikcija srčanog udara

Ovaj projekat ima za cilj da koristi dataset sa informacijama o osobama kako bi predvidio da li je osoba sklona srčanom udaru ili ne. 

## Dataset

Dataset sadrži sledeće kolone:

- **id**: Unikatni identifikator osobe (int).
- **gender**: Spol osobe (string).
- **age**: Broj godina osobe (int).
- **hypertension**: Da li osoba ima hipertenziju (0 ako nema, 1 ako ima; bool).
- **heart_disease**: Da li osoba ima oboljenje srca (0 ako nema, 1 ako ima; bool).
- **ever_married**: Da li je osoba ikada bila u braku (string).
- **work_type**: Način zapošljenja (string).
- **residence_type**: Mejsto življenja (string).
- **avg_glucose_level**: Prosječna količina glukoze u krvi (float).
- **bmi**: BMI index osobe (float).
- **smoking_status**: Da li osoba konzumira cigarete (string).
- **stroke**: Da li je osoba sklona srčanom udaru (1 ako jeste, 0 ako nije; bool). Ovo je ciljna vrijednost koju model treba da predviđa.

## Metode

U ovom projektu su korišćene različite metode za treniranje modela:

- **SVM (Support Vector Machines)**
- **Neuronske mreže**

## Instalacija


1. **Klonirajte repozitorijum**:
   ```bash
   git clone https://github.com/AldinKajmovic/heart_disease_ml.git
2. **Otvorite Google Collab ili Code Editor**
   
