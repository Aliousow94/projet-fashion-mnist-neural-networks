# Classification d'images Fashion-MNIST avec NumPy et TensorFlow/Keras

## 📌 Description

Ce projet consiste à développer et comparer deux approches de **réseaux de neurones artificiels** pour la classification d'images de la base **Fashion-MNIST** :

* une implémentation réalisée **from scratch avec NumPy** ;
* une implémentation utilisant **TensorFlow/Keras**.

L'objectif est de comprendre le fonctionnement d'un réseau de neurones à travers une implémentation manuelle avec NumPy, puis de découvrir comment **TensorFlow/Keras automatise ces différentes étapes**.

---

## 🎯 Objectifs

Les principaux objectifs du projet sont :

* comprendre le fonctionnement d'un réseau de neurones ;
* manipuler et prétraiter des images avec Python ;
* implémenter manuellement un réseau de neurones avec NumPy ;
* utiliser TensorFlow/Keras pour construire un modèle de Deep Learning ;
* entraîner et évaluer les modèles ;
* comparer les deux approches.

---

## 📊 Dataset : Fashion-MNIST

Fashion-MNIST est une base de données d'images représentant différentes catégories de vêtements.

Elle contient :

* **60 000 images d'entraînement** ;
* **10 000 images de test** ;
* des images en niveaux de gris de **28 × 28 pixels** ;
* **10 classes** de vêtements.

Les 10 classes sont :

| Label | Classe        |
| ----: | ------------- |
|     0 | T-shirt / Top |
|     1 | Pantalon      |
|     2 | Pull          |
|     3 | Robe          |
|     4 | Manteau       |
|     5 | Sandale       |
|     6 | Chemise       |
|     7 | Basket        |
|     8 | Sac           |
|     9 | Bottine       |

---

# 🧠 Approche 1 — Réseau de neurones avec NumPy

Dans cette première partie, le réseau de neurones est développé **from scratch avec NumPy**, sans utiliser de framework spécialisé de Deep Learning.

### Architecture

```text
784 → 32 → 32 → 32 → 1
```

| Couche   | Neurones | Rôle                               |
| -------- | -------: | ---------------------------------- |
| Entrée   |      784 | Représentation des pixels          |
| Cachée 1 |       32 | Apprentissage des caractéristiques |
| Cachée 2 |       32 | Combinaison des caractéristiques   |
| Cachée 3 |       32 | Représentation plus complexe       |
| Sortie   |        1 | Classification binaire             |

Dans cette expérimentation, le problème est simplifié en classification binaire :

```text
1 → T-shirt / Top
0 → Autres catégories
```

### Étapes implémentées

1. Chargement des données ;
2. Visualisation des images ;
3. Normalisation des pixels ;
4. Transformation des images `28 × 28` en vecteurs de `784` valeurs ;
5. Création de la classification binaire ;
6. Initialisation des poids et des biais ;
7. Propagation avant ;
8. Calcul de la Log Loss ;
9. Rétropropagation ;
10. Descente de gradient ;
11. Évaluation du modèle.

### Résultats NumPy

L'expérimentation a été réalisée sur **1 000 images d'entraînement** pendant **50 itérations**.

| Métrique              |   Résultat |
| --------------------- | ---------: |
| Accuracy entraînement | **88,6 %** |
| Accuracy test         | **89,0 %** |

---

# 🤖 Approche 2 — Réseau de neurones avec TensorFlow/Keras

Dans la deuxième partie, le même dataset Fashion-MNIST est utilisé pour réaliser une **classification multiclasses des 10 catégories** avec TensorFlow/Keras.

### Architecture

```text
784 → 128 → 64 → 10
```

| Couche   | Neurones | Activation | Rôle                                       |
| -------- | -------: | ---------- | ------------------------------------------ |
| Entrée   |      784 | —          | Pixels de l'image                          |
| Cachée 1 |      128 | ReLU       | Extraction des caractéristiques            |
| Cachée 2 |       64 | ReLU       | Apprentissage de représentations complexes |
| Sortie   |       10 | Softmax    | Classification des 10 classes              |

### Configuration

Le modèle utilise :

* **Optimiseur :** Adam
* **Fonction de perte :** Sparse Categorical Crossentropy
* **Époques :** 10
* **Batch size :** 32
* **Validation :** 10 % des données d'entraînement

### Résultats TensorFlow/Keras

| Métrique              |    Résultat |
| --------------------- | ----------: |
| Accuracy entraînement | **91,01 %** |
| Accuracy validation   | **88,48 %** |
| Accuracy test         | **88,23 %** |

Le modèle a également été évalué avec :

* une matrice de confusion ;
* un rapport de classification ;
* des exemples de prédictions ;
* une visualisation des erreurs.

---

# 📊 Comparaison des deux approches

| Élément               | NumPy                   | TensorFlow/Keras        |
| --------------------- | ----------------------- | ----------------------- |
| Type                  | Implémentation manuelle | Framework Deep Learning |
| Architecture          | 784 → 32 → 32 → 32 → 1  | 784 → 128 → 64 → 10     |
| Classification        | Binaire                 | Multiclasse             |
| Propagation avant     | Programmée manuellement | Automatisée             |
| Rétropropagation      | Programmée manuellement | Automatisée             |
| Descente de gradient  | Programmée manuellement | Gérée par l'optimiseur  |
| Optimisation          | Gradient Descent        | Adam                    |
| Fonction d'activation | Sigmoïde                | ReLU + Softmax          |
| Accuracy test         | **89,0 %**              | **88,23 %**             |

> Les performances ne doivent pas être comparées directement comme celles de deux modèles identiques, car les deux expérimentations utilisent des **architectures, des tâches et des configurations différentes**.

L'intérêt principal de cette comparaison est donc de comprendre la différence entre une **implémentation manuelle** et l'utilisation d'un **framework de Deep Learning**.

---

# 📈 Visualisations

Le projet contient plusieurs visualisations permettant d'analyser les performances des modèles :

* courbes d'apprentissage ;
* évolution de la Loss ;
* évolution de l'Accuracy ;
* matrice de confusion ;
* exemples de prédictions ;
* exemples de classifications incorrectes.

Les résultats graphiques sont disponibles dans :

```text
results/figures/
```

---

# 📁 Structure du projet

```text
projet-fashion-mnist/
│
├── data/
│
├── notebooks/
│   ├── 01_Fashion_MNIST_NumPy.ipynb
│   └── 02_Fashion_MNIST_TensorFlow.ipynb
│
├── src/
│
├── models/
│
├── results/
│   └── figures/
│
├── README.md
├── requirements.txt
├── .gitignore
└── rapport.pdf
```

---

# 🛠️ Technologies utilisées

### Python / NumPy

* Python 3.12
* NumPy
* Matplotlib
* Jupyter Notebook

### TensorFlow

* TensorFlow
* Keras
* NumPy
* Matplotlib
* Scikit-learn

### Environnement

* VS Code
* Google Colab pour l'expérimentation TensorFlow

---

# 🚀 Exécution du projet

## 1. Installer les dépendances

Pour l'approche NumPy :

```bash
pip install numpy matplotlib jupyter
```

Pour l'approche TensorFlow :

```bash
pip install tensorflow numpy matplotlib scikit-learn
```

> L'implémentation TensorFlow peut également être exécutée directement sur **Google Colab**.

## 2. Lancer les notebooks

### NumPy

```text
notebooks/01_Fashion_MNIST_NumPy.ipynb
```

### TensorFlow/Keras

```text
notebooks/02_Fashion_MNIST_TensorFlow.ipynb
```

---

# 🔮 Perspectives

Ce projet peut être amélioré en :

* utilisant l'ensemble des **60 000 images d'entraînement** pour l'expérimentation NumPy ;
* testant différentes architectures de réseaux de neurones ;
* améliorant les hyperparamètres ;
* utilisant davantage de techniques de régularisation ;
* expérimentant avec des **CNN (Convolutional Neural Networks)** ;
* comparant les performances avec d'autres architectures ;
* étudiant l'impact de différents optimiseurs ;
* utilisant TensorFlow/Keras pour développer des modèles de Deep Learning plus complexes.

---

# 👨‍💻 Auteur

**Mamadou Aliou SOW**

Licence 3 Informatique
Université Amadou Mahtar Mbow (UAM)

---

## 📚 Résumé du projet

Ce projet présente deux façons de construire un réseau de neurones pour Fashion-MNIST :

```text
                 Fashion-MNIST
                       │
              ┌────────┴────────┐
              │                 │
            NumPy          TensorFlow/Keras
              │                 │
       Implémentation       Framework
         manuelle          automatisé
              │                 │
        Classification       Classification
          binaire             10 classes
              │                 │
           89,0 %             88,23 %
          test                test
```

L'objectif principal est de comprendre à la fois **les mécanismes internes d'un réseau de neurones** et **l'utilisation d'un framework moderne de Deep Learning**.
