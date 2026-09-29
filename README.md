# Classification d'images Fashion-MNIST avec NumPy

## 📌 Description

Ce projet consiste à développer **from scratch un réseau de neurones artificiel avec NumPy** pour effectuer une classification binaire sur la base de données **Fashion-MNIST**.

L'objectif est de comprendre les principales étapes du fonctionnement d'un réseau de neurones sans utiliser de bibliothèque spécialisée de Deep Learning.

## 🎯 Objectif

Le problème initial de Fashion-MNIST contient 10 classes de vêtements.

Dans ce projet, nous avons simplifié le problème en une **classification binaire** :

* `1` → T-shirt/top
* `0` → autres catégories

## 📊 Dataset

Fashion-MNIST contient :

* **60 000 images d'entraînement**
* **10 000 images de test**
* Images en niveaux de gris de **28 × 28 pixels**
* **10 classes** à l'origine

Chaque image est transformée en un vecteur de **784 pixels**.

## 🧠 Architecture du réseau

Le réseau utilisé possède l'architecture suivante :

```text
784 → 32 → 32 → 32 → 1
```

| Couche   | Neurones | Rôle                               |
| -------- | -------: | ---------------------------------- |
| Entrée   |      784 | Pixels de l'image                  |
| Cachée 1 |       32 | Apprentissage des caractéristiques |
| Cachée 2 |       32 | Combinaison des caractéristiques   |
| Cachée 3 |       32 | Représentation plus complexe       |
| Sortie   |        1 | Classification binaire             |

Les couches utilisent la fonction d'activation **sigmoïde**.

## ⚙️ Méthodes implémentées

Le réseau a été entièrement développé avec **NumPy**.

Les principales étapes sont :

1. Chargement des données Fashion-MNIST
2. Visualisation des images
3. Normalisation des pixels
4. Transformation des images `28 × 28` en vecteurs de `784`
5. Classification binaire
6. Initialisation des poids et des biais
7. Propagation avant
8. Calcul de la **Log Loss**
9. Rétropropagation
10. Descente de gradient
11. Évaluation du modèle

## 📈 Résultats

Pour l'expérimentation réalisée sur **1 000 images d'entraînement** pendant **50 itérations**, nous avons obtenu :

| Métrique              |   Résultat |
| --------------------- | ---------: |
| Accuracy entraînement | **88,6 %** |
| Accuracy test         | **89,0 %** |

Les courbes de Loss et d'Accuracy sont disponibles dans le dossier :

```text
results/figures/
```

## 📁 Structure du projet

```text
projet_fashion_mnist/
│
├── data/
│   ├── train-images-idx3-ubyte.gz
│   ├── train-labels-idx1-ubyte.gz
│   ├── t10k-images-idx3-ubyte.gz
│   └── t10k-labels-idx1-ubyte.gz
│
├── notebooks/
│   └── 01_Fashion_MNIST_NumPy.ipynb
│
├── src/
│
├── models/
│
├── results/
│   └── figures/
│     
│       
│       └── courbes_apprentissage.png
│
├── README.md
└── .gitignore
```

## 🛠️ Technologies utilisées

* **Python 3.12**
* **NumPy**
* **Matplotlib**
* **Jupyter Notebook**
* **VS Code**

## 🚀 Exécution du projet

Après avoir installé Python et les dépendances nécessaires :

```bash
pip install numpy matplotlib jupyter
```

Lancer ensuite le notebook :

```bash
jupyter notebook
```

Puis ouvrir :

```text
notebooks/01_Fashion_MNIST_NumPy.ipynb
```

## 🔮 Perspectives

Ce projet peut être amélioré en :

* utilisant les **60 000 images d'entraînement** ;
* augmentant le nombre d'itérations ;
* testant différentes architectures ;
* utilisant **ReLU** pour les couches cachées ;
* réalisant une classification des **10 classes originales** ;
* comparant l'implémentation NumPy avec des frameworks de Deep Learning.

## 👨‍💻 Auteur

**Mamadou Aliou SOW**
Licence 3 Informatique — Université Amadou Mahtar Mbow (UAM)
