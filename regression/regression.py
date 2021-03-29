import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow import keras


# make numpy printouts easier to read
np.set_printoptions(precision=3, suppress=True)

# Obtenir les données (dataset Auto MPG)
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
column_names = [
    "MPG",
    "Cylinders",
    "Displacement",
    "Horsepower",
    "Weight",
    "Acceleration",
    "Model Year",
    "Origin",
]

raw_dataset = pd.read_csv(
    url, names=column_names, na_values="?", comment="\t", sep=" ", skipinitialspace=True
)

dataset = raw_dataset.copy()

# ------- Nettoyage
# check for bad stuff
# print(dataset.isna().sum())

# clean if bad stuff
dataset = dataset.dropna()

dataset["Origin"] = dataset["Origin"].map({1: "USA", 2: "Europe", 3: "Japan"})
dataset = pd.get_dummies(dataset, prefix="", prefix_sep="")
# ------- END


# ------- Division des données
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)
# ------- END

# sns.pairplot(train_dataset[['MPG', 'Cylinders', 'Displacement', 'Weight']], diag_kind='kde')
# plt.show()

# ------- Diviser les fonctionnalités des étiquettes
train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop("MPG")
test_labels = test_features.pop("MPG")
# ------- END

# ------- Normalisation
# Recommandé de normaliser les entités qui utilisent des échelles et des plages différentes
# print(train_dataset.describe().transpose()[["mean", "std"]])

# couche de normalisation
normalizer = preprocessing.Normalization()
# calcule la moyenne et la variance et les stocke dans la couche (de normalisation)
normalizer.adapt(np.array(train_features))

first = np.array(train_features[:1])

# with np.printoptions(precision=2, suppress=True):
#    print('First example:', first)
#    print()
#    print('Normalized:', normalizer(first).numpy())
# ------- END

# ------- Régression linaire
# 1ére couche de normalisation la "puissance"
horsepower = np.array(train_features["Horsepower"])

horsepower_normalizer = preprocessing.Normalization(input_shape=[1])
horsepower_normalizer.adapt(horsepower)

# Construire le modèle séquentiel
horsepower_model = tf.keras.Sequential([horsepower_normalizer, layers.Dense(units=1)])
# horsepower_model.summary()
# horsepower_model prédira MPG à partir de la "puissance" (Horsepower)

# prédiction sur les 10 première valeurs de puissance
# avec un modèle NON entraîné
# print(horsepower_model.predict(horsepower[:10]))

print("Compiling model..")
horsepower_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss="mean_absolute_error",
)
print("Compilation ended.")
print("Training horsepower_model..")
history = horsepower_model.fit(
    train_features["Horsepower"],
    train_labels,
    epochs=100,
    # supress verbose
    verbose=0,
    # calculate validation results on 20% of the training data
    validation_split=0.2,
)
print("Training ended.")

# Visualiser la progression de l'entrainement du modèle
# hist = pd.DataFrame(history.history)
# hist['epoch'] = history.epoch
# print(hist.tail())


def plot_loss(history, name):
    plt.clf()
    plt.plot(history.history["loss"], label="loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.ylim([0, 10])
    plt.xlabel("Epoch")
    plt.ylabel("Error [MPG]")
    plt.legend()
    plt.grid(True)
    plt.savefig(name)


plot_loss(history, "horsepower_model training.png")

test_results = {}
test_results["horsepower_model"] = horsepower_model.evaluate(
    test_features["Horsepower"],
    test_labels,
    verbose=0,
)

x = tf.linspace(0.0, 250, 251)
y = horsepower_model.predict(x)


def plot_horsepower(x, y, name):
    plt.clf()
    plt.scatter(train_features["Horsepower"], train_labels, label="Data")
    plt.plot(x, y, color="k", label="Predictions")
    plt.xlabel("Horsepower")
    plt.ylabel("MPG")
    plt.legend()
    plt.savefig(name)


# plot_horsepower(x, y, "horsepower_pred.png")
# ------- END


# ------- Entrées multiples
linear_model = tf.keras.Sequential(
    [
        normalizer,
        layers.Dense(units=1),
    ]
)

linear_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss="mean_absolute_error",
)

print("Training linear_model..")
history = linear_model.fit(
    train_features,
    train_labels,
    epochs=100,
    verbose=0,
    validation_split=0.2,
)
print("Training ended.")
plot_loss(history, "linear_model trainig.png")

test_results["linear_model"] = linear_model.evaluate(
    test_features,
    test_labels,
    verbose=0,
)
# ------- END

# ------- Régression DNN
def build_and_compile_model(norm):
    model = keras.Sequential(
        [
            norm,
            layers.Dense(64, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(1),
        ]
    )

    model.compile(loss="mean_absolute_error", optimizer=tf.keras.optimizers.Adam(0.001))

    return model


# Une variable
dnn_horsepower_model = build_and_compile_model(horsepower_normalizer)
# dnn_horsepower_model.summary()

print("Training dnn_horsepower_model..")
history = dnn_horsepower_model.fit(
    train_features["Horsepower"],
    train_labels,
    validation_split=0.2,
    verbose=0,
    epochs=100,
)
print("Training ended.")
plot_loss(history, "dnn_horsepower_model trainig.png")

test_results["dnn_horsepower_model"] = dnn_horsepower_model.evaluate(
    test_features["Horsepower"],
    test_labels,
    verbose=0,
)
# Multiple variables
dnn_model = build_and_compile_model(normalizer)


print("Training dnn_model..")
history = dnn_model.fit(
    train_features,
    train_labels,
    validation_split=0.2,
    verbose=0,
    epochs=100,
)
print("Training ended.")
plot_loss(history, "dnn_model training.png")
test_results["dnn_model"] = dnn_model.evaluate(
    test_features,
    test_labels,
    verbose=0,
)
# ------- END

# ------- Comparing results
print(pd.DataFrame(test_results, index=["Mean absoulte error [MPG]"]).T)
# ------- END

# ------- Faire les prédictions
test_predictions = dnn_model.predict(test_features).flatten()

a = plt.axes(aspect="equal")
plt.scatter(test_labels, test_predictions)
plt.xlabel("True Values [MPG]")
plt.ylabel("Predictions [MPG]")
lims = [0, 50]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
plt.savefig("dnn_model predictions.png")

# Afficher la distribution des erreurs
error = test_predictions - test_labels
plt.cla()
plt.hist(error, bins=25)
plt.xlabel("Prediction Error [MPG]")
_ = plt.ylabel("Count")
plt.show()
# ------- END

# Trying saving and loading a model :p
# Save model
dnn_model.save("dnn_model")

# Load model
reloaded = tf.keras.models.load_model("dnn_model")

# Add loaded model to test_results
test_results["reloaded"] = reloaded.evaluate(test_features, test_labels, verbose=0)

# ------- Comparing results (especially of the load)
print(pd.DataFrame(test_results, index=["Mean absolute error [MPG]"]).T)
# ------- END
