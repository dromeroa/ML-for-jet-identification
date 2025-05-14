# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import uproot
import awkward as ak
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 1. Load data from a ROOT file (TTJets MC, MiniAOD format)
file = uproot.open("TTJets_TuneCUETP8M1_13TeV-madgraphMLM-pythia8.root")
tree = file["Events"]

# Extract features from tree (use real branch names)
features = [
    "slimmedJetsAK8.pt", "slimmedJetsAK8.eta", "slimmedJetsAK8.phi", "slimmedJetsAK8.mass",
    "slimmedJetsAK8.userFloat('NjettinessAK8:tau1')",
    "slimmedJetsAK8.userFloat('NjettinessAK8:tau2')",
    "slimmedJetsAK8.userFloat('NjettinessAK8:tau3')",
    "slimmedJetsAK8.userFloat('ak8PFJetsCHSValueMap:massPruned')",
    "slimmedJetsAK8.userFloat('ak8PFJetsCHSValueMap:SoftDropMass')",
    "GenJetAK8.genPartIdx"
]

arrays = tree.arrays(features, library="ak")

# Generator-level particles (truth info)
gen_parts = tree.arrays(["GenPart.pdgId", "GenPart.status", "GenPart.pt"], library="ak")

# Flatten and convert to pandas
flat_arrays = ak.to_pandas(arrays).dropna()
flat_genparts = ak.to_pandas(gen_parts)

# Rename columns for simplicity
flat_arrays.columns = [
    "pt", "eta", "phi", "mass",
    "tau1", "tau2", "tau3", "prunedMass", "softdropMass",
    "genIdx"
]

# Merge generator-level information by index
flat_arrays = flat_arrays.reset_index()
flat_genparts = flat_genparts.reset_index()

merged = flat_arrays.merge(flat_genparts, left_on=["index", "genIdx"], right_on=["index", "subentry"], how="left")

# Label assignment based on generator-level PDG ID
merged["label"] = merged["GenPart.pdgId"].abs().apply(lambda x: 1 if x in [6, 23, 24] else 0)

# Drop rows without matched generator info
merged = merged.dropna(subset=["GenPart.pdgId"])

# 2. Preprocessing
features = ["pt", "eta", "phi", "mass", "tau1", "tau2", "tau3", "prunedMass", "softdropMass"]
scaler = StandardScaler()
features_scaled = scaler.fit_transform(merged[features])

X = pd.DataFrame(features_scaled, columns=features)
y = merged["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 3. Neural Network Model
model = Sequential([
    Dense(64, input_shape=(X_train.shape[1],), activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

# Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 4. Train model
history = model.fit(X_train, y_train,
                    validation_split=0.2,
                    epochs=50,
                    batch_size=32,
                    callbacks=[early_stop],
                    verbose=1)

# 5. Evaluate performance
y_pred_prob = model.predict(X_test).ravel()
y_pred = (y_pred_prob > 0.5).astype(int)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_pred_prob))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# 6. Dimensionality Reduction (PCA and t-SNE)
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X)

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
tsne_result = tsne.fit_transform(X)

# PCA plot
plt.figure(figsize=(8, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=y, cmap='coolwarm', alpha=0.6)
plt.title("PCA projection of jet features")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar(label='Jet type')
plt.show()

# t-SNE plot
plt.figure(figsize=(8, 6))
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=y, cmap='coolwarm', alpha=0.6)
plt.title("t-SNE projection of jet features")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.colorbar(label='Jet type')
plt.show()
