# eeg-seizure-classification
This project explores the use of machine learning to classify EEG signals for seizure detection. The goal is to build and evaluate different classification models to accurately identify various types of EEG signals, including seizure events, focal seizures, generalized seizures, and healthy brain activity.

Dataset
The dataset used in this project is the BEED: Bangalore EEG Epilepsy Dataset from the UCI Machine Learning Repository.

Source: [(https://archive.ics.uci.edu/dataset/1134/beed:+bangalore+eeg+epilepsy+dataset)]
The dataset contains EEG recordings from 80 individuals, categorized into four classes:

Class 0: Healthy subjects (control group)
Class 1: Generalized seizures
Class 2: Focal seizures
Class 3: Seizure events (e.g., eye blinking)
The dataset is balanced, with 2,000 samples for each class. Each sample consists of 16 features representing EEG signal characteristics.
Methods
Two main approaches were explored in this project:

1. Single-Stage Classification Models:

Three standard classification models were trained on the raw EEG features:
Logistic Regression
Support Vector Machine (SVM)
Random Forest
2. Two-Stage Model with Feature Engineering:

A more complex, two-stage model was implemented with the following pipeline:
Feature Engineering: FFT band-power features (Delta, Theta, Alpha, Beta, Gamma) were extracted from the EEG signals.
Dimensionality Reduction: UMAP was used to reduce the dimensionality of the band-power features.
AdaBoost Model: An AdaBoost classifier was trained on the UMAP embeddings.
LSTM Meta-Model: An LSTM meta-model was trained on the probability scores from the AdaBoost model to make the final classification.
Results
The performance of the different models was evaluated using accuracy as the primary metric:

Logistic Regression: 45% accuracy
SVM: 74% accuracy
Random Forest: 96% accuracy
Two-Stage Model (AdaBoost + LSTM): 63% accuracy
The Random Forest model trained on the original EEG features achieved the best performance, with an impressive accuracy of 96%.

Conclusion
This project demonstrates that a well-tuned Random Forest model can be highly effective for classifying EEG signals for seizure detection. While a more complex, two-stage model with feature engineering was explored, the simpler approach of using the raw EEG features with a powerful ensemble model like Random Forest yielded the best results.

How to Run the Notebook
To run this notebook, you will need to have Python 3 and the following libraries installed:

pandas
numpy
scikit-learn
matplotlib
seaborn
tensorflow
umap-learn
You can install these libraries using pip:

pip install pandas numpy scikit-learn matplotlib seaborn tensorflow umap-learn
Then, you can open the eeg-seizure-classification.ipynb file in a Jupyter Notebook or Google Colab environment and run the cells in order.

