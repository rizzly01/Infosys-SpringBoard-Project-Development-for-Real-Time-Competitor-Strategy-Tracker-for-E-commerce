# Infosys-SpringBoard-Project-Development-for-Real-Time-Competitor-Strategy-Tracker-for-E-commerce
Developed a project to analyze and track competitors’ pricing, product availability, and promotional strategies in real time for e-commerce platforms. The system helps businesses make data-driven decisions by providing insights into market trends, competitor behavior, and dynamic pricing strategies.

# 🚀 Milestone 1: Infrastructure, Tooling & Foundations


The goal of this milestone was to establish the development environment and validate the mathematical foundations of Deep Learning. I transitioned from writing raw mathematical logic in NumPy to leveraging industry-standard frameworks like PyTorch and TensorFlow to solve the MNIST handwritten digit classification problem.

---

## 🛠️ Tech Stack & Essential Tools

In this phase, I installed and configured the core dependencies required for high-performance tensor manipulation and model training:

* **Computational Engines:** * `NumPy`: Used for building the initial neural network logic from scratch.
* `CuPy`: Integrated to offload heavy matrix multiplications to the **NVIDIA GPU** for 10x faster training.


* **Deep Learning Frameworks:** * `PyTorch`: Leveraged for its dynamic computational graphs and `nn.Module` infrastructure.
* `TensorFlow/Keras`: Used to build high-level `Sequential` pipelines for rapid benchmarking.


* **Evaluation & Visualization:**
* `Matplotlib`: Used to plot Loss and Accuracy curves to monitor convergence.
* `Scikit-learn`: Utilized for calculating final `accuracy_score` metrics.



---

## 🧠 Model Development: From Scratch to Frameworks

### 1. The "From-Scratch" Logic Gate Model

Before using high-level libraries, I implemented a `GateNeuralNetwork` class to solve binary logic problems.

* **Manual Backpropagation:** Implemented the Chain Rule manually to calculate gradients for each layer.
* **Weight Tracking:** Added trackers (`FWC`, `MWC`, `LWC`) to observe the magnitude of weight changes, ensuring the model was actually learning and not just oscillating.

### 2. High-Level Framework Implementation (TensorFlow/Keras)

I built and trained a robust Multi-Layer Perceptron (MLP) with the following architecture:

* **Input Layer:** Flattened  images into a -pixel vector.
* **Hidden Layers:** Two dense layers with **Sigmoid** activation ( and  neurons).
* **Output Layer:**  neurons with **Softmax** activation to produce probability distributions for digits .
* **Optimization:** Used the `RMSprop` optimizer and `BinaryCrossentropy` loss function.

---

## 📊 Key Results & Observations

* **Rapid Convergence:** The TensorFlow model achieved **>91% accuracy** in the very first epoch.
* **Peak Performance:** Reached a final validation accuracy of **~98.06%** after 100 epochs.
* **Overfitting Analysis:** By plotting training vs. validation loss, I identified that the model reached its maximum generalization around Epoch 25, after which training loss continued to fall while validation loss stabilized.

---

## 📂 Deliverables

* ✅ **Infrastructure Ready:** GPU acceleration confirmed and library dependencies documented.
* ✅ **Data Pipeline:** Automated MNIST downloading, normalization, and one-hot encoding scripts.
* ✅ **Trained Weights:** Final weights exported to `.npy` format for future inference without re-training.

