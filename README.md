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

---
---

## 🕷️ Milestone 2: Web Scraping & Data Aggregation

### 📌 Overview

In this milestone, I transitioned from using static, pre-existing datasets to building a **custom data pipeline**. I developed a robust web crawler that automates the collection of unstructured data from the web and transforms it into a structured, "AI-ready" format. This stage is critical for real-world AI applications where data must be sourced and cleaned manually.

---

### 🛠️ Tech Stack & Advanced Tooling

To handle dynamic content and efficient data storage, I utilized:

* **Automation Engines:**
* `Playwright`: Used for high-speed browser automation and handling JavaScript-heavy pages.
* `BeautifulSoup4` & `lxml`: Employed for precise HTML parsing and "Deep Scrapping" of text content.


* **Data Processing:**
* `Pandas`: The primary engine for data aggregation, cleaning, and CSV/JSON export.
* `Regex (re)`: Used for data normalization (e.g., converting currency strings into floats).


* **Intelligence Layer:**
* `Transformers (RoBERTa)`: Integrated a Large Language Model to perform real-time sentiment analysis on the scraped headlines.



---

### 🧠 Core Engineering Concepts

#### **1. Master-Detail Aggregation Pattern**

Instead of a surface-level scrape, I implemented a **Two-Tier Crawler**. The script first scans a "Master" list of items and then "drills down" into each individual "Detail" page. This allows the collection of rich metadata like product descriptions, categories, and technical specifications that are hidden from the main view.

#### **2. Data Normalization & Transformation**

Raw web data is often "noisy." I built a transformation layer to:

* Convert text-based ratings (e.g., "Three") into numerical integers (`3`).
* Strip currency symbols and convert price strings into mathematical floats.
* Filter out "Stop Words" and special characters to extract high-value trending keywords.

#### **3. Ethical & Robust Crawling**

To ensure the scraper operates reliably in a production environment:

* **Politeness Latency:** Implemented `time.sleep()` intervals to mimic human behavior and avoid IP banning.
* **Dynamic Pagination:** Developed logic to automatically discover and follow "Next" buttons until the entire 1,000-item catalog was aggregated.
* **Error Handling:** Used `try-except` blocks to ensure that a single broken link doesn't crash the entire multi-hour scraping session.

---

### 📊 Deliverables & Results

* ✅ **Structured Dataset:** A consolidated `books.csv` featuring 1,000+ entries with 8+ data dimensions.
* ✅ **Trend Analysis:** Automated identification of the "Top 5 Trending Keywords" from live Google News feeds.
* ✅ **Sentiment Index:** A real-time market mood indicator powered by the **RoBERTa LLM**, mapping global headlines to a polarity score between -1 and 1.

---
---


# 📊 Milestone 3: AI Sentiment Analysis & Semantic Modeling

## 🎯 Overview

In this milestone, I moved beyond data collection to perform **Deep Text Analytics**. By integrating a Transformer-based LLM and mathematical similarity metrics, I developed a **Weighted Popularity Index** to rank the scraped bookstore data based on emotional tone, information density, and semantic quality.

## 🛠️ Technical Implementation

### 1. Contextual Sentiment Analysis (LLM)

Instead of using basic word-matching, I implemented the **cardiffnlp/twitter-roberta-base-sentiment-latest** model. This RoBERTa-based Transformer understands context, sarcasm, and complex sentence structures to classify descriptions into **Positive**, **Neutral**, or **Negative** categories.

### 2. Lexical Diversity (Jaccard Distance)

I used **Jaccard Distance** to measure the "Information Gap" between a book's Title and its Description.

* **Logic:** Calculates the intersection over union of word sets.
* **Goal:** To quantify how much unique narrative information is provided on the product page versus the surface-level metadata.

### 3. Semantic Alignment (Cosine Similarity)

I implemented **TF-IDF Vectorization** to map book descriptions into a multi-dimensional vector space.

* **Benchmark:** Every book was compared against a "Gold Standard" anchor (a description representing a literary masterpiece).
* **Metric:** Calculated the **Cosine Similarity** () to determine thematic alignment.

---

## 📈 The Popularity Index Formula

The core of this milestone is a custom-engineered **Popularity Index (0-100)**, calculated using a weighted linear combination:

| Feature | Weight | Purpose |
| --- | --- | --- |
| **Sentiment** | 40% | Measures the emotional "vibe" and market appeal. |
| **Cosine Similarity** | 40% | Measures literary quality and thematic depth. |
| **Jaccard Distance** | 20% | Measures informational richness and variety. |

---

## 🧪 Results & Deliverables

The pipeline generates a final report: `milestone3_popularity_report.csv`.

**Top 20 Analysis Observation:**

* **Semantic Clustering:** Books that used professional and evocative language (high Cosine score) significantly outperformed generic descriptions in the final index.
* **Model Accuracy:** The RoBERTa model successfully identified "Dark Dramas" as high-quality content, proving superior to basic NLP libraries which often mistake dark themes for negative sentiment.

## 🚀 How to Run

1. Ensure you have the required libraries: `pip install transformers torch scikit-learn pandas nltk`
2. Run the analysis script: `python milestone3_analysis.py`
3. View the generated report: `milestone3_popularity_report.csv`

---
---
