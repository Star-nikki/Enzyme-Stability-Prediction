# Enzyme-Stability-Prediction

## Project Description

This repository presents a machine learning approach to **predict enzyme stability**, a crucial property in fields like biotechnology, industrial catalysis, and drug design. Predicting stability, often measured by the melting temperature ($T_m$), helps in engineering novel enzymes with enhanced thermal resilience for various applications.

The core of this project involves utilizing the power of a **Convolutional Neural Network (CNN)** to learn patterns directly from enzyme sequence data to predict their stability.

---

## Key Features

* **Sequence-to-Stability Prediction:** Develops a model to map the primary amino acid sequence of an enzyme to a predicted stability value (e.g., $T_m$).
* **Deep Learning Implementation:** Implements a **Convolutional Neural Network (CNN)**, which is highly effective at extracting local and global features from sequence data.
* **Data Preprocessing:** Handles the necessary steps for encoding enzyme sequences into a format suitable for deep learning models.
* **Model Training and Evaluation:** Demonstrates the training process, performance metrics, and visualization of results.

---

## Technologies and Libraries

The project is implemented entirely within a Jupyter Notebook environment using Python's leading data science and deep learning libraries:

* **Language:** Python
* **Environment:** Jupyter Notebook
* **Deep Learning:** TensorFlow / Keras (for building and training the CNN model)
* **Data Handling:** Pandas, NumPy
* **Other Potential Libraries:** Scikit-learn (for utilities), Matplotlib/Seaborn (for visualization)

---

## Files in Repository

* **`Enzyme CNN.ipynb`**: The primary Jupyter Notebook containing all the code for data loading, preprocessing, CNN architecture definition, training, and evaluation.
* **`README.md`**: This file.

---

## Setup and Installation

To reproduce the analysis and run the notebook locally, follow these steps:

### Prerequisites

You need **Python (3.7+)** installed, along with the **Jupyter Notebook** environment.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Star-nikki/Enzyme-Stability-Prediction.git](https://github.com/Star-nikki/Enzyme-Stability-Prediction.git)
    cd Enzyme-Stability-Prediction
    ```

2.  **Create and activate a virtual environment (Recommended):**
    ```bash
    conda create --name enzyme_env python=3.9
    conda activate enzyme_env
    ```
    *or using venv:*
    ```bash
    python -m venv enzyme_env
    source enzyme_env/bin/activate  # On Windows, use `enzyme_env\Scripts\activate`
    ```

3.  **Install dependencies:**
    The most crucial dependencies for this project are `tensorflow` (for Keras/CNN), `pandas`, and `numpy`.
    ```bash
    pip install pandas numpy jupyter
    # Install the deep learning framework
    pip install tensorflow
    ```

---

## 🚀 Usage

1.  **Launch Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```

2.  **Open the notebook:**
    In your browser, click on **`Enzyme CNN.ipynb`**.

3.  **Run the analysis:**
    Execute the cells sequentially to load the data, process the enzyme sequences, define and train the CNN model, and view the final prediction results.

---


## 📞 Contact

* **GitHub Profile:** [Star-nikki](https://github.com/Star-nikki)
* **Project Link:** [https://github.com/Star-nikki/Enzyme-Stability-Prediction](https://github.com/Star-nikki/Enzyme-Stability-Prediction)
