# 🔥 Sorting Algorithm Benchmark Suite  
A Comprehensive Performance Analysis of CPU & GPU Sorting Algorithms

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20GPU-green)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

---

## 📘 Overview

This project is a research-grade benchmarking suite designed to evaluate and compare multiple CPU and GPU sorting algorithms across real and synthetic datasets.

It measures:

- Execution Time  
- Memory Usage  
- Dataset Scaling  
- GPU VRAM Usage  

Outputs include CSV result logs and multiple comparison graphs.

---

## 📂 Project Structure

```
comparison-of-sorting-algorithms/
│
├── compare_sort.py                # Main benchmarking script
├── README.md                      # Project documentation
├── requirements.txt               # Dependencies
│
├── datasets/
│   ├── duplicate_heavy/
│   ├── nearly_sorted/
│   ├── reversed/
│   └── uniform/
│
└── results/
    ├── benchmark_heatmap.png
    ├── benchmark_memory_by_type.png
    ├── benchmark_runtime_by_type.png
    └── results_all_datasets.csv
```

---

## 🚀 Features

- 12+ CPU algorithms  
- GPU modes (Bitonic & Sample Sort)
- Auto dataset discovery  
- Automated benchmarking workflow  
- Auto CSV export  
- Graph generation  

---

## ⚙️ Installation

Clone the repo:

```
git clone https://github.com/<your-username>/comparison-of-sorting-algorithms
cd comparison-of-sorting-algorithms
```

Install dependencies:

```
pip install -r requirements.txt
```

GPU support (optional):

```
pip install cupy-cuda12x
```

---

## ▶️ Run Benchmark

```
python compare_sort.py
```

Generates:

- CSV results  
- Runtime plot  
- Memory usage plot  
- Performance heatmap  

---

## 📝 License
MIT License — free for research and academic use.

