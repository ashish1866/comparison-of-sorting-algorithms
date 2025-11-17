<div align="center">

# 🔥 Comparative Benchmarking of Sorting Algorithms  
### **A High-Performance Evaluation Suite for CPU & GPU Sorting**

<br>

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge)
![Status](https://img.shields.io/badge/Project-Active-brightgreen?style=for-the-badge)
![Platform](https://img.shields.io/badge/Platform-CPU%20%7C%20GPU-orange?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

<br>

A complete benchmark suite comparing **14+ sorting algorithms** across  
✔ Multiple dataset types  
✔ CPU vs GPU implementations  
✔ Time + Memory metrics  
✔ Auto-generated performance graphs  

</div>

---

# 📌 Table of Contents

1. [Overview](#overview)  
2. [Features](#features)  
3. [Project Structure](#project-structure)  
4. [Algorithms Included](#algorithms-included)  
5. [Dataset Types](#dataset-types)  
6. [Installation](#installation)  
7. [How to Run](#how-to-run)  
8. [Output Generated](#output-generated)  
9. [Screenshots](#screenshots)  
10. [Future Enhancements](#future-enhancements)  
11. [Contributing & License](#contributing--license)

---

# 🧠 Overview

This project is a full experimental framework to compare the performance of classical, modern, and hybrid sorting algorithms. It measures:

- Execution time (with repeated trials)  
- Memory usage (RSS + peak memory via `tracemalloc`)  
- GPU VRAM usage (if CuPy is available)  
- Plotting time, memory, and dataset-wise comparisons

Designed for research, academic projects, and performance studies.

---

# ⚡ Features

- 🔹 Benchmarks **14+ CPU algorithms**  
- 🔹 Benchmarks **2 GPU algorithms** (Bitonic & Sample Sort) when CuPy is installed  
- 🔹 Auto-detects dataset folders and files  
- 🔹 Multi-run averaging for robust timing  
- 🔹 Peak memory measurement and RSS snapshots  
- 🔹 VRAM monitoring for GPU runs  
- 🔹 Auto-generation of comparison plots (runtime, memory, heatmap)  
- 🔹 CSV export of results for further analysis

---

# 🔢 Algorithms Included

### ✅ **CPU Sorting Algorithms**
- QuickSort (optimized iterative)
- MergeSort (top-down)
- HeapSort (enhanced sift-down)
- Radix Sort (LSD)
- Bucket Sort
- Counting Sort
- Burst Sort (MSD-based)
- Cluster Sort (hybrid bucket + quicksort)
- Spread Sort (bitwise MSD)
- TimSort (Python built-in)
- OptiFlexSort (research-based hybrid)
- AdaptiveHybridSort (multi-strategy)

### ⚡ **GPU Algorithms** (optional)
- GPU Bitonic Sort (CUDA/CuPy)
- GPU Sample Sort (custom CUDA kernel)

---

# 📁 Project Structure

```md
sorting-benchmark/
│
├── datasets/
│   ├── duplicate_heavy/
│   │     ├── file1.txt
│   │     ├── file2.txt
│   │     └── file3.txt
│   ├── nearly_sorted/
│   │     ├── file1.txt
│   │     ├── file2.txt
│   │     └── file3.txt
│   ├── reversed/
│   │     ├── file1.txt
│   │     ├── file2.txt
│   │     └── file3.txt
│   └── uniform/
│         ├── file1.txt
│         ├── file2.txt
│         └── file3.txt
│
├── results/
│   ├── benchmark_runtime_by_type.png
│   ├── benchmark_memory_by_type.png
│   ├── benchmark_heatmap.png
│   └── results_all_datasets.csv
│
├── compare_sort.py
├── requirements.txt
└── README.md
```
