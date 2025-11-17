#!/usr/bin/env python3

import os, sys, glob, time, math, random, csv
import tracemalloc, psutil
from statistics import mean, stdev
import numpy as np
import matplotlib.pyplot as plt

# Optional GPU (cupy) for bitonic
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except Exception:
    cp = None
    CUPY_AVAILABLE = False

PROCESS = psutil.Process()

# ========================================================================================
# DATASET CONFIGURATION - EASILY EDITABLE SECTION
# ========================================================================================

def get_dataset_configurations():
    """
    Configure all dataset folders here.
    Returns a dictionary with dataset type as key and folder path as value.
    
    To add/remove datasets:
    1. Add/remove entries in the dictionary below
    2. Each key is a descriptive name for the dataset type
    3. Each value is the full path to the folder containing .txt files
    """
    datasets = {
        "duplicate_heavy": r"C:\Users\ashis\OneDrive\Documents\python face detection\duplicate heavy dataset",
        "nearly_sorted": r"C:\Users\ashis\OneDrive\Documents\python face detection\nearly sorted",
        "reversed": r"C:\Users\ashis\OneDrive\Documents\python face detection\reversed dataset",
        "uniform": r"C:\Users\ashis\OneDrive\Documents\python face detection\uniform dataset"
    }
    return datasets

def get_files_per_dataset_type():
    """
    Configure how many files to process per dataset type.
    Change this number to process more/fewer files from each folder.
    """
    return 4  # Process up to 4 files from each dataset folder

# ========================================================================================
# DATASET DISCOVERY AND LOADING
# ========================================================================================

def find_dataset_files_by_type(dataset_type, folder_path, limit=4):
    """
    Find dataset files for a specific dataset type.
    
    Args:
        dataset_type: Name/description of the dataset type
        folder_path: Path to the folder containing dataset files
        limit: Maximum number of files to return
    
    Returns:
        List of file paths
    """
    if not os.path.exists(folder_path):
        print(f"Warning: Dataset folder '{dataset_type}' not found at: {folder_path}")
        return []
    
    search_pattern = os.path.join(folder_path, "*.txt")
    files = glob.glob(search_pattern)
    
    if not files:
        print(f"Warning: No .txt files found in '{dataset_type}' folder: {folder_path}")
        return []
    
    files.sort()
    selected_files = files[:limit]
    print(f"Found {len(selected_files)} files in '{dataset_type}' dataset")
    return selected_files

def discover_all_datasets():
    """
    Discover all dataset files from configured folders.
    
    Returns:
        Dictionary mapping dataset_type to list of file paths
        Example: {'duplicate_heavy': ['file1.txt', 'file2.txt'], ...}
    """
    dataset_configs = get_dataset_configurations()
    files_per_type = get_files_per_dataset_type()
    
    all_datasets = {}
    
    print("\n" + "="*80)
    print("DISCOVERING DATASETS")
    print("="*80)
    
    for dataset_type, folder_path in dataset_configs.items():
        files = find_dataset_files_by_type(dataset_type, folder_path, limit=files_per_type)
        if files:
            all_datasets[dataset_type] = files
    
    # Print summary
    total_files = sum(len(files) for files in all_datasets.values())
    print(f"\nTotal: {total_files} files across {len(all_datasets)} dataset types")
    print("="*80 + "\n")
    
    return all_datasets

def read_numbers_from_file(path):
    """Read integers from a text file."""
    nums = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            for p in parts:
                try:
                    nums.append(int(p))
                except ValueError:
                    pass
    return nums

# ========================================================================================
# SORTING ALGORITHMS (keeping your existing implementations)
# ========================================================================================

def insertion_sort(arr, left=0, right=None):
    """Optimized insertion sort for small arrays"""
    if right is None:
        right = len(arr) - 1
    for i in range(left + 1, right + 1):
        key = arr[i]
        j = i - 1
        while j >= left and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr

def quick_sort(arr):
    """Optimized iterative quicksort with better pivot selection"""
    if len(arr) <= 1:
        return arr
    
    if len(arr) <= 20:
        return insertion_sort(arr)
    
    stack = [(0, len(arr)-1)]
    while stack:
        low, high = stack.pop()
        
        if high - low <= 20:
            insertion_sort(arr, low, high)
            continue
            
        mid = (low + high) // 2
        if arr[mid] < arr[low]:
            arr[low], arr[mid] = arr[mid], arr[low]
        if arr[high] < arr[low]:
            arr[low], arr[high] = arr[high], arr[low]
        if arr[high] < arr[mid]:
            arr[mid], arr[high] = arr[high], arr[mid]
        
        pivot = arr[mid]
        arr[mid], arr[high-1] = arr[high-1], arr[mid]
        
        i, j = low, high-1
        while True:
            i += 1
            j -= 1
            while arr[i] < pivot: 
                i += 1
            while arr[j] > pivot: 
                j -= 1
            if i >= j: 
                break
            arr[i], arr[j] = arr[j], arr[i]
        
        arr[i], arr[high-1] = arr[high-1], arr[i]
        
        if i - low < high - i:
            stack.append((low, i-1))
            stack.append((i+1, high))
        else:
            stack.append((i+1, high))
            stack.append((low, i-1))
    return arr

def merge_sort(arr):
    """Simple recursive mergesort"""
    if len(arr) <= 1:
        return arr
        
    if len(arr) <= 64:
        arr.sort()
        return arr
    
    mid = len(arr) // 2
    left = arr[:mid]
    right = arr[mid:]
    
    merge_sort(left)
    merge_sort(right)
    
    i = j = k = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            arr[k] = left[i]
            i += 1
        else:
            arr[k] = right[j]
            j += 1
        k += 1
    
    while i < len(left):
        arr[k] = left[i]
        i += 1
        k += 1
        
    while j < len(right):
        arr[k] = right[j]
        j += 1
        k += 1
    
    return arr

def heap_sort(arr):
    """
    Efficient heap sort inspired by Baagyere et al. 2025:
    - Uses a highly optimized sift-down based heapify.
    - Minimizes swaps and compares.
    - Designed for large datasets processing efficiency.
    """

    def sift_down(a, start, end):
        root = start
        while True:
            child = 2 * root + 1
            if child > end:
                break
            # Choose the larger child
            if child + 1 <= end and a[child + 1] > a[child]:
                child += 1

            # Swap if root is less than larger child
            if a[root] < a[child]:
                a[root], a[child] = a[child], a[root]
                root = child
            else:
                break

    n = len(arr)
    # Build heap
    for start in range(n // 2 - 1, -1, -1):
        sift_down(arr, start, n - 1)

    # Extract elements from heap
    for end in range(n - 1, 0, -1):
        arr[0], arr[end] = arr[end], arr[0]
        sift_down(arr, 0, end - 1)

    return arr


def radix_sort(arr):
    """LSD Radix Sort for non-negative integers"""
    if not arr: 
        return arr
    if any(x < 0 for x in arr):
        raise ValueError("RadixSort requires non-negative integers")
        
    maxv = max(arr)
    exp = 1
    n = len(arr)
    out = [0]*n
    while maxv // exp > 0:
        count = [0]*10
        for x in arr:
            count[(x // exp) % 10] += 1
        for i in range(1, 10):
            count[i] += count[i-1]
        for i in range(n-1, -1, -1):
            d = (arr[i] // exp) % 10
            count[d] -= 1
            out[count[d]] = arr[i]
        arr[:] = out[:]
        exp *= 10
    return arr

def bucket_sort(arr):
    if not arr: 
        return arr
    if any(x < 0 for x in arr):
        raise ValueError("BucketSort requires non-negative integers")
        
    n = len(arr)
    minv, maxv = min(arr), max(arr)
    if maxv == minv:
        return arr
    k = int(math.sqrt(n)) or 1
    buckets = [[] for _ in range(k)]
    width = (maxv - minv + 1) / k
    for x in arr:
        idx = int((x - minv) / width)
        if idx == k: 
            idx = k-1
        buckets[idx].append(x)
    out = []
    for b in buckets:
        out.extend(sorted(b))
    arr[:] = out
    return arr

def cluster_sort(arr, clusters=None):
    if not arr: 
        return arr
    n = len(arr)
    k = clusters or max(2, int(math.sqrt(n)))
    minv, maxv = min(arr), max(arr)
    if maxv == minv:
        return arr
    buckets = [[] for _ in range(k)]
    width = (maxv - minv + 1) / k
    for x in arr:
        idx = int((x - minv) / width)
        if idx == k: 
            idx = k-1
        buckets[idx].append(x)
    out = []
    for b in buckets:
        if len(b) < 32:
            b.sort()
        else:
            b = quick_sort(b)
        out.extend(b)
    arr[:] = out
    return arr

def tim_sort(arr):
    arr[:] = sorted(arr)
    return arr

def burst_sort(arr):
    if not arr: 
        return arr
    if any(x < 0 for x in arr):
        raise ValueError("BurstSort requires non-negative integers")
        
    maxlen = len(str(max(arr)))
    def msd_radix(a, pos):
        if pos >= maxlen or len(a) <= 1:
            return a
        buckets = [[] for _ in range(10)]
        for x in a:
            s = str(x).zfill(maxlen)
            digit = ord(s[pos]) - ord('0')
            buckets[digit].append(x)
        out = []
        for b in buckets:
            if b:
                out.extend(msd_radix(b, pos+1))
        return out
    arr[:] = msd_radix(arr, 0)
    return arr

def spread_sort(arr):
    if not arr: 
        return arr
    mn = min(arr)
    if mn < 0:
        offset = -mn
        arr2 = [x + offset for x in arr]
    else:
        offset = 0
        arr2 = arr[:]
    maxv = max(arr2)
    if maxv == 0:
        arr[:] = arr2 if offset==0 else [x-offset for x in arr2]
        return arr
    msb = maxv.bit_length() - 1
    def msd_bit(a, bit):
        if bit < 0 or len(a) <= 1:
            return a
        zero = []; one = []
        mask = 1 << bit
        for x in a:
            if x & mask:
                one.append(x)
            else:
                zero.append(x)
        return msd_bit(zero, bit-1) + msd_bit(one, bit-1)
    out = msd_bit(arr2, msb)
    if offset:
        out = [x - offset for x in out]
    arr[:] = out
    return arr

def counting_sort(arr):
    if not arr: 
        return arr
    if any(x < 0 for x in arr):
        raise ValueError("CountingSort requires non-negative integers")
        
    maxv = max(arr)
    cnt = [0]*(maxv+1)
    for x in arr: 
        cnt[x] += 1
    out = []
    for val, c in enumerate(cnt):
        if c:
            out.extend([val]*c)
    arr[:] = out
    return arr

def optiflex_sort(arr):
    """
    OptiFlexSort: Hybrid sorting algorithm based on the methodology described in
    'OptiFlexSort A Hybrid Sorting Algorithm for Efficient Large-Scale Data Processing'.
    - Enhanced pivot selection (median-of-three).
    - Adaptive partitioning.
    - Fast exit for small arrays.
    """

    def median_of_three(a, i, j, k):
        trio = [(a[i], i), (a[j], j), (a[k], k)]
        trio.sort(key=lambda x: x[0])
        return trio[1][1]

    def partition(a, low, high):
        """Adaptive partitioning as described in the paper."""
        # Select pivot index using median-of-three between first, middle, last
        mid = (low + high) // 2
        m_idx = median_of_three(a, low, mid, high)
        a[m_idx], a[high] = a[high], a[m_idx]
        pivot = a[high]
        
        # Adaptive split: maintains tracking of min/max as proposed
        left = low
        right = high - 1

        # Tracking min/max in partition zones (the paper's innovation)
        min_left = a[low] if high > low else pivot
        max_right = a[high - 1] if high > low else pivot

        while left <= right:
            while left <= right and a[left] < pivot:
                min_left = min(min_left, a[left])
                left += 1
            while left <= right and a[right] > pivot:
                max_right = max(max_right, a[right])
                right -= 1
            if left <= right:
                a[left], a[right] = a[right], a[left]
                left += 1
                right -= 1

        # Final swap to put pivot into position
        a[left], a[high] = a[high], a[left]

        # Return pivot boundaries and min/max values for potential future optimizations
        return left, min_left, max_right

    def _rec(a, low, high):
        # Paper: immediate return for subarrays < 2
        if high - low < 1:
            return
        # Optional enhancement: insertion sort for very small arrays (<= 32)
        if high - low + 1 <= 32:
            insertion_sort(a, low, high)
            return
        # Partition and recursively sort subarrays
        pivot_pos, min_left, max_right = partition(a, low, high)
        # Recursively sort partitions
        _rec(a, low, pivot_pos - 1)
        _rec(a, pivot_pos + 1, high)

    _rec(arr, 0, len(arr) - 1)
    return arr

# You need to already have an 'insertion_sort(arr, left, right)' helper defined as in your code.


# Adaptive Hybrid Sort components
def insertion_sort_adaptive(arr, left, right):
    """Adaptive insertion sort with early termination"""
    for i in range(left + 1, right + 1):
        key = arr[i]
        j = i - 1
        
        if arr[j] <= key:
            continue
            
        while j >= left and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr

def median_of_three(arr, low, high):
    """Return index of median of first, middle, last"""
    mid = (low + high) // 2
    
    if arr[low] > arr[mid]:
        if arr[mid] > arr[high]:
            return mid
        elif arr[low] > arr[high]:
            return high
        else:
            return low
    else:
        if arr[low] > arr[high]:
            return low
        elif arr[mid] > arr[high]:
            return high
        else:
            return mid

def adaptive_quicksort(arr):
    """Quicksort that adapts based on partition characteristics"""
    stack = [(0, len(arr)-1)]
    
    while stack:
        low, high = stack.pop()
        size = high - low + 1
        
        if size <= 32:
            insertion_sort_adaptive(arr, low, high)
            continue
        
        if size <= 100:
            pivot_idx = median_of_three(arr, low, high)
        else:
            pivot_idx = random.randint(low, high)
        
        arr[pivot_idx], arr[high] = arr[high], arr[pivot_idx]
        pivot = arr[high]
        
        i = low
        for j in range(low, high):
            if arr[j] <= pivot:
                arr[i], arr[j] = arr[j], arr[i]
                i += 1
        
        arr[i], arr[high] = arr[high], arr[i]
        
        if i - low < high - i:
            stack.append((low, i-1))
            stack.append((i+1, high))
        else:
            stack.append((i+1, high))
            stack.append((low, i-1))
    
    return arr

def timsort_inspired(arr):
    """Timsort-inspired approach for mostly sorted data"""
    n = len(arr)
    min_run = 32
    
    i = 0
    runs = []
    
    while i < n:
        j = i
        if j < n - 1:
            if arr[j] <= arr[j + 1]:
                while j < n - 1 and arr[j] <= arr[j + 1]:
                    j += 1
            else:
                while j < n - 1 and arr[j] > arr[j + 1]:
                    j += 1
                arr[i:j+1] = arr[i:j+1][::-1]
        
        run_length = j - i + 1
        
        if run_length < min_run:
            j = min(i + min_run - 1, n - 1)
            insertion_sort_adaptive(arr, i, j)
            run_length = j - i + 1
        
        runs.append((i, run_length))
        i += run_length
    
    while len(runs) > 1:
        new_runs = []
        for i in range(0, len(runs), 2):
            if i + 1 < len(runs):
                left_start, left_len = runs[i]
                right_start, right_len = runs[i + 1]
                merge_runs(arr, left_start, left_start + left_len - 1, 
                          right_start + right_len - 1)
                new_runs.append((left_start, left_len + right_len))
            else:
                new_runs.append(runs[i])
        runs = new_runs
    
    return arr

def merge_runs(arr, left, mid, right):
    """Merge two sorted runs"""
    left_copy = arr[left:mid + 1]
    right_copy = arr[mid + 1:right + 1]
    
    i = j = 0
    k = left
    
    while i < len(left_copy) and j < len(right_copy):
        if left_copy[i] <= right_copy[j]:
            arr[k] = left_copy[i]
            i += 1
        else:
            arr[k] = right_copy[j]
            j += 1
        k += 1
    
    while i < len(left_copy):
        arr[k] = left_copy[i]
        i += 1
        k += 1
    
    while j < len(right_copy):
        arr[k] = right_copy[j]
        j += 1
        k += 1

def counting_sort_adaptive(arr):
    """Adaptive counting sort"""
    if not arr:
        return arr
    
    min_val, max_val = min(arr), max(arr)
    range_size = max_val - min_val + 1
    
    if range_size <= len(arr) * 10:
        count = [0] * range_size
        for x in arr:
            count[x - min_val] += 1
        
        output = []
        for i in range(range_size):
            output.extend([i + min_val] * count[i])
        
        arr[:] = output
        return arr
    else:
        return adaptive_quicksort(arr)

def radix_sort_adaptive(arr):
    """Adaptive radix sort with base selection"""
    if not arr:
        return arr
    
    max_val = max(arr)
    if max_val == 0:
        return arr
    
    if max_val < 256:
        base = 256
    elif max_val < 65536:
        base = 65536
    else:
        return adaptive_quicksort(arr)
    
    n = len(arr)
    output = [0] * n
    exp = 1
    
    while max_val // exp > 0:
        count = [0] * base
        
        for x in arr:
            count[(x // exp) % base] += 1
        
        for i in range(1, base):
            count[i] += count[i - 1]
        
        for i in range(n - 1, -1, -1):
            digit = (arr[i] // exp) % base
            count[digit] -= 1
            output[count[digit]] = arr[i]
        
        arr[:] = output[:]
        exp *= base
    
    return arr

def adaptive_hybrid_sort(arr):
    """True Adaptive Hybrid Sort"""
    if len(arr) <= 1:
        return arr
    
    n = len(arr)
    
    min_val, max_val = min(arr), max(arr)
    range_size = max_val - min_val
    is_non_negative = all(x >= 0 for x in arr)
    
    sample_size = min(100, n // 100)
    if sample_size > 1:
        sample_indices = random.sample(range(n), sample_size)
        sample = [arr[i] for i in sample_indices]
        inversions = sum(1 for i in range(1, len(sample)) if sample[i] < sample[i-1])
        is_mostly_sorted = (inversions / len(sample)) < 0.1
    else:
        is_mostly_sorted = False
    
    if n <= 32:
        return insertion_sort_adaptive(arr, 0, n-1)
    
    if is_mostly_sorted:
        return timsort_inspired(arr)
    
    if is_non_negative and range_size <= n * 10:
        return counting_sort_adaptive(arr)
    elif is_non_negative and range_size <= n * 1000:
        return radix_sort_adaptive(arr)
    
    return adaptive_quicksort(arr)

# GPU algorithms
if CUPY_AVAILABLE:
    bitonic_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void bitonic_sort(int *data, int n, int j, int k) {
        unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
        if (i >= n) return;
        unsigned int ixj = i ^ j;
        if (ixj > i) {
            bool ascending = ((i & k) == 0);
            int a = data[i];
            int b = data[ixj];
            if ((a > b) == ascending) {
                data[i] = b;
                data[ixj] = a;
            }
        }
    }
    ''', 'bitonic_sort')

    def gpu_bitonic_sort(cp_arr):
        N = int(cp_arr.size)
        threads_per_block = 512
        blocks = (N + threads_per_block - 1)//threads_per_block
        k = 2
        while k <= N:
            j = k//2
            while j > 0:
                bitonic_kernel((blocks,), (threads_per_block,), (cp_arr, N, j, k))
                j //= 2
            k *= 2
        cp.cuda.Stream.null.synchronize()
        return cp_arr

    sample_sort_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void sample_sort(int *data, int *pivots, int *output, int *bucket_offsets, int *bucket_counts, 
                    int n, int num_pivots, int num_buckets) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= n) return;
        
        int value = data[tid];
        int bucket_id = num_buckets - 1;
        
        for (int i = 0; i < num_pivots; i++) {
            if (value <= pivots[i]) {
                bucket_id = i;
                break;
            }
        }
        
        int pos = bucket_offsets[bucket_id] + atomicAdd(&bucket_counts[bucket_id], 1);
        output[pos] = value;
    }
    ''', 'sample_sort')

    def gpu_sample_sort(cp_arr, num_buckets=8):
        """Simple GPU Sample Sort using custom kernel"""
        N = int(cp_arr.size)
        threads_per_block = 256
        blocks = (N + threads_per_block - 1) // threads_per_block
        
        sample_size = min(1000, N // 10)
        sample_indices = cp.random.randint(0, N, size=sample_size, dtype=cp.int32)
        samples = cp.sort(cp_arr[sample_indices])
        
        pivot_indices = cp.linspace(0, sample_size - 1, num_buckets, dtype=cp.int32)[1:]
        pivots = samples[pivot_indices]
        
        bucket_counts = cp.zeros(num_buckets, dtype=cp.int32)
        for i in range(num_buckets):
            if i == 0:
                count = cp.sum(cp_arr <= pivots[0])
            elif i == num_buckets - 1:
                count = cp.sum(cp_arr > pivots[-1])
            else:
                count = cp.sum((cp_arr > pivots[i-1]) & (cp_arr <= pivots[i]))
            bucket_counts[i] = count
        
        bucket_offsets = cp.zeros(num_buckets, dtype=cp.int32)
        bucket_offsets[1:] = cp.cumsum(bucket_counts)[:-1]
        
        output = cp.empty_like(cp_arr)
        temp_counts = cp.zeros(num_buckets, dtype=cp.int32)
        
        sample_sort_kernel((blocks,), (threads_per_block,),
                         (cp_arr, pivots, output, bucket_offsets, temp_counts, 
                          N, pivots.size, num_buckets))
        cp.cuda.Stream.null.synchronize()
        
        for i in range(num_buckets):
            start = int(bucket_offsets[i])
            end = start + int(bucket_counts[i])
            if end > start:
                output[start:end] = cp.sort(output[start:end])
        
        return output

# ========================================================================================
# MEASUREMENT FUNCTIONS
# ========================================================================================

def measure_cpu_sort(sort_fn, arr, repeats=3):
    """Optimized measurement without memory overhead during timing"""
    times = []
    
    import gc
    gc.collect()
    
    original_arr = arr.copy()
    
    for _ in range(repeats):
        a = arr.copy()
        
        t0 = time.perf_counter()
        sort_fn(a)
        t1 = time.perf_counter()
        
        times.append(t1 - t0)
    
    gc.collect()
    a_test = arr.copy()
    
    process = psutil.Process()
    mem_before = process.memory_info().rss
    
    tracemalloc.start()
    sort_fn(a_test)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    mem_after = process.memory_info().rss
    
    expected = sorted(original_arr)
    if a_test != expected:
        print(f"WARNING: Sort verification failed for {sort_fn.__name__}")
    
    return mean(times), (stdev(times) if len(times) > 1 else 0.0), peak / 1024.0, mem_before / (1024**2), mem_after / (1024**2)

def measure_gpu_bitonic(cp_arr):
    free_before, total = cp.cuda.runtime.memGetInfo()
    used_before = (total - free_before) / (1024**2)
    t0 = time.perf_counter()
    out = gpu_bitonic_sort(cp_arr)
    cp.cuda.Stream.null.synchronize()
    t1 = time.perf_counter()
    free_after, total = cp.cuda.runtime.memGetInfo()
    used_after = (total - free_after) / (1024**2)
    return (t1 - t0), used_before, used_after, out

def measure_gpu_sample_sort(sort_fn, cp_arr):
    """Measure GPU Sample Sort performance"""
    free_before, total = cp.cuda.runtime.memGetInfo()
    used_before = (total - free_before) / (1024**2)
    t0 = time.perf_counter()
    out = sort_fn(cp_arr)
    cp.cuda.Stream.null.synchronize()
    t1 = time.perf_counter()
    free_after, total = cp.cuda.runtime.memGetInfo()
    used_after = (total - free_after) / (1024**2)
    return (t1 - t0), used_before, used_after, out

# ========================================================================================
# ALGORITHM REGISTRY
# ========================================================================================

def get_algorithm_registry():
    """
    Returns list of all algorithms to benchmark.
    Format: (name, function, type)
    
    To add/remove algorithms:
    - Add/remove entries from this list
    - Set type to "cpu" for CPU algorithms or "gpu" for GPU algorithms
    """
    algorithms = [
        ("QuickSort", quick_sort, "cpu"),
        ("MergeSort", merge_sort, "cpu"),
        ("HeapSort", heap_sort, "cpu"),
        ("RadixSort", radix_sort, "cpu"),
        ("BucketSort", bucket_sort, "cpu"),
        ("CountingSort", counting_sort, "cpu"),
        ("OptiFlexSort", optiflex_sort, "cpu"),
        ("ClusterSort", cluster_sort, "cpu"),
        ("TimSort (Python)", tim_sort, "cpu"),
        ("BurstSort", burst_sort, "cpu"),
        ("SpreadSort", spread_sort, "cpu"),
        ("AdaptiveHybridSort", adaptive_hybrid_sort, "cpu"),
    ]
    
    if CUPY_AVAILABLE:
        algorithms.append(("GPU_Bitonic", None, "gpu"))
        algorithms.append(("GPU_SampleSort", gpu_sample_sort, "gpu"))
    
    return algorithms

# ========================================================================================
# BENCHMARKING EXECUTION
# ========================================================================================

def pad_to_pow2(lst, pad_value=(2**31-1)):
    n = len(lst)
    if n == 0: 
        return lst, 0
    next_pow2 = 1 << (n-1).bit_length()
    if next_pow2 == n: 
        return lst, n
    return lst + [pad_value]*(next_pow2 - n), next_pow2

def run_benchmarks_on_file(path, dataset_type, csv_writer, repeats=3):
    """
    Run benchmarks on a single file.
    
    Args:
        path: Path to the dataset file
        dataset_type: Type/category of the dataset
        csv_writer: CSV writer object
        repeats: Number of times to repeat each sort
    """
    arr = read_numbers_from_file(path)
    if not arr:
        print(f"{path}: empty or no ints -> skip")
        return None
    
    n = len(arr)
    filename = os.path.basename(path)
    print(f"\n--- Dataset: {filename} ({dataset_type}, n={n}) ---")
    
    meta = {}
    meta['_meta_n'] = n
    meta['_meta_type'] = dataset_type
    
    algorithms = get_algorithm_registry()
    
    for name, fn, kind in algorithms:
        if kind == "gpu":
            if not CUPY_AVAILABLE:
                print(f"  -> {name}: skipped (cupy not available)")
                csv_writer.writerow([dataset_type, path, name, "skipped", "cupy_unavailable", "", "", "", ""])
                meta[name] = {"time": float('nan'), "mem_kb": float('nan'), "std": float('nan')}
                continue
            
            arr_pos = [int(x) & 0x7fffffff for x in arr]
            
            if name == "GPU_Bitonic":
                padded, newn = pad_to_pow2(arr_pos)
                cp_arr = cp.array(padded, dtype=cp.int32)
                try:
                    t_gpu, v_before, v_after, out = measure_gpu_bitonic(cp_arr)
                    vram_used_mb = max(0.0, v_after - v_before)
                    print(f"  -> {name}: time={t_gpu:.6f}s, vram_used~{vram_used_mb:.2f} MB (padded->{newn})")
                    csv_writer.writerow([dataset_type, path, name, f"{t_gpu:.6f}", "", f"{vram_used_mb:.2f}", f"{v_before:.2f}", f"{v_after:.2f}", repeats])
                    meta[name] = {"time": t_gpu, "mem_kb": vram_used_mb*1024.0, "std": 0.0}
                except Exception as e:
                    print(f"  -> {name}: FAILED: {e}")
                    csv_writer.writerow([dataset_type, path, name, "FAILED", str(e), "", "", "", ""])
                    meta[name] = {"time": float('nan'), "mem_kb": float('nan'), "std": float('nan')}
            elif name == "GPU_SampleSort":
                cp_arr = cp.array(arr_pos, dtype=cp.int32)
                try:
                    t_gpu, v_before, v_after, out = measure_gpu_sample_sort(fn, cp_arr)
                    vram_used_mb = max(0.0, v_after - v_before)
                    print(f"  -> {name}: time={t_gpu:.6f}s, vram_used~{vram_used_mb:.2f} MB")
                    csv_writer.writerow([dataset_type, path, name, f"{t_gpu:.6f}", "", f"{vram_used_mb:.2f}", f"{v_before:.2f}", f"{v_after:.2f}", repeats])
                    meta[name] = {"time": t_gpu, "mem_kb": vram_used_mb*1024.0, "std": 0.0}
                except Exception as e:
                    print(f"  -> {name}: FAILED: {e}")
                    csv_writer.writerow([dataset_type, path, name, "FAILED", str(e), "", "", "", ""])
                    meta[name] = {"time": float('nan'), "mem_kb": float('nan'), "std": float('nan')}
            continue

        # Check non-negative requirement
        if name in ("RadixSort", "CountingSort", "BucketSort", "BurstSort") and any(x < 0 for x in arr):
            print(f"  -> {name}: SKIPPED (contains negative numbers)")
            csv_writer.writerow([dataset_type, path, name, "skipped", "negatives_in_input", "", "", "", ""])
            meta[name] = {"time": float('nan'), "mem_kb": float('nan'), "std": float('nan')}
            continue

        # Measure CPU algorithm
        try:
            avg_t, t_std, avg_peak_kb, rss_b, rss_a = measure_cpu_sort(fn, arr, repeats=repeats)
            rss_used = max(0.0, rss_a - rss_b)
            print(f"  -> {name} ... time={avg_t:.6f}s (±{t_std:.6f}), peak={avg_peak_kb:.1f} KB, rss_used~{rss_used:.2f} MB")
            csv_writer.writerow([dataset_type, path, name, f"{avg_t:.6f}", f"{t_std:.6f}", f"{avg_peak_kb:.2f}", f"{rss_b:.2f}", f"{rss_a:.2f}", repeats])
            meta[name] = {"time": avg_t, "mem_kb": avg_peak_kb, "std": t_std}
        except Exception as e:
            print(f"  -> {name}: FAILED: {e}")
            csv_writer.writerow([dataset_type, path, name, "FAILED", str(e), "", "", "", ""])
            meta[name] = {"time": float('nan'), "mem_kb": float('nan'), "std": float('nan')}
    
    return meta

def run_all_benchmarks(all_datasets, csv_writer, repeats=5):
    """
    Run benchmarks on all discovered datasets.
    
    Args:
        all_datasets: Dictionary mapping dataset_type to list of file paths
        csv_writer: CSV writer object
        repeats: Number of times to repeat each sort
        
    Returns:
        Summary dictionary with results
    """
    summary = {}
    
    for dataset_type, files in all_datasets.items():
        print(f"\n{'='*80}")
        print(f"PROCESSING DATASET TYPE: {dataset_type.upper()}")
        print(f"{'='*80}")
        
        for filepath in files:
            meta = run_benchmarks_on_file(filepath, dataset_type, csv_writer, repeats=repeats)
            if meta:
                # Use composite key: (dataset_type, filepath)
                summary[(dataset_type, filepath)] = meta
    
    return summary

# ========================================================================================
# PLOTTING FUNCTIONS
# ========================================================================================

def plot_results_by_dataset_type(summary, out_prefix='combined'):
    """
    Create comprehensive plots showing performance across dataset types.
    
    Args:
        summary: Dictionary with results
        out_prefix: Prefix for output filenames
    """
    # Organize data by dataset type
    dataset_types = {}
    for (ds_type, filepath), metrics in summary.items():
        if ds_type not in dataset_types:
            dataset_types[ds_type] = []
        dataset_types[ds_type].append((filepath, metrics))
    
    # Get all algorithm names
    sample_key = list(summary.keys())[0]
    algs = [alg for alg in summary[sample_key].keys() if not alg.startswith('_meta')]
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(algs)))
    if len(algs) > 20:
        extra_colors = plt.cm.Set3(np.linspace(0, 1, len(algs) - 20))
        colors = np.vstack([colors, extra_colors])
    
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'd', '|', '_']
    
    # Plot 1: Runtime comparison across dataset types
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('Sorting Performance Across Different Dataset Types', fontsize=16, fontweight='bold')
    
    for idx, (ds_type, data) in enumerate(dataset_types.items()):
        if idx >= 4:
            break
        
        ax = axes[idx // 2, idx % 2]
        
        # Extract sizes and times for this dataset type
        sizes = [metrics['_meta_n'] for _, metrics in data]
        
        for i, alg in enumerate(algs):
            times = [metrics[alg]['time'] for _, metrics in data]
            stds = [metrics[alg]['std'] for _, metrics in data]
            
            valid_indices = [j for j, t in enumerate(times) if not math.isnan(t)]
            if valid_indices:
                valid_times = [times[j] for j in valid_indices]
                valid_sizes = [sizes[j] for j in valid_indices]
                valid_stds = [stds[j] for j in valid_indices]
                
                marker = markers[i % len(markers)]
                ax.errorbar(valid_sizes, valid_times, yerr=valid_stds,
                           marker=marker, markersize=6, markeredgewidth=1,
                           color=colors[i], label=alg,
                           capsize=3, capthick=1, elinewidth=1,
                           linestyle='-', linewidth=1.5, alpha=0.7)
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Input Size (log scale)', fontsize=10, fontweight='bold')
        ax.set_ylabel('Time (seconds, log scale)', fontsize=10, fontweight='bold')
        ax.set_title(f'{ds_type.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        ax.grid(True, which="both", ls="--", alpha=0.3)
        ax.legend(fontsize=7, ncol=2, loc='upper left', framealpha=0.8)
    
    plt.tight_layout()
    plt.savefig(f'{out_prefix}_runtime_by_type.png', dpi=300, bbox_inches='tight')
    print(f"Saved {out_prefix}_runtime_by_type.png")
    
    # Plot 2: Memory comparison
    plt.figure(figsize=(18, 10))
    
    dataset_labels = []
    for ds_type, data in dataset_types.items():
        for filepath, _ in data:
            dataset_labels.append(f"{ds_type}_{os.path.basename(filepath)}")
    
    idx = range(len(dataset_labels))
    bar_width = 0.8 / len(algs)
    
    for i, alg in enumerate(algs):
        mems = []
        for ds_type, data in dataset_types.items():
            for _, metrics in data:
                mem = metrics[alg]['mem_kb']
                mems.append(0 if math.isnan(mem) else mem)
        
        pos = [x + i * bar_width - (len(algs) - 1) * bar_width / 2 for x in idx]
        
        plt.bar(pos, mems, width=bar_width * 0.9,
               label=alg, color=colors[i], alpha=0.8,
               edgecolor='black', linewidth=0.5)
    
    plt.xlabel('Dataset (Type_File)', fontsize=12, fontweight='bold')
    plt.ylabel('Peak Memory Usage (KB)', fontsize=12, fontweight='bold')
    plt.title('Memory Consumption Across Dataset Types', fontsize=14, fontweight='bold')
    plt.xticks(idx, dataset_labels, rotation=45, ha='right', fontsize=8)
    plt.legend(fontsize=9, ncol=3, loc='upper left', framealpha=0.9)
    plt.grid(True, axis='y', ls="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{out_prefix}_memory_by_type.png', dpi=300, bbox_inches='tight')
    print(f"Saved {out_prefix}_memory_by_type.png")
    
    # Plot 3: Algorithm comparison across dataset types (heatmap-style)
    plt.figure(figsize=(16, 10))
    
    # Create performance matrix
    perf_matrix = []
    row_labels = []
    
    for ds_type, data in dataset_types.items():
        for filepath, metrics in data:
            row = []
            for alg in algs:
                time_val = metrics[alg]['time']
                row.append(time_val if not math.isnan(time_val) else 0)
            perf_matrix.append(row)
            row_labels.append(f"{ds_type}_{os.path.basename(filepath)[:15]}")
    
    # Normalize for better visualization
    perf_array = np.array(perf_matrix)
    perf_array_log = np.log10(perf_array + 1e-10)  # Add small value to avoid log(0)
    
    im = plt.imshow(perf_array_log, cmap='YlOrRd', aspect='auto')
    plt.colorbar(im, label='Log10(Time in seconds)')
    
    plt.xticks(range(len(algs)), algs, rotation=45, ha='right')
    plt.yticks(range(len(row_labels)), row_labels, fontsize=8)
    plt.xlabel('Algorithm', fontsize=12, fontweight='bold')
    plt.ylabel('Dataset', fontsize=12, fontweight='bold')
    plt.title('Performance Heatmap: Log Time Across Algorithms and Datasets', 
             fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{out_prefix}_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"Saved {out_prefix}_heatmap.png")

# ========================================================================================
# MAIN EXECUTION
# ========================================================================================

def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("SORTING ALGORITHM BENCHMARK SUITE")
    print("="*80 + "\n")
    
    # Discover all datasets
    all_datasets = discover_all_datasets()
    
    if not all_datasets:
        print("ERROR: No datasets found. Please check your folder paths in get_dataset_configurations()")
        return
    
    # Setup output CSV
    out_csv = 'results_all_datasets.csv'
    repeats = 1  # Number of times to repeat each sort
    
    print(f"\nStarting benchmarks with {repeats} repeat(s) per algorithm...")
    print(f"Results will be saved to: {out_csv}\n")
    
    # Run all benchmarks
    summary = {}
    with open(out_csv, 'w', newline='') as cf:
        writer = csv.writer(cf)
        writer.writerow(["dataset_type", "dataset_file", "algorithm", 
                        "avg_time_s_or_status", "time_std_s", 
                        "mem_kb_or_vram_mb", "rss_or_vram_before_mb", 
                        "rss_or_vram_after_mb", "repeats_or_note"])
        
        summary = run_all_benchmarks(all_datasets, writer, repeats=repeats)
    
    print(f"\n{'='*80}")
    print(f"Results CSV written to: {out_csv}")
    print(f"{'='*80}\n")
    
    # Generate plots
    if summary:
        print("Generating plots...")
        plot_results_by_dataset_type(summary, out_prefix='benchmark')
        print("\nAll plots generated successfully!")
    
    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80 + "\n")

if __name__ == '__main__':
    main()
