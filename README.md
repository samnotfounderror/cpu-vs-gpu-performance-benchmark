# Is a GPU Always Faster Than a CPU?

A hands-on **PyTorch matrix multiplication benchmark** comparing CPU computation against an NVIDIA Tesla T4 GPU in Google Colab. This is a small learning experiment, not a comprehensive hardware ranking.

![CPU vs GPU benchmark chart](images/cpu_gpu_benchmark.png)

## Research question

Does a GPU outperform a CPU for every workload size? Does the answer change if we include time spent transferring matrices between CPU and GPU memory?

## Environment

- Platform: Google Colab (Linux)
- GPU: **NVIDIA Tesla T4**
- CPU: Colab-hosted CPU; **PyTorch reported 1 CPU computation thread** (CPU model not recorded)
- Python: **3.13.16**
- PyTorch: **2.11.0+cu130**
- Data type: 32-bit floating point (`float32`)
- Operation: `A @ B` for square matrices of size 64, 256, 1024, and 2048

## Method

Three timing scenarios were measured for the same mathematical operation:

1. **CPU compute:** Both matrices are already on the CPU.
2. **GPU compute:** Both matrices are already on the GPU.
3. **GPU + transfers:** Inputs start on the CPU, are copied to the GPU for multiplication, and the result is copied back to the CPU.

The notebook uses **3 warm-up trials**, then reports the **median of 7 timing trials** per scenario and size. It repeats operations within a trial and synchronizes CUDA operations for GPU measurements so asynchronous execution does not distort timings. The displayed CSV values are rounded from the notebook's displayed results.

## Recorded results

Lower is better. Times are milliseconds per matrix multiplication.

| Matrix size | CPU compute (ms) | GPU compute (ms) | GPU + transfers (ms) | GPU compute speedup | GPU end-to-end speedup |
|---|---:|---:|---:|---:|---:|
| 64 × 64 | **0.009** | 0.016 | 0.085 | 0.59× | 0.11× |
| 256 × 256 | 0.295 | **0.042** | 0.355 | 7.11× | 0.83× |
| 1024 × 1024 | 16.974 | **0.835** | 3.820 | 20.34× | 4.44× |
| 2048 × 2048 | 141.609 | **6.066** | 14.689 | 23.35× | 9.64× |

A speedup greater than 1× means the GPU is faster; below 1× means the CPU is faster.

## Key observations

- **64 × 64:** CPU was faster than GPU compute, even before data transfers.
- **256 × 256:** GPU compute was faster, but **CPU won overall** once GPU data transfers were included.
- **1024 × 1024 and 2048 × 2048:** GPU was faster in both compute-only and transfer-inclusive scenarios.
- **Main lesson:** The optimal device depends on workload size and data movement, not just advertised compute capability.

## Reproduce in Google Colab

1. Open [Google Colab](https://colab.research.google.com/).
2. Upload [`CPU_vs_GPU.ipynb`](CPU_vs_GPU.ipynb).
3. Set **Runtime → Change runtime type → T4 GPU** (or another available CUDA GPU).
4. Select **Runtime → Run all**.
5. Review the output table and chart. The notebook generates `cpu_gpu_benchmark.png` in the active Colab working directory.

If you are running in an environment with the required GPU configuration, install packages from `requirements.txt`. Package versions and hardware may differ from the recorded Colab run.

## Limitations

- Only **one recorded benchmark run** is included; repeated runs would help assess stability.
- The Colab CPU model was not captured and the CPU was limited to one PyTorch computation thread. Results should not be generalized to all CPUs or GPUs.
- GPU performance can vary with runtime availability, library versions, thermal/power constraints, and background load.
- Matrix multiplication is one important AI operation; it does not represent all machine-learning workloads.
- The transfer-inclusive measurement models one particular data-flow pattern; real applications may reuse data already on the GPU.

## Repository contents

- `CPU_vs_GPU.ipynb` — experiment code and captured notebook outputs
- `images/cpu_gpu_benchmark.png` — chart generated from the recorded result notebook
- `results.csv` — rounded values shown in the notebook
- `requirements.txt` — Python dependencies

## Next questions

- Does the crossover point change with different CPU thread counts?
- How repeatable are the results across several Colab sessions?
- What changes when the same matrices are multiplied repeatedly without transfers?
