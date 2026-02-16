"""
Benchmark script for Oscillatory Synchronization Network (OSN)
==============================================================
Compares SSA vs standard Transformer attention on:
  - Throughput (tokens/sec) at different sequence lengths
  - Memory usage (GPU via CUDA, CPU via tracemalloc)
  - Forward pass timing

Author: Hasi Hays (hasih@uark.edu)
"""

import time
import csv
import os
import gc
import torch
import torch.nn as nn
import numpy as np
import argparse

from osn_model import (
    SelectiveSynchronizationAttention,
    StandardTransformerAttention,
    OSNBlock,
    StandardTransformerBlock,
)


def benchmark_model(model, B, N, D, n_warmup=10, n_trials=50, device="cuda"):
    """Benchmark a model: throughput, timing, memory."""
    model = model.to(device).eval()

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            x = torch.randn(B, N, D, device=device)
            _ = model(x)

    if device == "cuda":
        torch.cuda.synchronize()

    # Memory measurement
    peak_mem = 0
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        with torch.no_grad():
            x = torch.randn(B, N, D, device=device)
            _ = model(x)
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        # CPU memory tracking via PyTorch autograd profiler
        gc.collect()
        with torch.autograd.profiler.profile(profile_memory=True) as prof:
            with torch.no_grad():
                x = torch.randn(B, N, D, device=device)
                _ = model(x)
        # Compute peak memory from cumulative allocation/deallocation events
        cumulative = 0
        peak_bytes = 0
        for evt in prof.function_events:
            cumulative += evt.cpu_memory_usage
            peak_bytes = max(peak_bytes, cumulative)
        peak_mem = peak_bytes / (1024 ** 2)
        # Fallback: if profiler gives 0, use parameter memory estimate
        if peak_mem < 0.01:
            peak_mem = sum(
                p.nelement() * p.element_size() for p in model.parameters()
            ) / (1024 ** 2)

    # Timed trials
    times = []
    with torch.no_grad():
        for _ in range(n_trials):
            x = torch.randn(B, N, D, device=device)

            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            _ = model(x)

            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            times.append(t1 - t0)

    avg_time = np.mean(times)
    std_time = np.std(times)

    return {
        "tokens_per_sec": (B * N) / avg_time,
        "avg_ms": avg_time * 1000,
        "std_ms": std_time * 1000,
        "peak_mem_mb": peak_mem,
    }


def get_batch_size(base_batch_size, seq_len, device):
    """Dynamically reduce batch size for long sequences to avoid OOM."""
    if device != "cuda":
        return base_batch_size
    if seq_len > 2048:
        return max(1, base_batch_size // 8)
    elif seq_len > 1024:
        return max(1, base_batch_size // 4)
    elif seq_len > 512:
        return max(1, base_batch_size // 2)
    return base_batch_size


def run_benchmark(d_model=512, n_heads=8, batch_size=8, device="cuda",
                  seq_lengths=None, sparsity_k=64):
    """Run full benchmark suite."""
    if seq_lengths is None:
        seq_lengths = [128, 256, 512, 1024, 2048, 4096]

    results = {"transformer": {}, "osn_dense": {}, "osn_sparse": {}}

    # Print parameter counts
    osn_block = OSNBlock(d_model, n_heads, sparsity_k=sparsity_k)
    tf_block = StandardTransformerBlock(d_model, n_heads)
    osn_params = sum(p.numel() for p in osn_block.parameters())
    tf_params = sum(p.numel() for p in tf_block.parameters())

    print("=" * 75)
    print(f"OSN Benchmark: d_model={d_model}, n_heads={n_heads}, base_batch={batch_size}")
    print(f"Device: {device}")
    print(f"OSN block: {osn_params:,} params | Transformer block: {tf_params:,} params")
    print(f"OSN advantage: {tf_params - osn_params:+,} fewer params")
    print("=" * 75)
    del osn_block, tf_block

    for N in seq_lengths:
        B = get_batch_size(batch_size, N, device)
        print(f"\n--- Sequence length: {N}, batch_size: {B} ---")

        # Standard Transformer attention
        try:
            model = StandardTransformerBlock(d_model, n_heads).to(device)
            r = benchmark_model(model, B, N, d_model, device=device)
            results["transformer"][N] = r
            print(f"  Transformer:  {r['tokens_per_sec']:.0f} tok/s, "
                  f"{r['avg_ms']:.2f}+-{r['std_ms']:.2f} ms, "
                  f"{r['peak_mem_mb']:.1f} MB")
            del model
        except RuntimeError as e:
            print(f"  Transformer:  OOM or error ({e})")
            results["transformer"][N] = None

        # OSN Dense
        try:
            model = OSNBlock(d_model, n_heads, sparsity_k=None).to(device)
            r = benchmark_model(model, B, N, d_model, device=device)
            results["osn_dense"][N] = r
            print(f"  OSN (dense):  {r['tokens_per_sec']:.0f} tok/s, "
                  f"{r['avg_ms']:.2f}+-{r['std_ms']:.2f} ms, "
                  f"{r['peak_mem_mb']:.1f} MB")
            del model
        except RuntimeError as e:
            print(f"  OSN (dense):  OOM or error ({e})")
            results["osn_dense"][N] = None

        # OSN Sparse
        try:
            k = min(sparsity_k, N)
            model = OSNBlock(d_model, n_heads, sparsity_k=k).to(device)
            r = benchmark_model(model, B, N, d_model, device=device)
            results["osn_sparse"][N] = r
            print(f"  OSN (k={k:3d}):  {r['tokens_per_sec']:.0f} tok/s, "
                  f"{r['avg_ms']:.2f}+-{r['std_ms']:.2f} ms, "
                  f"{r['peak_mem_mb']:.1f} MB")
            del model
        except RuntimeError as e:
            print(f"  OSN (sparse): OOM or error ({e})")
            results["osn_sparse"][N] = None

        if device == "cuda":
            torch.cuda.empty_cache()

    return results


def print_summary(results):
    """Print a formatted summary table."""
    print("\n" + "=" * 75)
    print("SUMMARY: Throughput (tokens/sec)")
    print("=" * 75)
    print(f"{'Seq Len':>8} | {'Transformer':>14} | {'OSN (dense)':>14} | {'OSN (sparse)':>14}")
    print("-" * 75)

    for N in sorted(set(
        list(results["transformer"].keys()) +
        list(results["osn_dense"].keys()) +
        list(results["osn_sparse"].keys())
    )):
        row = f"{N:>8} |"
        for model in ["transformer", "osn_dense", "osn_sparse"]:
            r = results[model].get(N)
            if r is not None:
                row += f" {r['tokens_per_sec']:>12.0f}  |"
            else:
                row += f" {'OOM':>12}  |"
        print(row)

    print("\n" + "=" * 75)
    print("SUMMARY: Peak Memory (MB)")
    print("=" * 75)
    print(f"{'Seq Len':>8} | {'Transformer':>14} | {'OSN (dense)':>14} | {'OSN (sparse)':>14}")
    print("-" * 75)

    for N in sorted(set(
        list(results["transformer"].keys()) +
        list(results["osn_dense"].keys()) +
        list(results["osn_sparse"].keys())
    )):
        row = f"{N:>8} |"
        for model in ["transformer", "osn_dense", "osn_sparse"]:
            r = results[model].get(N)
            if r is not None:
                row += f" {r['peak_mem_mb']:>12.1f}  |"
            else:
                row += f" {'OOM':>12}  |"
        print(row)


def save_results_csv(results, output_dir=".", prefix="osn_benchmark"):
    """Save benchmark results to CSV files."""
    os.makedirs(output_dir, exist_ok=True)

    seq_lengths = sorted(set(
        list(results["transformer"].keys()) +
        list(results["osn_dense"].keys()) +
        list(results["osn_sparse"].keys())
    ))

    # Throughput CSV
    throughput_path = os.path.join(output_dir, f"{prefix}_throughput.csv")
    with open(throughput_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seq_len", "transformer_tok_per_sec",
                         "osn_dense_tok_per_sec", "osn_sparse_tok_per_sec"])
        for N in seq_lengths:
            row = [N]
            for model in ["transformer", "osn_dense", "osn_sparse"]:
                r = results[model].get(N)
                row.append(f"{r['tokens_per_sec']:.2f}" if r else "OOM")
            writer.writerow(row)
    print(f"\nSaved: {throughput_path}")

    # Memory CSV
    memory_path = os.path.join(output_dir, f"{prefix}_memory.csv")
    with open(memory_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seq_len", "transformer_peak_mb",
                         "osn_dense_peak_mb", "osn_sparse_peak_mb"])
        for N in seq_lengths:
            row = [N]
            for model in ["transformer", "osn_dense", "osn_sparse"]:
                r = results[model].get(N)
                row.append(f"{r['peak_mem_mb']:.2f}" if r else "OOM")
            writer.writerow(row)
    print(f"Saved: {memory_path}")

    # Detailed CSV (all metrics in one file)
    detailed_path = os.path.join(output_dir, f"{prefix}_detailed.csv")
    with open(detailed_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seq_len", "model", "tokens_per_sec",
                         "avg_ms", "std_ms", "peak_mem_mb"])
        for N in seq_lengths:
            for model_name in ["transformer", "osn_dense", "osn_sparse"]:
                r = results[model_name].get(N)
                if r:
                    writer.writerow([
                        N, model_name,
                        f"{r['tokens_per_sec']:.2f}",
                        f"{r['avg_ms']:.4f}",
                        f"{r['std_ms']:.4f}",
                        f"{r['peak_mem_mb']:.2f}",
                    ])
                else:
                    writer.writerow([N, model_name, "OOM", "OOM", "OOM", "OOM"])
    print(f"Saved: {detailed_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OSN Benchmark")
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--sparsity_k", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seq_lengths", type=int, nargs="+",
                        default=[128, 256, 512, 1024, 2048, 4096])
    parser.add_argument("--output_dir", type=str, default="benchmark_results",
                        help="Directory to save CSV results")
    args = parser.parse_args()

    results = run_benchmark(
        d_model=args.d_model,
        n_heads=args.n_heads,
        batch_size=args.batch_size,
        device=args.device,
        seq_lengths=args.seq_lengths,
        sparsity_k=args.sparsity_k,
    )
    print_summary(results)
    save_results_csv(results, output_dir=args.output_dir)
