# Oscillatory Synchronization Network (OSN)

Official implementation of **Selective Synchronization Attention (SSA)**, a novel attention mechanism derived from the steady-state Kuramoto model of coupled oscillators.

## Overview

SSA replaces standard dot-product self-attention with a closed-form synchronization operator. Each token is represented as an oscillator with a learnable natural frequency and phase; the synchronization strength between token pairs serves as the attention weight. Key properties:

- **Natural sparsity** from the phase-locking condition (no explicit masking needed)
- **Unified positional-semantic encoding** through learnable natural frequencies
- **Single-pass closed-form computation** (no iterative ODE integration)
- **Drop-in replacement** for standard Transformer blocks

## Repository Structure

```
OSN/
  osn_model.py          # Core SSA and OSN block implementation
  benchmark.py          # Throughput, latency, and memory benchmarking
  osn_benchmark.ipynb   # Google Colab notebook (GPU benchmarks on A100)
  requirements.txt      # Python dependencies
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run the model

```python
import torch
from osn_model import OSNBlock

# Create an OSN block (drop-in Transformer replacement)
osn = OSNBlock(d_model=512, n_heads=8, d_ff=2048, top_k=64)

# Forward pass
x = torch.randn(2, 128, 512)  # (batch, seq_len, d_model)
y = osn(x)  # Same shape: (2, 128, 512)
```

### Run benchmarks

```bash
# Local CPU benchmark
python benchmark.py --seq_lens 128 256 512 --output_dir results/

# For GPU benchmarks, use the Colab notebook:
# osn_benchmark.ipynb (requires NVIDIA A100 or similar)
```

## Benchmark Results (NVIDIA A100)

| Seq Len | Batch | TF (K tok/s) | OSN Dense | OSN Sparse | TF (MB) | OSN Dense | OSN Sparse |
|---------|-------|-------------|-----------|------------|---------|-----------|------------|
| 128     | 8     | 1,033       | 665       | 638        | 313     | 345       | 351        |
| 256     | 8     | 1,527       | 1,118     | 788        | 353     | 481       | 493        |
| 512     | 8     | 1,636       | 950       | 700        | 481     | 993       | 1,017      |
| 1,024   | 4     | 1,334       | 593       | 457        | 609     | 1,633     | 1,657      |
| 2,048   | 2     | 977         | 350       | 277        | 865     | 2,913     | 2,937      |
| 4,096   | 1     | 628         | 194       | 155        | 1,377   | 5,465     | 5,489      |

Parameter counts: OSN block 3,152,393 vs Transformer block 3,152,384 (difference of only 9 parameters).

## Citation

If you use this code, please cite:

```bibtex
@article{hays2026ssa,
  title={Selective Synchronization Attention},
  author={Hays, Hasi},
  year={2026}
}
```

## License

MIT License
