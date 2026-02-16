"""
Oscillatory Synchronization Network (OSN)
==========================================
Full implementation of the Selective Synchronization Attention (SSA) mechanism
and OSN block as described in "Resonant Sparse Geometry Networks".

Author: Hasi Hays (hasih@uark.edu)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveSynchronizationAttention(nn.Module):
    """
    Selective Synchronization Attention (SSA).

    Replaces standard dot-product attention with a closed-form operator
    derived from the steady-state Kuramoto model of coupled oscillators.

    Each token is represented as an oscillator with:
      - Natural frequency omega (encodes position + semantics)
      - Phase theta (encodes content; used for order parameter computation)
      - Coupling J (frequency-dependent: J_ij = exp(-alpha * ||omega_i - omega_j||^2))

    The synchronization matrix S replaces the attention weight matrix,
    with natural sparsity from the phase-locking condition. The order
    parameter r is computed empirically from the phase distribution at
    each forward pass, consistent with the Kuramoto formulation.
    """

    def __init__(self, d_model, n_heads, K_init=1.0, sparsity_k=None, dropout=0.1, alpha_init=1.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.sparsity_k = sparsity_k

        # Learnable global coupling strength (softplus ensures K > 0)
        self.K = nn.Parameter(torch.tensor(K_init))

        # Learnable coupling bandwidth per head (controls frequency-dependent
        # coupling decay; softplus ensures alpha > 0)
        self.alpha = nn.Parameter(torch.full((n_heads,), alpha_init))

        # Projections
        self.W_omega = nn.Linear(d_model, d_model)   # Natural frequency projection
        self.W_theta = nn.Linear(d_model, d_model)   # Phase projection
        self.W_v = nn.Linear(d_model, d_model)        # Value projection
        self.W_o = nn.Linear(d_model, d_model)        # Output projection

        self.attn_dropout = nn.Dropout(dropout)
        self.eps = 1e-8

    def forward(self, x, return_sync_matrix=False):
        """
        Args:
            x: Input tensor of shape (B, N, D)
            return_sync_matrix: If True, also return the synchronization matrix

        Returns:
            Output tensor of shape (B, N, D)
            Optionally: synchronization matrix of shape (B, H, N, N)
        """
        B, N, D = x.shape
        H = self.n_heads
        d = self.d_head

        # Project to oscillatory parameters
        omega = self.W_omega(x).view(B, N, H, d)   # Natural frequencies
        theta = self.W_theta(x).view(B, N, H, d)   # Initial phases
        V = self.W_v(x).view(B, N, H, d)           # Values

        # Compute pairwise frequency mismatch (memory-efficient)
        # Uses ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b to avoid O(N^2*d) intermediate
        omega_sq_sum = omega.pow(2).sum(dim=-1)  # (B, N, H)
        omega_i_sq = omega_sq_sum.unsqueeze(2)  # (B, N, 1, H)
        omega_j_sq = omega_sq_sum.unsqueeze(1)  # (B, 1, N, H)
        # Batch matmul for dot products: (B, H, N, d) @ (B, H, d, N) -> (B, H, N, N)
        omega_t = omega.permute(0, 2, 1, 3)  # (B, H, N, d)
        dot_prod = torch.matmul(omega_t, omega_t.transpose(-2, -1))  # (B, H, N, N)
        dot_prod = dot_prod.permute(0, 2, 3, 1)  # (B, N, N, H)
        delta_omega_sq = torch.relu(omega_i_sq + omega_j_sq - 2 * dot_prod)  # (B, N, N, H)
        delta_omega = delta_omega_sq.sqrt()  # (B, N, N, H)

        # Frequency-dependent coupling: J_ij = exp(-alpha_h * ||omega_i - omega_j||^2)
        # Physics-motivated: oscillators with similar frequencies couple more strongly
        alpha = F.softplus(self.alpha)  # (H,) -> positive coupling bandwidth
        J = torch.exp(-alpha * delta_omega_sq)  # (B, N, N, H)

        # Compute empirical order parameter from phase distribution (Kuramoto Eq. 3)
        # r = (1/d) * sum_l |mean_j exp(i * theta_j^l)| per head
        r = (theta.cos().mean(dim=1).pow(2)
             + theta.sin().mean(dim=1).pow(2)).sqrt()  # (B, H, d)
        r = r.mean(dim=-1)  # (B, H)
        r = r.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, H)

        # Compute synchronization threshold
        K = F.softplus(self.K)  # smooth positive constraint
        threshold = K * r * J + self.eps

        # Compute synchronization matrix using closed-form Kuramoto steady-state
        ratio = torch.clamp(delta_omega / threshold, -1.0 + self.eps, 1.0 - self.eps)

        # Phase-locking condition: S_ij > 0 only if delta_omega <= threshold
        phase_locked = (delta_omega <= threshold).float()

        # Synchronization strength: J * cos(arcsin(ratio))
        S = J * torch.cos(torch.asin(ratio)) * phase_locked  # (B, N, N, H)

        # Optional top-k sparsification
        if self.sparsity_k is not None and self.sparsity_k < N:
            topk_vals, topk_idx = S.topk(self.sparsity_k, dim=2)
            S_sparse = torch.zeros_like(S)
            S_sparse.scatter_(2, topk_idx, topk_vals)
            S = S_sparse

        # Row-normalize (analogous to softmax normalization)
        S_sum = S.sum(dim=2, keepdim=True) + self.eps
        S_norm = S / S_sum

        # Apply dropout to synchronization weights
        S_norm = self.attn_dropout(S_norm)

        # Rearrange for batched matrix multiply: (B, N, N, H) -> (B, H, N, N)
        S_norm = S_norm.permute(0, 3, 1, 2)
        V = V.permute(0, 2, 1, 3)  # (B, H, N, d)

        # Compute output: synchronization-weighted sum of values
        out = torch.matmul(S_norm, V)  # (B, H, N, d)

        # Concatenate heads and project
        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, D)
        out = self.W_o(out)

        if return_sync_matrix:
            return out, S.permute(0, 3, 1, 2)  # Return (B, H, N, N)
        return out


class OSNBlock(nn.Module):
    """
    Oscillatory Synchronization Network block.

    Drop-in replacement for a standard Transformer block.
    Uses pre-norm architecture with SSA replacing multi-head attention.
    """

    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1,
                 K_init=1.0, sparsity_k=None):
        super().__init__()

        if d_ff is None:
            d_ff = 4 * d_model

        self.ssa = SelectiveSynchronizationAttention(
            d_model, n_heads, K_init=K_init,
            sparsity_k=sparsity_k, dropout=dropout
        )

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, N, D)
        Returns:
            Output tensor of shape (B, N, D)
        """
        # Pre-norm SSA with residual
        x = x + self.dropout(self.ssa(self.norm1(x)))
        # Pre-norm FFN with residual
        x = x + self.ffn(self.norm2(x))
        return x


class OSN(nn.Module):
    """
    Full Oscillatory Synchronization Network.

    Stacks OSN blocks with token embedding and optional output head.
    """

    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff=None,
                 max_seq_len=2048, dropout=0.1, K_init=1.0, sparsity_k=None):
        super().__init__()

        self.d_model = d_model

        # Token embedding (no positional encoding -- frequencies handle position)
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.embed_dropout = nn.Dropout(dropout)

        # Stack of OSN blocks
        self.blocks = nn.ModuleList([
            OSNBlock(d_model, n_heads, d_ff=d_ff, dropout=dropout,
                     K_init=K_init, sparsity_k=sparsity_k)
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)

        # Language modeling head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.token_embed.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids):
        """
        Args:
            input_ids: Token IDs of shape (B, N)
        Returns:
            Logits of shape (B, N, vocab_size)
        """
        x = self.token_embed(input_ids) * math.sqrt(self.d_model)
        x = self.embed_dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits


class StandardTransformerAttention(nn.Module):
    """
    Standard multi-head dot-product attention for benchmarking comparison.
    """

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, D = x.shape
        H = self.n_heads
        d = self.d_head

        Q = self.W_q(x).view(B, N, H, d).permute(0, 2, 1, 3)
        K = self.W_k(x).view(B, N, H, d).permute(0, 2, 1, 3)
        V = self.W_v(x).view(B, N, H, d).permute(0, 2, 1, 3)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d)
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, V)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, D)
        return self.W_o(out)


class StandardTransformerBlock(nn.Module):
    """Standard Transformer block for benchmarking comparison."""

    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model

        self.attn = StandardTransformerAttention(d_model, n_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.ffn(self.norm2(x))
        return x


if __name__ == "__main__":
    # Quick test
    B, N, D = 2, 128, 256
    n_heads = 8

    x = torch.randn(B, N, D)

    # Test SSA
    ssa = SelectiveSynchronizationAttention(D, n_heads, sparsity_k=32)
    out, S = ssa(x, return_sync_matrix=True)
    print(f"SSA output shape: {out.shape}")
    print(f"Sync matrix shape: {S.shape}")
    print(f"Sync matrix sparsity: {(S == 0).float().mean():.2%}")

    # Test OSN block
    block = OSNBlock(D, n_heads, sparsity_k=32)
    out = block(x)
    print(f"OSN block output shape: {out.shape}")

    # Test full OSN
    model = OSN(vocab_size=1000, d_model=D, n_layers=4, n_heads=n_heads, sparsity_k=32)
    ids = torch.randint(0, 1000, (B, N))
    logits = model(ids)
    print(f"OSN logits shape: {logits.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
