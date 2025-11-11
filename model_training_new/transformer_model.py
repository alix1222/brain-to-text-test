import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.layers import trunc_normal_, DropPath
from einops import rearrange, repeat, pack, unpack
from typing import Tuple


class PatchEmbed_Day(nn.Module):
    def __init__(self, patch_size=100, emb_dim=64):
        super().__init__()
        self.proj = nn.Conv2d(in_channels=1, out_channels=emb_dim, kernel_size=(1, patch_size), stride=(1, patch_size))

    def forward(self, x):
        x = rearrange(x, 'B D N -> B 1 N D')
        x = self.proj(x)
        x = rearrange(x, 'B D N T -> B N T D')
        return x
    

class PatchEmbed(nn.Module):
    def __init__(self, patch_size=100, emb_dim=64):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size, emb_dim)
        self.norm1 = nn.LayerNorm(patch_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(emb_dim, eps=1e-6)

    def forward(self, x):
        x = rearrange(x, 'B (T D) N -> B N T D', D=self.patch_size)
        x = self.norm1(x)
        x = self.proj(x)
        x = self.norm2(x)
        return x
    

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def apply_scaling(freqs: torch.Tensor, scale_factor: float, high_freq_factor: float):
    low_freq_factor = 1
    old_context_len = 8192  # original llama3 length

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / scale_factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
            new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)


def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float,
    use_scaled: bool,
    scale_factor: float,
    high_freq_factor: float,
):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    if use_scaled:
        freqs = apply_scaling(freqs, scale_factor, high_freq_factor)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis
    

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    

class Self_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, freqs_cis, attn_mask=None):
        B, N, C = x.shape
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # (3, B, num_heads, N, head_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3, 4) # (3, B, N, num_heads, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis) 
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 0, float('-inf'))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        self.attn = Self_Attention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=hidden_dim, out_features=dim, drop=drop)

    def forward(self, x, freqs_cis, attn_mask=None):
        x = x + self.drop_path(self.attn(self.norm1(x), freqs_cis=freqs_cis, attn_mask=attn_mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    

class Transformer(nn.Module):
    def __init__(self, dim, depth, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0.):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop, drop_path=dpr[i]
            )
            for i in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        
    def forward(self, x, freqs_cis, attn_mask=None):
        for blk in self.blocks:
            x = blk(x, freqs_cis=freqs_cis, attn_mask=attn_mask)
        return self.norm(x)


class TransformerDecoder(nn.Module):
    def __init__(
            self, 
            num_features=512,    
            num_days=50,         
            num_classes=41,      
            patch_size=4,
            patch_stride=4,   
            emb_dim=512, 
            depth=8, 
            num_heads=8, 
            mlp_ratio=4., 
            qkv_bias=True, 
            qk_scale=None, 
            input_dropout=0.2, 
            drop_rate=0.4, 
            attn_drop_rate=0.0, 
            drop_path_rate=0.0, 
            init_std=0.02,
            use_day_layer=True,
            max_mask_pct=0,
            num_masks=0,
            mask_token_zeros=False,
            use_register_tokens=False,
            num_register_tokens=1,
            ):
        super().__init__()
        self.num_features = num_features
        self.num_days = num_days
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.patch_stride = patch_stride if patch_stride > 0 else patch_size
        self.emb_dim = emb_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.init_std = init_std
        self.use_day_layer = use_day_layer
        self.max_mask_pct = max_mask_pct
        self.num_masks = num_masks
        self.mask_token_zeros = mask_token_zeros
        self.use_register_tokens = use_register_tokens
        self.num_register_tokens = num_register_tokens

        self.input_dropout = input_dropout
        if use_day_layer:
            self.day_layer_activation = nn.Softsign()
            self.day_weights = nn.ParameterList(
                [nn.Parameter(torch.eye(self.num_features, self.num_features)) for _ in range(self.num_days)]
            )
            self.day_biases = nn.ParameterList(
                [nn.Parameter(torch.zeros(1, self.num_features)) for _ in range(self.num_days)]
            )
            self.day_layer_dropout = nn.Dropout(input_dropout)
            self.patch_embed = nn.Linear(self.num_features * self.patch_size, emb_dim) if self.patch_size > 0 else nn.Linear(self.num_features, emb_dim)
        else:
            self.patch_embed = nn.Sequential(
                nn.LayerNorm(self.num_features * self.patch_size, eps=1e-6),
                nn.Linear(self.num_features * self.patch_size, emb_dim) if self.patch_size > 0 else nn.Linear(self.num_features, emb_dim),
                nn.LayerNorm(emb_dim, eps=1e-6)
            )
        max_seq_len = 2048  # Conservative estimate, can be adjusted based on your data
        rope_theta = 10000.0  # Good default for neural sequences
        use_scaled_rope = False # Set to False unless you need extrapolation beyond max_seq_len
        rope_scaling_factor = 1.0 # Scaling factor for longer sequences (only used if use_scaled_rope=True). If your sequences become 2x longer, use rope_scaling_factor=2.0
        rope_high_freq_factor = 4.0 # High frequency factor for scaled RoPE (only used if use_scaled_rope=True). Controls which frequencies get scaled: higher = more aggressive scaling
        
        # Precompute RoPE frequency matrix
        self.freqs_cis = precompute_freqs_cis(
            emb_dim // num_heads,      # Head dimension for rotary embedding
            max_seq_len * 2,           # 2x for safety margin
            rope_theta,
            use_scaled_rope,
            rope_scaling_factor,
            rope_high_freq_factor,
        )
        self.pos_drop = nn.Dropout(drop_rate)
        if use_register_tokens:
            self.register_tokens = nn.Parameter(torch.randn(num_register_tokens, emb_dim))
        self.transformer = Transformer(
            dim=emb_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate
        )   
        self.head = nn.Linear(emb_dim, num_classes)
        # Time Masking parameters
        self.max_mask_pct = max_mask_pct
        self.num_masks = num_masks 
        if mask_token_zeros:
            self.mask_token = nn.Parameter(torch.zeros(self.num_features * self.patch_size), requires_grad=False)
        else:
            self.mask_token = nn.Parameter(torch.randn(self.num_features * self.patch_size))
        # initialize weights
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.transformer.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            if hasattr(layer.mlp, 'fc2'):
                rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def apply_time_mask(self, X, X_len, constant_mask=False, mask_range=[]):
        
        """
        Fully vectorized SpecAugment-style time masking (no loops at all).
        
        Args:
            X: (B, P, D) input tensor
            X_len: (B,) valid lengths in timepoints
            constant_mask_lengths: if True, make the mask lengths the same across all batches

        Returns:
            X_masked: (B, P, D) with masked patches
            mask: (B, P) boolean mask of where values were masked
            masked_indices: list of 1D LongTensors, each with indices of masked patches per batch
            unmasked_indices: list of 1D LongTensors, each with indices of unmasked patches per batch
        """
        B, P, D = X.shape
        device = X.device

        if constant_mask:
            # get valid len of smallest trial in batch and repeat for all batches. 
            valid_lens = torch.min((X_len // self.patch_stride).to(device)).repeat(B)
        else:
            valid_lens = (X_len // self.patch_stride).to(device)
            
        max_mask_lens = (self.max_mask_pct * valid_lens).long()  # (B,)

        # Repeat B num_masks times to simulate multiple masks per sample
        B_rep = B * self.num_masks

        # Expand inputs for vectorized masking
        # repeat_interleave works like tile, so values corresponding to the same batch are next to each other
        valid_lens_rep = valid_lens.repeat_interleave(self.num_masks)            # (B * num_masks,)
        max_mask_lens_rep = max_mask_lens.repeat_interleave(self.num_masks)      # (B * num_masks,)

        if constant_mask:
            # select the same t for every batch. 
            t = (torch.rand(self.num_masks, device=device).repeat(B) * (max_mask_lens_rep + 1).float()).floor().long().clamp(min=1)  # (B * num_masks,)
        else:
            t = (torch.rand(B_rep, device=device) * (max_mask_lens_rep + 1).float()).floor().long()  # (B * num_masks,)
            
        max_start = (valid_lens_rep - t + 1).clamp(min=1)
        
        if constant_mask:
            t0 = (torch.rand(self.num_masks, device=device).repeat(B) * max_start.float()).floor().long()               # (B * num_masks,)
        else:
            t0 = (torch.rand(B_rep, device=device) * max_start.float()).floor().long()               # (B * num_masks,)

        # Build the global mask (B, P)
        arange = torch.arange(P, device=device).unsqueeze(0)       # (1, P)
        t0_exp = t0.unsqueeze(1)                                   # (B_rep, 1)
        t1_exp = (t0 + t).unsqueeze(1)                             # (B_rep, 1)
        mask_chunks = (arange >= t0_exp) & (arange < t1_exp)       # (B_rep, P)
        
        # Get index of sample in batch for each mask chunk
        batch_idx = torch.arange(B, device=device).repeat_interleave(self.num_masks)  # (B * num_masks,)

        # Now scatter all the masks into the full mask (B, P)
        patch_idx = mask_chunks.nonzero(as_tuple=False)  # (N_masked, 2)
        b_indices = batch_idx[patch_idx[:, 0]]           # (N_masked,)
        p_indices = patch_idx[:, 1]                      # (N_masked,)

        mask = torch.zeros(B, P, dtype=torch.bool, device=device)
        mask[b_indices, p_indices] = True
        
        # mask: (B, P) boolean, True for masked
        #B, P = mask.shape

        # Number of masked patches per batch (assumed same for all batches)
        if constant_mask:
            N = mask.sum(dim=1)[0].item()
            U = P - N  # Number of unmasked per batch
                            
            masked_indices = mask.nonzero(as_tuple=False)  # (B * N, 2) — rows: [batch_idx, patch_idx]
            masked_indices = masked_indices[:, 1].reshape(B, N)
            masked_indices = torch.sort(masked_indices, dim=-1).values  # sort within batch
        
            unmasked = ~mask  # invert the mask
            unmasked_indices = unmasked.nonzero(as_tuple=False)[:, 1].reshape(B, U)
            unmasked_indices = torch.sort(unmasked_indices, dim=-1).values
        
            return masked_indices, unmasked_indices
        
        # Apply the mask
        X_masked = X.clone()
        # X_masked[mask] = self.mask_token
        # Ensure mask token matches X's dtype and device
        if self.mask_token_zeros:
            # Use zeros with matching dtype
            X_masked[mask] = 0.0
        else:
            # Convert mask_token to match X's dtype and device
            mask_token = self.mask_token.to(dtype=X.dtype, device=X.device)
            X_masked[mask] = mask_token

        return X_masked, mask

    def forward(self, x, day_idx, X_len, attn_mask=None):
        if self.use_day_layer:
            day_weights = torch.stack([self.day_weights[i] for i in day_idx], dim=0)
            day_biases = torch.cat([self.day_biases[i] for i in day_idx], dim=0).unsqueeze(1)
            x = torch.einsum("btd,bdk->btk", x, day_weights) + day_biases
            x = self.day_layer_activation(x)
            if self.input_dropout > 0:
                x = self.day_layer_dropout(x)
        if self.patch_size > 0: 
            B, T, C = x.shape
            x = x.permute(0, 2, 1).contiguous()  # (B, C, T)
            x_unfold = x.unfold(2, self.patch_size, self.patch_stride).contiguous()  # (B, C, num_patches, patch_size)
            x_unfold = x_unfold.permute(0, 2, 3, 1).contiguous()  # (B, num_patches, patch_size, C)
            x = x_unfold.reshape(B, x_unfold.size(1), -1).contiguous()  # (B, num_patches, patch_size * C)
        if self.training and self.max_mask_pct > 0:
            x, _ = self.apply_time_mask(x, X_len)    
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        freq_cis = self.freqs_cis[:x.size(1), :].to(x.device)
        if self.use_register_tokens:
            r = repeat(self.register_tokens, 'N D -> B N D', B=B)
            x, ps = pack([x, r], 'B * D')
        x = self.transformer(x, freqs_cis=freq_cis, attn_mask=attn_mask)
        if self.use_register_tokens:
            x, _ = unpack(x, ps, 'B * D')
        x = self.head(x)
        return x