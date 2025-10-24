import math
import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
from functools import partial

def named_apply(fn, module: nn.Module, name="", depth_first=True, include_root=False):
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = (bias, bias)
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class MemEffAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, proj_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class NestedTensorBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, proj_bias=True, ffn_bias=True,
                 drop=0., attn_drop=0., init_values=None, drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, attn_class=MemEffAttention, ffn_layer=Mlp):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(dim, num_heads=num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias,
                              attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer,
                            drop=drop, bias=ffn_bias)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.flatten = flatten
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

def init_weights_vit(module: nn.Module, name: str = ""):
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

class DinoVisionTransformer(nn.Module):
    def __init__(self, img_size=518, patch_size=14, in_chans=3, embed_dim=768, depth=12, num_heads=12,
                 mlp_ratio=4., qkv_bias=True, ffn_bias=True, proj_bias=True, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU,
                 block_fn=NestedTensorBlock, ffn_layer=Mlp, block_chunks=0, num_register_tokens=0, 
                 interpolate_antialias=False, interpolate_offset=0.1, init_values=1.0, **kwargs):
        super().__init__()
        
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.register_tokens = nn.Parameter(torch.zeros(self.num_register_tokens, embed_dim)) if num_register_tokens else None
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

        # Build blocks
        self.blocks = nn.ModuleList([
            block_fn(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                    proj_bias=proj_bias, ffn_bias=ffn_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                    drop_path=0., norm_layer=norm_layer, act_layer=act_layer, ffn_layer=ffn_layer,
                    init_values=init_values)
            for i in range(depth)
        ])

        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()
        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)
        named_apply(init_weights_vit, self)

    def interpolate_pos_encoding(self, x, w, h):
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        M = int(math.sqrt(N))
        assert N == M * M
        kwargs = {}
        if self.interpolate_offset:
            sx = float(w0 + self.interpolate_offset) / M
            sy = float(h0 + self.interpolate_offset) / M
            kwargs["scale_factor"] = (sx, sy)
        else:
            kwargs["size"] = (w0, h0)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=self.interpolate_antialias,
            **kwargs,
        )
        assert (w0, h0) == (patch_pos_embed.shape[-2], patch_pos_embed.shape[-1])
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_tokens_with_masks(self, x, masks=None, prompt=None):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)

        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        
        # Insert prompt tokens between CLS token and patch tokens
        if prompt is not None:
            x = torch.cat((x[:, :1], prompt, x[:, 1:]), dim=1)

        if self.register_tokens is not None:
            x = torch.cat((x[:, :1], self.register_tokens.expand(x.shape[0], -1, -1), x[:, 1:]), dim=1)

        return x

    def forward_features(self, x, masks=None, prompt=None):
        x = self.prepare_tokens_with_masks(x, masks, prompt)
        for blk in self.blocks:
            x = blk(x)
        x_norm = self.norm(x)
        if prompt is not None:
            n_prompts = prompt.shape[1]
            return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
            "x_norm_prompts": x_norm[:, self.num_register_tokens + 1 : n_prompts + 1],
            "x_norm_patchtokens": x_norm[:, self.num_register_tokens + n_prompts + 1 :],
            "x_prenorm": x,
            "masks": masks,
        }
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
            "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
            "x_prenorm": x,
            "masks": masks,
        }

    def forward(self, x, is_training=False, prompt=None):
        ret = self.forward_features(x=x, prompt=prompt)
        if is_training:
            return ret
        else:
            return self.head(ret["x_norm_clstoken"])

def vit_base(patch_size=16, num_register_tokens=0, init_values=1.0, block_chunks=0, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        block_fn=partial(NestedTensorBlock, attn_class=MemEffAttention),
        num_register_tokens=num_register_tokens,
        init_values=init_values,
        block_chunks=block_chunks,
        **kwargs,
    )
    return model

def freeze_all_but_bn(m):
    if not isinstance(m, torch.nn.LayerNorm):
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.requires_grad_(False)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.requires_grad_(False)

class DinoV2Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = vit_base(patch_size=14, block_chunks=0, init_values=1.0)

    def forward(self, x, prompt, is_training=False):
        return self.model(x, prompt=prompt, is_training=is_training)

class S3birDinov2(nn.Module):
    def __init__(self, n_prompts=3, prompt_dim=768):
        super().__init__()
        self.encoder = DinoV2Encoder()
        self.encoder.apply(freeze_all_but_bn)
        
        # Prompt Engineering
        self.sk_prompt = nn.Parameter(torch.randn(n_prompts, prompt_dim))
        self.img_prompt = nn.Parameter(torch.randn(n_prompts, prompt_dim))

    def forward(self, data, dtype='image', is_training=False):
        if dtype == 'image':
            feat = self.encoder(data, 
                                prompt=self.img_prompt.expand(data.shape[0], -1, -1),
                                is_training=is_training)
        else:
            feat = self.encoder(data, 
                                prompt=self.sk_prompt.expand(data.shape[0], -1, -1),
                                is_training=is_training)
        return feat
