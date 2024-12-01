import torch.nn as nn
import torch
from torchinfo import summary  ## pip install torchinfo
from ptflops import get_model_complexity_info  ## pip install ptflops
'''
https://debuggercafe.com/vision-transformer-from-scratch/
'''
class CreatePatches(nn.Module):
    def __init__(
        self, channels=3, embed_dim=768, patch_size=16
    ):
        super().__init__()
        self.patch = nn.Conv2d(
            in_channels=channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
    def forward(self, x):
        # Flatten along dim = 2 to maintain channel dimension.
        patches = self.patch(x).flatten(2).transpose(1, 2)
        return patches


class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        super().__init__()
        self.pre_norm = nn.LayerNorm(embed_dim, eps=1e-06)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-06)
        self.MLP = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        x_norm = self.pre_norm(x)
        # MultiheadAttention returns attention output and weights,
        # we need only the outputs, so [0] index.
        x = x + self.attention(x_norm, x_norm, x_norm)[0]
        x = x + self.MLP(self.norm(x))
        return x


class ViT(nn.Module):  ## default as ViT_base setting
    def __init__(
            self,
            img_size=224,
            in_channels=3,
            patch_size=16,
            embed_dim=768,
            hidden_dim=3072,
            num_heads=12,
            num_layers=12,
            dropout=0.0,
            num_classes=1000
    ):
        super().__init__()
        self.patch_size = patch_size
        num_patches = (img_size // patch_size) ** 2
        self.patches = CreatePatches(
            channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size
        )
        # Postional encoding.
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.attn_layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.attn_layers.append(
                AttentionBlock(embed_dim, hidden_dim, num_heads, dropout)
            )
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(embed_dim, eps=1e-06)
        self.head = nn.Linear(embed_dim, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
    #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #         if hasattr(m, 'bias') and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.patches(x)
        b, n, _ = x.shape

        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)
        for layer in self.attn_layers:
            x = layer(x)
        x = self.ln(x)
        x = x[:, 0]
        return self.head(x)

if __name__ == '__main__':
    model = ViT(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        hidden_dim=3072,
        num_heads=12,
        num_layers=12,
        num_classes=4
    )
    # print(model)

    ### 以下代码检查网络大小 ###
    macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)
    # 检查是否有 None 值
    macs = macs if macs is not None else "Not Available"
    params = params if params is not None else "Not Available"
    print('{:<30}  {:<8}'.format('Number of parameters:', params))
    print('{:<30}  {:<8}'.format('Computational complexity:', macs))