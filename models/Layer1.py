import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class Attention(nn.Module):
    def __init__(
        self, 
        dim, 
        heads = 8, 
        dim_head = 64, 
        dropout = 0.
    ):
        super().__init__()

        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(dim_head)
        self.norm_v = nn.LayerNorm(dim_head)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(
            dim, 
            inner_dim * 3, 
            bias = False
        )
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)

        q, k, v = map(
            lambda t: rearrange(
                t, 
                'b n (h d) -> b h n d', 
                h = self.heads
            ), qkv)
        k = self.norm_k(k)
        v = self.norm_v(v)
        with torch.backends.cuda.sdp_kernel(enable_math=True):
            out = F.scaled_dot_product_attention(q, k, v)
            out = self.dropout(out)
            out = rearrange(out, 'b h n d -> b n (h d)')
            return self.to_out(out)

class LinearAttention(nn.Module):
    def __init__(self, dim=1024, output_layer='None', dropout=0.0, heads = 8):
        super(LinearAttention, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.d_model = dim
        self.out = nn.Linear(dim, dim) if output_layer == 'linear' else nn.Identity()
        self.dropout = nn.Dropout(dropout) 
        self.heads = heads
        self.to_qkv = nn.Linear(
            dim, 
            dim * 3, 
            bias = False
        )
        self.sig= nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.gl = nn.GELU()
    def forward(self, x, mask=None):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(
            lambda t: rearrange(
                t, 
                'b n (h d) -> b h n d', 
                h = self.heads
            ), qkv)
        
        q = self.sig(q)
        k = self.tanh(k)
        if mask is not None:
            k = k.masked_fill(mask.unsqueeze(-1), 0)
        k = k.transpose(-1, -2)
        kvw = torch.matmul(k, v)
        if self.dropout.p > 0:
            kvw = self.dropout(kvw.transpose(-1, -2)).transpose(-1, -2)
        out = torch.matmul(q, kvw)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.out(out)
    
# with out FFN
class TransLayer_N(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=1024):
        super().__init__()
        self.attn = Attention(dim=dim)
    def forward(self, x):
        x = x + self.attn((x))
        return x
# with out FFN
class TransLayer_L(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=1024):
        super().__init__()
        self.attn = LinearAttention(dim=dim)
    def forward(self, x):
        x = x + self.attn(x)
        return x
class Layer1(nn.Module):
    def __init__(self, n_classes=3, ds=512):
        super(Layer1, self).__init__()
        self._fc1 = nn.Sequential(nn.Linear(768, ds), nn.GELU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, ds))
        self.n_classes = n_classes
        self.layer1 = TransLayer_N(dim=ds)
        self.layer2 = TransLayer_N(dim=ds)
        self.layer3 = TransLayer_L(dim=ds)
        self.layer4 = TransLayer_L(dim=ds)
        self.norm = nn.LayerNorm(ds)
        self._fc2 = nn.Linear(ds, self.n_classes)
    def forward(self, **kwargs):

        h = kwargs['data'].float() #[B, n, 768]
        h = self._fc1(h)
        h = self.layer1(h)
        h = self.layer2(h)
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).to(h.device)
        h = torch.cat((cls_tokens, h), dim=1)
        h = self.layer3(h)
        h = self.layer4(h)
        h = self.norm(h)[:,0]
        #---->predict
        logits = self._fc2(h) #[B, n_classes]
        Y_hat = torch.ge(logits, 0.5).float()
        Y_prob = F.sigmoid(logits)
        results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
        return results_dict, h

if __name__ == "__main__":
    data = torch.randn((1, 6000, 768)).cuda()
    model = Layer1().cuda()
    print(model.eval())
    results_dict = model(data = data)
