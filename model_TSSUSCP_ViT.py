import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_TSSUSCP_ViT import TSSUSCP_ViT

def _to_2tuple(x, name):
    if isinstance(x, (tuple, list)):
        if len(x) != 2:
            raise ValueError(
                f"{name} must be a tuple/list of length 2, got {x}."
            )
        return int(x[0]), int(x[1])

    return int(x), int(x)



class TSSUSCP_ViT_PatchEmbedding(nn.Module):
    def __init__(
        self,
        patch_size,
        squeeze_rate,
        in_channel,
        embed_channel,
    ):
        super().__init__()

        self.patch_size = _to_2tuple(patch_size,"patch_size")
        self.squeeze_rate = _to_2tuple(squeeze_rate,"squeeze_rate")

        self.in_channel = int(in_channel)
        self.embed_channel = int(embed_channel)

        self.W_embedding = nn.Conv2d(
            in_channels=self.in_channel,
            out_channels=self.embed_channel,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.W_squeeze = nn.Conv2d(
            in_channels=self.embed_channel,
            out_channels=self.embed_channel,
            kernel_size=self.squeeze_rate,
            stride=self.squeeze_rate,
            padding=0,
            groups=self.embed_channel,
        )

    def forward(self,X):
        if X.dim() != 4:
            raise ValueError(
                "Expected input shape [B, C_in, H, W], "
                f"got {tuple(X.shape)}."
            )

        B, C_in, H, W = X.shape
        patch_h, patch_w = self.patch_size

        if C_in != self.in_channel:
            raise ValueError(
                f"Expected {self.in_channel} input channels, "
                f"got {C_in}."
            )

        if H % patch_h != 0 or W % patch_w != 0:
            raise ValueError(
                "Input spatial size must be divisible by patch_size, "
            )
            
        #X:B,Cin,H,W
        num_patch_h=H//patch_h
        num_patch_w=W//patch_w
        X=X.reshape(B,C_in,num_patch_h,patch_h,num_patch_w,patch_w)
        #X:B,Cin,num_patch_h,patch_h,num_patch_w,patch_w
       
        X=X.permute(0,2,4,1,3,5)
        #X:B,num_patch_h,num_patch_w,Cin,patch_h,patch_w
    
        N= num_patch_h* num_patch_w
        X=X.reshape(B,N,C_in,patch_h,patch_w)
        #X:B,N,Cin,patch_h,patch_w

        X = X.reshape(B * N,C_in,patch_h,patch_w)

        X = self.W_embedding(X)
        X = self.W_squeeze(X)

       
        output = X.reshape(B,N,X.shape[1],X.shape[2],X.shape[3])

        return output





class UnsharedConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,feature_size,
                 kernel_size,stride= (1, 1),padding=(1, 1),
                 bias= True):
        
        super().__init__()

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)

        self.feature_h, self.feature_w = _to_2tuple(feature_size,"feature_size")

        # 允许从外部设置以下卷积参数。
        self.kernel_size = _to_2tuple(kernel_size, "kernel_size")
        self.stride = _to_2tuple(stride,"stride")
        self.padding = _to_2tuple(padding,"padding")

        if self.in_channels <= 0:
            raise ValueError(
                "in_channels must be positive, "
            )

        if self.out_channels <= 0:
            raise ValueError(
                "out_channels must be positive, "
            )

        if self.feature_h <= 0 or self.feature_w <= 0:
            raise ValueError(
                "feature_size must be positive, "
            )

        if self.kernel_size[0] <= 0 or self.kernel_size[1] <= 0:
            raise ValueError(
                "kernel_size must be positive, "
            )

        if self.stride[0] <= 0 or self.stride[1] <= 0:
            raise ValueError(
                "stride must be positive, "
            )

        if self.padding[0] < 0 or self.padding[1] < 0:
            raise ValueError(
                "padding must be non-negative, "
            )


        self.output_h = self._conv_output_size(
            input_size=self.feature_h,
            kernel_size=self.kernel_size[0],
            stride=self.stride[0],
            padding=self.padding[0],
        )

        self.output_w = self._conv_output_size(
            input_size=self.feature_w,
            kernel_size=self.kernel_size[1],
            stride=self.stride[1],
            padding=self.padding[1],
        )

        if self.output_h <= 0 or self.output_w <= 0:
            raise ValueError(
                "UnsharedConv2d produces a non-positive output size. "
            )

    

        self.num_positions = (
            self.output_h * self.output_w
        )

        self.Fan_In = (
            self.in_channels
            * self.kernel_size[0]
            * self.kernel_size[1]
        )
        
        self.init_bound=1/self.Fan_In

    
        self.weight = nn.Parameter(
torch.empty(self.output_h,self.output_w,self.Fan_In,self.out_channels).uniform_(-self.init_bound,self.init_bound)
        )

        if bias:
            self.bias = nn.Parameter(
    torch.empty(self.output_h,self.output_w,1,self.out_channels).uniform_(-self.init_bound,self.init_bound)
            )
        else:
            self.register_parameter(
                "bias",
                None,
            )


    @staticmethod
    def _conv_output_size(input_size,kernel_size,stride,padding):
        return (
            input_size
            + 2 * padding
            -  kernel_size 
        ) // stride + 1

    

    def forward(self,x):
        if x.dim() != 4:
            raise ValueError(
                "Expected input shape [B, C_in, H, W], "
                f"got {tuple(x.shape)}."
            )

        padding_h,padding_w=self.padding



       
        
        B, C_in, H, W = x.shape

        if C_in != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, "
                f"got {C_in}."
            )

        if (H, W) != (
            self.feature_h,
            self.feature_w,
        ):
            raise ValueError("Input feature size does not match configured ")
        
        x=F.pad(x, (padding_w, padding_w, padding_h, padding_h), mode="constant", value=0)
      
        #X:B,Cin,H_padding,W_padding
        x=x.unfold(dimension=2, size=self.kernel_size[0], step=self.stride[0])
        #X:B,Cin,H_out,W_padding,kernel_h
        x=x.unfold(dimension=3, size=self.kernel_size[1], step=self.stride[1])
        #X:B,Cin,H_out,W_out,kernel_h,kernel_w
        x=x.permute(2,3,0,1,4,5)
        #X:H_out,W_out,B,Cin,kernel_h,kernel_w
        x=x.reshape(self.output_h,self.output_w,B,self.Fan_In)
        #X:H_out,W_out,B,Cin*kernel_h*kernel_w
       
        output = torch.matmul(x,self.weight)

        #output:H_out,W_out,B,Cout


        if self.bias is not None:
            output = output + self.bias

       
        output=output.permute(2,3,0,1)
       

        return output
    
    
class TSSUSCP_ViT_MultiheadAttention(nn.Module):
    def __init__(self,squeezed_patch_size,embed_channel,num_heads):
        super().__init__()

        self.squeezed_patch_size = _to_2tuple(squeezed_patch_size,"squeezed_patch_size")

        self.embed_channel = int(embed_channel)
        self.num_heads = int(num_heads)

        if self.embed_channel <= 0:
            raise ValueError(
                "embed_channel must be positive, "
                f"got {self.embed_channel}."
            )

        if self.num_heads <= 0:
            raise ValueError(
                "num_heads must be positive, "
                f"got {self.num_heads}."
            )

        if self.embed_channel % self.num_heads != 0:
            raise ValueError(
                "embed_channel must be divisible by num_heads."
            )

        self.head_channel = self.embed_channel // self.num_heads
        
        self.Wq = UnsharedConv2d(
            in_channels=self.embed_channel,
            out_channels=self.embed_channel,
            feature_size=self.squeezed_patch_size,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )

        self.Wk = UnsharedConv2d(
            in_channels=self.embed_channel,
            out_channels=self.embed_channel,
            feature_size=self.squeezed_patch_size,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )

        self.Wv = UnsharedConv2d(
            in_channels=self.embed_channel,
            out_channels=self.embed_channel,
            feature_size=self.squeezed_patch_size,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )

        self.Wo = UnsharedConv2d(
            in_channels=self.embed_channel,
            out_channels=self.embed_channel,
            feature_size=self.squeezed_patch_size,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )

    def forward(self,X,Y1,Y2):
        if X.dim() != 5:
            raise ValueError(
                "X must have shape [B, N, C, H, W], "
                f"got {tuple(X.shape)}."
            )

        if Y1.dim() != 5:
            raise ValueError(
                "Y1 must have shape [B, N, C, H, W], "
                f"got {tuple(Y1.shape)}."
            )

        if Y2.dim() != 5:
            raise ValueError(
                "Y2 must have shape [B, N, C, H, W], "
                f"got {tuple(Y2.shape)}."
            )

        B, N_X, C, token_h, token_w = X.shape
        B_y1, N_Y1, C_y1, token_h_y1, token_w_y1 = Y1.shape
        B_y2, N_Y2, C_y2, token_h_y2, token_w_y2 = Y2.shape

        if B_y1 != B or B_y2 != B:
            raise ValueError(
                "Batch size of X, Y1 and Y2 must be the same."
            )

        if N_Y1 != N_Y2:
            raise ValueError(
                "Token number of Y1 and Y2 must be the same."
            )

        if C != self.embed_channel:
            raise ValueError(
                f"Expected X channel count {self.embed_channel}, "
                f"got {C}."
            )

        if C_y1 != C or C_y2 != C:
            raise ValueError(
                "Channel count of X, Y1 and Y2 must be the same."
            )

        if (token_h, token_w) != self.squeezed_patch_size:
            raise ValueError(
                "X spatial size does not match squeezed_patch_size: "
                f"expected {self.squeezed_patch_size}, "
                f"got {(token_h, token_w)}."
            )

        if (token_h_y1, token_w_y1) != (token_h,token_w):
            raise ValueError(
                "Spatial size of X and Y1 must be the same."
            )

        if (token_h_y2, token_w_y2) != (token_h,token_w):
            raise ValueError(
                "Spatial size of X and Y2 must be the same."
            )

        N_Y = N_Y1

        X = X.reshape(B * N_X,C,token_h,token_w)

        Y1 = Y1.reshape(B * N_Y,C,token_h,token_w)

        Y2 = Y2.reshape(B * N_Y,C,token_h,token_w)

        Q = self.Wq(X)
        K = self.Wk(Y1)
        V = self.Wv(Y2)

        Q = Q.reshape(B,N_X,C,token_h,token_w)

        K = K.reshape(B,N_Y,C,token_h,token_w)

        V = V.reshape(B,N_Y,C,token_h,token_w)

        Q = Q.reshape(B,N_X,self.num_heads,self.head_channel,token_h,token_w)
        K = K.reshape(B,N_Y,self.num_heads,self.head_channel,token_h,token_w)
        V = V.reshape(B,N_Y,self.num_heads,self.head_channel,token_h,token_w)

        Q = Q.transpose(1,2)
        K = K.transpose(1,2)
        V = V.transpose(1,2)

        Q = Q.reshape(B,self.num_heads,N_X,self.head_channel * token_h * token_w)
        K = K.reshape(B,self.num_heads,N_Y,self.head_channel * token_h * token_w)
        V = V.reshape(B,self.num_heads,N_Y,self.head_channel * token_h * token_w)

        output = F.scaled_dot_product_attention(Q,K,V,attn_mask=None,dropout_p=0.0,is_causal=False)
        output = output.reshape(B,self.num_heads,N_X,self.head_channel,token_h,token_w)
        output = output.transpose(1,2)
        
        output = output.reshape(B,N_X,C,token_h,token_w)
        output = output.reshape(B * N_X,C,token_h,token_w)
        output = self.Wo(output)
        output = output.reshape(B,N_X,C,token_h,token_w)

        return output
    
class TSSUSCP_ViT_Encoder_Block(nn.Module):
    def __init__(self,squeezed_patch_size,embed_channel,num_heads,ffn_channel):
        super().__init__()

        self.squeezed_patch_size = _to_2tuple(squeezed_patch_size,"squeezed_patch_size")
        self.embed_channel = int(embed_channel)
        self.ffn_channel = int(ffn_channel)

        self.TSSUSCP_ViT_attention=TSSUSCP_ViT_MultiheadAttention(
                squeezed_patch_size=self.squeezed_patch_size,
                embed_channel=self.embed_channel,
                num_heads=num_heads,
            )
        

        self.layernorm_1 = nn.LayerNorm(normalized_shape=[
                self.embed_channel,
                self.squeezed_patch_size[0],
                self.squeezed_patch_size[1],
            ]
        )

        self.layernorm_2 = nn.LayerNorm(
            normalized_shape=[
                self.embed_channel,
                self.squeezed_patch_size[0],
                self.squeezed_patch_size[1],
            ]
        )

        self.ffn = nn.Sequential(
            UnsharedConv2d(
                in_channels=self.embed_channel,
                out_channels=self.ffn_channel,
                feature_size=self.squeezed_patch_size,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
            ),
            nn.GELU(),
            UnsharedConv2d(
                in_channels=self.ffn_channel,
                out_channels=self.embed_channel,
                feature_size=self.squeezed_patch_size,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
            ),
        )

    def forward(self,X):
        if X.dim() != 5:
            raise ValueError(
                "Expected input shape [B, N, C, H, W], "
                f"got {tuple(X.shape)}."
            )

        B, N, C, H, W = X.shape

        if C != self.embed_channel:
            raise ValueError(
                f"Expected {self.embed_channel} channels, "
                f"got {C}."
            )

        if (H, W) != self.squeezed_patch_size:
            raise ValueError(
                "Input spatial size does not match "
                "squeezed_patch_size: "
            )

        ln1 = self.layernorm_1(X)
        X =self.TSSUSCP_ViT_attention(ln1,ln1,ln1)+X
        
        ln2 = self.layernorm_2(X)
        ffn_input = ln2.reshape(B * N,self.embed_channel,self.squeezed_patch_size[0],self.squeezed_patch_size[1])
        ffn_output = self.ffn(ffn_input)
        ffn_output = ffn_output.reshape(B,N,self.embed_channel,self.squeezed_patch_size[0],self.squeezed_patch_size[1])

        output = ffn_output + X

        return output
    
class TSSUSCP_ViT_Decoder_Block(nn.Module):
    """
    Decoder Transformer block.

    与 encoder 使用同构的注意力投影和 FFN；每个实例独立初始化，
    不与 encoder 或其他 decoder block 共享参数。

    X1: 生成 Q 的前一级输出，形状为 [B, T_q, C, h, w]。
        n=1 时为最后一层 encoder 的输出 E_M；n>1 时为 D_{n-1}。
    X2: 生成 K/V 的 encoder 输出 E_{N-n+1}，形状为 [B, T_kv, C, h, w]。

    流程：
        ln1 = layernorm_1(X1)
        ln2 = layernorm_2(X2)
        X = MHA(ln1, ln2, ln2) + X1
        ln3 = layernorm_3(X)
        output = reshape(FFN(reshape(ln3))) + X

    继承仅复用模块构造代码，不调用 encoder 的 forward。
    decoder 的 layernorm_2 用于 K/V 分支；layernorm_3 用于 FFN。
    """

    def __init__(self,squeezed_patch_size,embed_channel,num_heads,ffn_channel):
        super().__init__()
        
        self.squeezed_patch_size = _to_2tuple(squeezed_patch_size,"squeezed_patch_size")
        self.embed_channel = int(embed_channel)
        self.ffn_channel = int(ffn_channel)
        
        self.TSSUSCP_ViT_attention=TSSUSCP_ViT_MultiheadAttention(
                squeezed_patch_size=self.squeezed_patch_size,
                embed_channel=self.embed_channel,
                num_heads=num_heads,
            )
        
        self.layernorm_1 = nn.LayerNorm(normalized_shape=[
                self.embed_channel,
                self.squeezed_patch_size[0],
                self.squeezed_patch_size[1],
            ]
        )

        self.layernorm_2 = nn.LayerNorm(
            normalized_shape=[
                self.embed_channel,
                self.squeezed_patch_size[0],
                self.squeezed_patch_size[1],
            ]
        )

        self.layernorm_3 = nn.LayerNorm(
            normalized_shape=[
                self.embed_channel,
                self.squeezed_patch_size[0],
                self.squeezed_patch_size[1],
            ]
        )
        
        self.ffn = nn.Sequential(
            UnsharedConv2d(
                in_channels=self.embed_channel,
                out_channels=self.ffn_channel,
                feature_size=self.squeezed_patch_size,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
            ),
            nn.GELU(),
            UnsharedConv2d(
                in_channels=self.ffn_channel,
                out_channels=self.embed_channel,
                feature_size=self.squeezed_patch_size,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
            ),
        )

    def forward(self,X1,X2):
        for name, tensor in (("X1", X1), ("X2", X2)):
            if tensor.dim() != 5:
                raise ValueError(
                    f"{name} must have shape [B, T, C, h, w], "
                    f"got {tuple(tensor.shape)}."
                )
            if tensor.shape[2] != self.embed_channel:
                raise ValueError(
                    f"Expected {name} channel count {self.embed_channel}, "
                    f"got {tensor.shape[2]}."
                )
            if tuple(tensor.shape[-2:]) != self.squeezed_patch_size:
                raise ValueError(
                    f"{name} spatial size must be {self.squeezed_patch_size}, "
                    f"got {tuple(tensor.shape[-2:])}."
                )

        if X1.shape[0] != X2.shape[0]:
            raise ValueError("X1 and X2 must have the same batch size.")

        B, N, C, H, W = X1.shape

        # Q 和 K/V 分支分别归一化，使用两套独立的 LayerNorm 参数。
        ln1 = self.layernorm_1(X1)
        ln2 = self.layernorm_2(X2)

        # 注意力残差连接到 Q 的来源 X1，而不是 K/V 的来源 X2。
        X = self.TSSUSCP_ViT_attention(ln1, ln2, ln2) + X1

        ln3 = self.layernorm_3(X)
        ffn_input = ln3.reshape(B * N, C, H, W)
        ffn_output = self.ffn(ffn_input)
        ffn_output = ffn_output.reshape(B, N, C, H, W)
        output = ffn_output + X

        return output
    
class TSSUSCP_ViT(nn.Module):
    def __init__(
        self,
        img_size=(224, 224),
        patch_size=(16, 16),
        squeeze_rate=(4, 4),
        in_channel=3,
        embed_channel=72,
        num_heads=12,
        num_encoder_layers=12,
        ffn_channel=288,
        num_classes=7,
        num_decoder_layers=12,
    ):
        super().__init__()

        self.img_size = _to_2tuple(img_size, "img_size")
        self.patch_size = _to_2tuple(patch_size, "patch_size")
        self.squeeze_rate = _to_2tuple(squeeze_rate, "squeeze_rate")

        self.in_channel = int(in_channel)
        self.embed_channel = int(embed_channel)
        self.num_heads = int(num_heads)
        self.num_encoder_layers = int(num_encoder_layers)
        self.num_decoder_layers = int(num_decoder_layers)
        self.ffn_channel = int(ffn_channel)
        self.num_classes = int(num_classes)

        H, W = self.img_size
        patch_h, patch_w = self.patch_size
        squeeze_h, squeeze_w = self.squeeze_rate

        if H <= 0 or W <= 0:
            raise ValueError(
                f"img_size must be positive."
            )

        if patch_h <= 0 or patch_w <= 0:
            raise ValueError(
                "patch_size must be positive. "
            )

        if squeeze_h <= 0 or squeeze_w <= 0:
            raise ValueError(
                "squeeze_rate must be positive. "
            )

        if H % patch_h != 0 or W % patch_w != 0:
            raise ValueError(
                "img_size must be divisible by patch_size "
                "in both dimensions."
            )

        if squeeze_h > patch_h or squeeze_w > patch_w:
            raise ValueError(
                "squeeze_rate must be less than or equal to "
                "patch_size in both dimensions."
            )

        if (
            patch_h % squeeze_h != 0
            or patch_w % squeeze_w != 0
        ):
            raise ValueError(
                "patch_size must be divisible by squeeze_rate "
                "in both dimensions."
            )

        if self.embed_channel <= 0:
            raise ValueError(
                "embed_channel must be positive. "
            )

        if self.num_heads <= 0:
            raise ValueError(
                "num_heads must be positive."
            )

        if self.embed_channel % self.num_heads != 0:
            raise ValueError(
                "embed_channel must be divisible by num_heads."
            )

        if self.num_encoder_layers <= 0:
            raise ValueError(
                "num_encoder_layers must be positive."
            )

        if self.num_decoder_layers <= 0:
            raise ValueError(
                "num_decoder_layers must be positive."
            )

        if self.num_encoder_layers < self.num_decoder_layers:
            raise ValueError(
                "num_encoder_layers must be >= num_decoder_layers."
            )

        if self.ffn_channel <= 0:
            raise ValueError(
                "ffn_channel must be positive."
            )

        if self.num_classes <= 0:
            raise ValueError(
                "num_classes must be positive."
            )

        self.num_patch_h = H // patch_h
        self.num_patch_w = W // patch_w
        # T 表示 patch/token 数量，与 decoder 层数 N 区分。
        self.num_patches = self.num_patch_h * self.num_patch_w
        self.N = self.num_patches  # 保留原代码的 patch 数量属性。

        self.squeezed_patch_size = (
            patch_h // squeeze_h,
            patch_w // squeeze_w,
        )

        self.TSSUSCP_ViT_embedding = TSSUSCP_ViT_PatchEmbedding(
            patch_size=self.patch_size,
            squeeze_rate=self.squeeze_rate,
            in_channel=self.in_channel,
            embed_channel=self.embed_channel,
        )

        self.positional_encoding = nn.Parameter(
            torch.zeros(
                1,
                self.num_patches,
                self.embed_channel,
                self.squeezed_patch_size[0],
                self.squeezed_patch_size[1],
            )
        )
        
        self.TSSUSCP_ViT_encoder_block_list = nn.ModuleList()
        for _ in range(self.num_encoder_layers):
            self.TSSUSCP_ViT_encoder_block_list.append(TSSUSCP_ViT_Encoder_Block(
                squeezed_patch_size=self.squeezed_patch_size,embed_channel=self.embed_channel, num_heads=self.num_heads, ffn_channel=self.ffn_channel))
        
        self.TSSUSCP_ViT_decoder_block_list = nn.ModuleList()
        for _ in range(self.num_decoder_layers):
            self.TSSUSCP_ViT_decoder_block_list.append(TSSUSCP_ViT_Decoder_Block(
                squeezed_patch_size=self.squeezed_patch_size,embed_channel=self.embed_channel, num_heads=self.num_heads, ffn_channel=self.ffn_channel
                ))

        # 普通共享卷积；必须在拼图和恢复原图尺寸之后执行。
        self.segmentation_head = nn.Conv2d(
            in_channels=self.embed_channel,
            out_channels=self.num_classes,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def _tokens_to_feature_map(self,X):
     
        if X.dim() != 5:
            raise ValueError(
                "Decoder tokens must have shape [B, N, C, h, w], "
                f"got {tuple(X.shape)}."
            )

        B, N, C, h, w = X.shape

        if N != self.num_patches:
            raise ValueError(
                f"Expected {self.num_patches} tokens, got {N}."
            )
        if C != self.embed_channel:
            raise ValueError(
                f"Expected {self.embed_channel} token channels, got {C}."
            )
        if (h, w) != self.squeezed_patch_size:
            raise ValueError(
                "Decoder token spatial size does not match "
                "squeezed_patch_size: "
                f"expected {self.squeezed_patch_size}, got {(h, w)}."
            )

        # [B, n, C, h, w] -> [B,num_patch_h,num_patch_w, C, h, w]
        
        X=X.reshape(B,self.num_patch_h,self.num_patch_w,C,h,w)
       
        X=X.permute(0,3,1,4,2,5)
        #[B,C,num_patch_h,h,num_patch_w,w]
        X=X.reshape(B,C,self.num_patch_h*h,self.num_patch_w*w)
        #X:B,C,H,W
        
        return X
        
        
    def forward(self,X):
        if X.dim() != 4:
            raise ValueError(
                "Input must be a 4D tensor [B, C, H, W], "
                f"got {tuple(X.shape)}."
            )

        B, C_in, H, W = X.shape

        if (H, W) != self.img_size:
            raise ValueError(
                f"Input image size {(H, W)} does not match "
                f"model img_size {self.img_size}."
            )

        if C_in != self.in_channel:
            raise ValueError(
                f"Input channel count {C_in} does not match "
                f"model in_channel {self.in_channel}."
            )

        # ---- Encoder ----
        X = self.TSSUSCP_ViT_embedding(X)


        X = X + self.positional_encoding

        # ---- Encoder: 原 block 的计算流程不变 ----
        encoder_outputs = []
        for block in self.TSSUSCP_ViT_encoder_block_list:
            X = block(X)
            encoder_outputs.append(X)  # E_1, E_2, ..., E_M；不 detach。

        # ---- Transformer decoder ----
        # 第一个 decoder 的 Q 来源为最后一个 encoder 的输出 E_M。
        decoder_output = encoder_outputs[-1]

        for n, block in enumerate(self.TSSUSCP_ViT_decoder_block_list, start=1):
            # Q 来源于前一级输出：n=1 时为 E_M，n>1 时为 D_{n-1}。
            X1 = decoder_output
            # K/V 来源于 E_{N-n+1}；Python 零基索引为 N-n。
            X2 = encoder_outputs[self.num_decoder_layers - n]
            decoder_output = block(X1, X2)

        # ---- 拼回完整的低分辨率特征图 ----
        X = self._tokens_to_feature_map(decoder_output)

        # 先双线性插值恢复原图尺寸，再用普通 3x3 卷积调整类别通道。
        X = F.interpolate(
            X,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )
        logits = self.segmentation_head(X)

        # Return raw logits. No softmax is applied in the model.
        return logits
    
    
    
    
class TSSUSCP_ViT_pretrain(nn.Module):
    def __init__(
        self,
        img_size=(224, 224),
        patch_size=(16, 16),
        squeeze_rate=(4, 4),
        in_channel=3,
        embed_channel=72,
        num_heads=12,
        num_encoder_layers=12,
        ffn_channel=288,
        num_classes=1000,
        num_decoder_layers=12,
    ):
        super().__init__()

        self.img_size = _to_2tuple(img_size, "img_size")
        self.patch_size = _to_2tuple(patch_size, "patch_size")
        self.squeeze_rate = _to_2tuple(squeeze_rate, "squeeze_rate")

        self.in_channel = int(in_channel)
        self.embed_channel = int(embed_channel)
        self.num_heads = int(num_heads)
        self.num_encoder_layers = int(num_encoder_layers)
        self.ffn_channel = int(ffn_channel)
        self.num_classes = int(num_classes)

        H, W = self.img_size
        patch_h, patch_w = self.patch_size
        squeeze_h, squeeze_w = self.squeeze_rate

        if H <= 0 or W <= 0:
            raise ValueError(
                f"img_size must be positive."
            )

        if patch_h <= 0 or patch_w <= 0:
            raise ValueError(
                "patch_size must be positive. "
            )

        if squeeze_h <= 0 or squeeze_w <= 0:
            raise ValueError(
                "squeeze_rate must be positive. "
            )

        if H % patch_h != 0 or W % patch_w != 0:
            raise ValueError(
                "img_size must be divisible by patch_size "
                "in both dimensions."
            )

        if squeeze_h > patch_h or squeeze_w > patch_w:
            raise ValueError(
                "squeeze_rate must be less than or equal to "
                "patch_size in both dimensions."
            )

        if (
            patch_h % squeeze_h != 0
            or patch_w % squeeze_w != 0
        ):
            raise ValueError(
                "patch_size must be divisible by squeeze_rate "
                "in both dimensions."
            )

        if self.embed_channel <= 0:
            raise ValueError(
                "embed_channel must be positive. "
            )
        
        
        if self.num_encoder_layers <= 0:
            raise ValueError(
                "num_encoder_layers must be positive."
            )

        if self.num_heads <= 0:
            raise ValueError(
                "num_heads must be positive."
            )

        if self.embed_channel % self.num_heads != 0:
            raise ValueError(
                "embed_channel must be divisible by num_heads."
            )


        if self.ffn_channel <= 0:
            raise ValueError(
                "ffn_channel must be positive."
            )

        if self.num_classes <= 0:
            raise ValueError(
                "num_classes must be positive."
            )

        self.num_patch_h = H // patch_h
        self.num_patch_w = W // patch_w
        # T 表示 patch/token 数量，与 decoder 层数 N 区分。
        self.num_patches = self.num_patch_h * self.num_patch_w
        self.N = self.num_patches  # 保留原代码的 patch 数量属性。

        self.squeezed_patch_size = (
            patch_h // squeeze_h,
            patch_w // squeeze_w,
        )

        self.TSSUSCP_ViT_embedding = TSSUSCP_ViT_PatchEmbedding(
            patch_size=self.patch_size,
            squeeze_rate=self.squeeze_rate,
            in_channel=self.in_channel,
            embed_channel=self.embed_channel,
        )

        self.positional_encoding = nn.Parameter(
            torch.zeros(
                1,
                self.num_patches,
                self.embed_channel,
                self.squeezed_patch_size[0],
                self.squeezed_patch_size[1],
            )
        )
        
        self.TSSUSCP_ViT_encoder_block_list = nn.ModuleList()
        for _ in range(self.num_encoder_layers):
            self.TSSUSCP_ViT_encoder_block_list.append(TSSUSCP_ViT_Encoder_Block(
                squeezed_patch_size=self.squeezed_patch_size,embed_channel=self.embed_channel, num_heads=self.num_heads, ffn_channel=self.ffn_channel))
        
        self.classifier = nn.Linear(self.embed_channel*self.squeezed_patch_size[0]*self.squeezed_patch_size[1],self.num_classes)
    
    
    def forward(self,X):
        if X.dim() != 4:
            raise ValueError(
                "Input must be a 4D tensor [B, C, H, W], "
                f"got {tuple(X.shape)}."
            )

        B, C_in, H, W = X.shape

        if (H, W) != self.img_size:
            raise ValueError(
                f"Input image size {(H, W)} does not match "
                f"model img_size {self.img_size}."
            )

        if C_in != self.in_channel:
            raise ValueError(
                f"Input channel count {C_in} does not match "
                f"model in_channel {self.in_channel}."
            )

        # ---- Encoder ----
        X = self.TSSUSCP_ViT_embedding(X)


        X = X + self.positional_encoding

        # ---- Encoder: 原 block 的计算流程不变 ----
        encoder_outputs = []
        for block in self.TSSUSCP_ViT_encoder_block_list:
            X = block(X)
           
       
        X = X.mean(dim=1,keepdim=False)

        X = X.reshape(B,self.embed_channel*self.squeezed_patch_size[0]*self.squeezed_patch_size[1])

        X = self.classifier(X)

        return X
    
    
    
    
    
    
    
    
    
    
    