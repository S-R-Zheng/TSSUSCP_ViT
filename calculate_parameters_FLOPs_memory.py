import torch
from model_TSSUSCP_ViT import TSSUSCP_ViT
import segmentation_models_pytorch as smp
from torch.utils.flop_counter import FlopCounterMode

def bytes_to_mib(num_bytes):
    """Convert bytes to MiB."""
    return num_bytes / 1024 ** 2


if __name__ == "__main__":
    img_size = 1024
    num_classes = 7
    batch_size = 16
    patch_size = (32, 32)
    squeeze_rate = (8, 8)
    in_channel=3
    embed_channel=36
    num_heads=12
    num_encoder_layers=6
    num_decoder_layers=6
    ffn_channel=144
    num_classes=7

    model = TSSUSCP_ViT(
         img_size=img_size,
         patch_size=patch_size,
         squeeze_rate=squeeze_rate,
         in_channel=in_channel,
         embed_channel=embed_channel,
         num_heads=num_heads,
         num_encoder_layers=num_encoder_layers,
         num_decoder_layers=num_decoder_layers,
         ffn_channel=ffn_channel,
         num_classes=num_classes,
     )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This script requires a CUDA device to measure GPU memory.")

    model.to(device)
    model.eval()

    total = sum(p.numel() for p in model.parameters())
    print(f"parameter counts are {total / 1e6}M")

   
    x = torch.zeros(1, in_channel, img_size, img_size, device=device)

    
    with torch.no_grad():
        y = model(x)
    torch.cuda.synchronize(device)
    del y

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        y = model(x)
    torch.cuda.synchronize(device)

    peak_memory = torch.cuda.max_memory_allocated(device)
    print(f"peak memory is {bytes_to_mib(peak_memory)}MiB")

    del y
    torch.cuda.empty_cache()

   
   
    with FlopCounterMode(display=False, depth=None) as flop_counter:
        _ = model(x)

    FLOPs = flop_counter.get_total_flops()
    print(f"FLOPs are {FLOPs / 1e9} G")