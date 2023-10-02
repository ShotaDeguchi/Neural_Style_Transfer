"""
test to check PyTorch is working with MPS
"""

import torch

def main():
    print(f"PyTorch version: {torch.__version__}")

    # check if CUDA is available
    if torch.cuda.is_available():
        print("CUDA is available")
    else:
        print("CUDA is not available")

    # check if MPS is built
    if torch.backends.mps.is_built():
        print("MPS is built")
    else:
        print("MPS is not built")

    # check if MPS is available
    if torch.backends.mps.is_available():
        print("MPS is available")
    else:
        print("MPS is not available")

    # set device
    device = "mps" if torch.backends.mps.is_available() else "cuda"
    print(f"device: {device}")

    # create data and move to device
    x = torch.rand(1000, 1000)
    print(f"before sending to divice: x.device: {x.device}")
    x = x.to(device)
    print(f"after sending to divice : x.device: {x.device}")

if __name__ == "__main__":
    main()
