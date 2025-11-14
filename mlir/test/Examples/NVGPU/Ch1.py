# RUN: env SUPPORT_LIB=%mlir_cuda_runtime \
# RUN:   %PYTHON %s | FileCheck %s

# ===----------------------------------------------------------------------===//
#  Chapter 1 : 2D Saxpy
# ===----------------------------------------------------------------------===//
#
# This program demonstrates 2D Saxpy:
#  1. Use GPU dialect to allocate and copy memory host to gpu and vice versa
#  2. Computes 2D SAXPY kernel using operator overloading
#  3. Pass numpy arrays to MLIR as memref arguments
#  4. Verify MLIR program with reference computation in python
#
# ===----------------------------------------------------------------------===//


from mlir import ir
from mlir.dialects import gpu, memref
from tools.nvdsl import *
import numpy as np


@NVDSL.mlir_func
def saxpy(x, y, alpha):
    # 1. Use MLIR GPU dialect to allocate and copy memory
    token_ty = gpu.AsyncTokenType.get()
    t1 = gpu.wait(token_ty, [])
    x_dev, t2 = gpu.alloc(x.type, token_ty, [t1], [], [])
    y_dev, t3 = gpu.alloc(y.type, token_ty, [t2], [], [])
    t4 = gpu.memcpy(token_ty, [t3], x_dev, x)
    t5 = gpu.memcpy(token_ty, [t4], y_dev, y)
    t6 = gpu.wait(token_ty, [t5])

    # 2. Compute 2D SAXPY kernel
    @NVDSL.mlir_gpu_launch(grid=(M, 1, 1), block=(N, 1, 1))
    def saxpy_kernel():
        bidx = gpu.block_id(gpu.Dimension.x)
        tidx = gpu.thread_id(gpu.Dimension.x)
        x_val = memref.load(x_dev, [bidx, tidx])
        y_val = memref.load(y_dev, [bidx, tidx])

        # SAXPY: y[i] += a * x[i];
        y_val += x_val * alpha

        memref.store(y_val, y_dev, [bidx, tidx])

    saxpy_kernel()

    t7 = gpu.memcpy(token_ty, [t6], y, y_dev)
    gpu.wait(token_ty, [t7])


# 3. Pass numpy arrays to MLIR
M = 256
N = 32
alpha = 2.0
x = np.random.randn(M, N).astype(np.float32)
y = np.ones((M, N), np.float32)
saxpy(x, y, alpha)

#  4. Verify MLIR with reference computation
ref = np.ones((M, N), np.float32)
ref += x * alpha
np.testing.assert_allclose(y, ref, rtol=5e-03, atol=1e-01)
print("PASS")
# CHECK-NOT: Mismatched elements
