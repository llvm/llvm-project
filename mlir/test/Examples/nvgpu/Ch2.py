# RUN: env SUPPORT_LIB=%mlir_cuda_runtime \
# RUN:   %PYTHON %s | FileCheck %s

# ===----------------------------------------------------------------------===//
#  Chapter 2 : 2D Saxpy with TMA
# ===----------------------------------------------------------------------===//
#
# This program demonstrates 2D Saxpy. It is same as Chapter 1,
# but it loads data using TMA (Tensor Memory Accelerator)
#
# This chapter introduces demonstrates:
#  1. Create and initialize asynchronous transactional barrier (mbarrier)
#  2. Execute Tensor Memory Accelerator (TMA) Load
#  3. Wait for completion of TMA load with mbarrier
#
# ===----------------------------------------------------------------------===//

from mlir import ir
from mlir.dialects import nvgpu, scf, arith, memref, vector, gpu
from tools.nvdsl import *
from mlir import runtime as rt
from mlir.extras import types as T
import numpy as np


@NVDSL.mlir_func
def saxpy_tma(x, y, alpha):
    token_ty = ir.Type.parse("!gpu.async.token")
    t1 = gpu.wait(token_ty, [])
    x_dev, t2 = gpu.alloc(x.type, token_ty, [t1], [], [])
    y_dev, t3 = gpu.alloc(y.type, token_ty, [t2], [], [])
    t4 = gpu.memcpy(token_ty, [t3], x_dev, x)
    t5 = gpu.memcpy(token_ty, [t4], y_dev, y)
    t6 = gpu.wait(token_ty, [t5])

    x_tma = TMA((M, N), x.type)
    y_tma = TMA((M, N), y.type)
    x_tma.create_descriptor(x_dev)
    y_tma.create_descriptor(y_dev)
    smem_size_in_bytes = get_type_size(x.type) + get_type_size(y.type)

    @NVDSL.mlir_gpu_launch(grid=(M, 1, 1), block=(N, 1, 1), smem=smem_size_in_bytes)
    def saxpy_tma_kernel():
        bidx = gpu.block_id(gpu.Dimension.x)
        tidx = gpu.thread_id(gpu.Dimension.x)
        isThread0 = tidx == 0

        # 1. Create and initialize asynchronous transactional barrier (mbarrier)
        mbar_group = Mbarriers(number_of_barriers=1)
        mbar_group[0].init(1, predicate=isThread0)

        # 2. Execute Tensor Memory Accelerator (TMA) Load
        x_smem = get_dynamic_shared_memory((M, N), T.f32())
        y_smem = get_dynamic_shared_memory((M, N), T.f32(), offset=M * N * 2)
        x_tma.load(x_smem, mbar_group[0], predicate=isThread0)
        y_tma.load(y_smem, mbar_group[0], predicate=isThread0)
        mbar_group[0].arrive(txcount=M * N * 2 * 4, predicate=isThread0)

        # 3. Wait for completion of TMA load with mbarrier
        mbar_group[0].try_wait()

        x_val = memref.load(x_smem, [bidx, tidx])
        y_val = memref.load(y_smem, [bidx, tidx])

        # SAXPY: y[i] += a * x[i];
        y_val += x_val * alpha

        memref.store(y_val, y_dev, [bidx, tidx])

    saxpy_tma_kernel()

    t7 = gpu.memcpy(token_ty, [t6], y, y_dev)
    gpu.wait(token_ty, [t7])


M = 256
N = 32
alpha = 2.0
x = np.ones((M, N), np.float32)
y = np.ones((M, N), np.float32)
ref = np.ones((M, N), np.float32)
saxpy_tma(x, y, alpha)

#  4. Verify MLIR with reference computation
ref += x * alpha
np.testing.assert_allclose(y, ref, rtol=5e-03, atol=1e-01)
print("PASS")
# CHECK-NOT: Mismatched elements
