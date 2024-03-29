# RUN: env SUPPORT_LIB=%mlir_cuda_runtime \
# RUN:   %PYTHON %s | FileCheck %s

# ===----------------------------------------------------------------------===//
#  Chapter 3 : GEMM 64x64x64 with Tensor Core
# ===----------------------------------------------------------------------===//
#
# This program demonstrates a GEMM operation with 64x64x64 matrix multiplication
#
# This chapter introduces demonstrates:
# 1. Execute TMA Load for two input matrices
# 2. Performs Tensor Core GEMM 64x64x64 by warpgroup
# 3. Stores fragmented registers to global memory by warpgroup
#
# ===----------------------------------------------------------------------===//


from mlir import ir
from mlir.dialects import nvgpu, scf, arith, memref, vector, gpu
from tools.nvdsl import *
from mlir.extras import types as T
import numpy as np


@NVDSL.mlir_func
def gemm_64_64_64(x, y, z):
    token_ty = ir.Type.parse("!gpu.async.token")
    t1 = gpu.wait(token_ty, [])
    x_dev, t2 = gpu.alloc(x.type, token_ty, [t1], [], [])
    y_dev, t3 = gpu.alloc(y.type, token_ty, [t2], [], [])
    z_dev, t4 = gpu.alloc(z.type, token_ty, [t3], [], [])
    t5 = gpu.memcpy(token_ty, [t4], x_dev, x)
    t6 = gpu.memcpy(token_ty, [t5], y_dev, y)
    t7 = gpu.wait(token_ty, [t6])

    sw = nvgpu.TensorMapSwizzleKind.SWIZZLE_128B
    x_tma = TMA([N, N], x.type, swizzle=sw)
    y_tma = TMA([N, N], y.type, swizzle=sw)
    x_tma.create_descriptor(x_dev)
    y_tma.create_descriptor(y_dev)

    @NVDSL.mlir_gpu_launch(grid=(1, 1, 1), block=(128, 1, 1), smem=16384)
    def gemm_tma_kernel():
        tidx = gpu.thread_id(gpu.Dimension.x)

        mbar_group = Mbarriers(number_of_barriers=1)
        isThread0 = tidx == 0
        with ir.InsertionPoint(scf.IfOp(isThread0).then_block):
            mbar_group[0].init(1)
            x_tma.prefetch()
            y_tma.prefetch()
            scf.yield_([])

        x_smem = get_dynamic_shared_memory((N, N), T.f16())
        y_smem = get_dynamic_shared_memory((N, N), T.f16(), offset=N * N * 2)

        # 1. Execute TMA Load for two input matrices
        with ir.InsertionPoint(scf.IfOp(isThread0).then_block):
            x_tma.load(x_smem, mbar_group[0])
            y_tma.load(y_smem, mbar_group[0])
            tx_count = get_type_size(x_tma.tma_memref) + get_type_size(y_tma.tma_memref)
            mbar_group[0].arrive(tx_count)
            scf.yield_([])

        mbar_group[0].try_wait()

        # 2. Performs Tensor Core GEMM 64x64x64 by warpgroup
        A = Matrix(x_smem, x_tma, N, N)
        B = Matrix(y_smem, y_tma, N, N)
        C = MatrixAccumulator(N, N, T.f32()).op()
        D = Matrix.matmul(A, B, C)

        # 3. Stores fragmented registers to global memory by warpgroup
        nvgpu.warpgroup_mma_store(D, z_dev)

    gemm_tma_kernel()

    t8 = gpu.memcpy(token_ty, [t7], z, z_dev)
    gpu.wait(None, [t8])


# Python pass arguments to MLIR
N = 64
x = np.random.randn(N, N).astype(np.float16)
y = np.random.randn(N, N).astype(np.float16)
z = np.zeros((N, N), np.float32)
gemm_64_64_64(x, y, z)

ref = x.astype(np.float16) @ y.astype(np.float16)
np.testing.assert_allclose(z, ref, rtol=5e-03, atol=1e-01)
print("PASS")
# CHECK-NOT: Mismatched elements
