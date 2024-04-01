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
def gemm_64_64_64(x, y, d):
    token_ty = ir.Type.parse("!gpu.async.token")
    t1 = gpu.wait(token_ty, [])
    a_dev, t2 = gpu.alloc(x.type, token_ty, [t1], [], [])
    b_dev, t3 = gpu.alloc(y.type, token_ty, [t2], [], [])
    d_dev, t4 = gpu.alloc(d.type, token_ty, [t3], [], [])
    t5 = gpu.memcpy(token_ty, [t4], a_dev, x)
    t6 = gpu.memcpy(token_ty, [t5], b_dev, y)
    t7 = gpu.wait(token_ty, [t6])

    sw = nvgpu.TensorMapSwizzleKind.SWIZZLE_128B
    a_tma = TMA([N, N], x.type, swizzle=sw)
    b_tma = TMA([N, N], y.type, swizzle=sw)
    a_tma.create_descriptor(a_dev)
    b_tma.create_descriptor(b_dev)
    smem_size_in_bytes = get_type_size(x.type) + get_type_size(y.type)

    @NVDSL.mlir_gpu_launch(grid=(1, 1, 1), block=(128, 1, 1), smem=smem_size_in_bytes)
    def gemm_tma_kernel():
        tidx = gpu.thread_id(gpu.Dimension.x)

        mbar_group = Mbarriers(number_of_barriers=1)
        isThread0 = tidx == 0
        with ir.InsertionPoint(scf.IfOp(isThread0).then_block):
            mbar_group[0].init(1)
            a_tma.prefetch()
            b_tma.prefetch()
            scf.yield_([])

        a_smem = get_dynamic_shared_memory((N, N), T.f16())
        b_smem = get_dynamic_shared_memory((N, N), T.f16(), offset=N * N * 2)

        # 1. Execute TMA Load for two input matrices
        with ir.InsertionPoint(scf.IfOp(isThread0).then_block):
            a_tma.load(a_smem, mbar_group[0])
            b_tma.load(b_smem, mbar_group[0])
            ta_count = get_type_size(a_tma.tma_memref) + get_type_size(b_tma.tma_memref)
            mbar_group[0].arrive(ta_count)
            scf.yield_([])

        mbar_group[0].try_wait()

        # 2. Performs Tensor Core GEMM 64x64x64 by warpgroup
        A = WarpgroupMatrix(a_smem, a_tma, N, N)
        B = WarpgroupMatrix(b_smem, b_tma, N, N)
        C = WarpgroupAccumulatorMatrix(N, N, T.f32()).op()
        D = WarpgroupMatrix.matmul(A, B, C)

        # 3. Stores fragmented registers to global memory by warpgroup
        nvgpu.warpgroup_mma_store(D, d_dev)

    gemm_tma_kernel()

    t8 = gpu.memcpy(token_ty, [t7], d, d_dev)
    gpu.wait(None, [t8])


# Python pass arguments to MLIR
N = 64
a = np.random.randn(N, N).astype(np.float16)
b = np.random.randn(N, N).astype(np.float16)
d = np.zeros((N, N), np.float32)
gemm_64_64_64(a, b, d)

ref_d = a.astype(np.float16) @ b.astype(np.float16)
np.testing.assert_allclose(d, ref_d, rtol=5e-03, atol=1e-01)
print("PASS")
# CHECK-NOT: Mismatched elements
