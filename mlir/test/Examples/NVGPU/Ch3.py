# RUN: env SUPPORT_LIB=%mlir_cuda_runtime \
# RUN:   %PYTHON %s | FileCheck %s

# ===----------------------------------------------------------------------===//
#  Chapter 3 : GEMM 128x128x64 with Tensor Core
# ===----------------------------------------------------------------------===//
#
# This program demonstrates a GEMM operation with 128x128x64 matrix multiplication
#
# This chapter introduces demonstrates:
# 1. Execute TMA Load for two input matrices
# 2. Performs Tensor Core GEMM 128x128x64 by warpgroup
# 3. Stores fragmented registers to global memory by warpgroup
#
# ===----------------------------------------------------------------------===//


from mlir import ir
from mlir.dialects import nvgpu, scf, arith, memref, vector, gpu
from tools.nvdsl import *
from mlir.extras import types as T
import numpy as np


def tma_load(
    mbar_group: Mbarriers,
    a_tma: TMA,
    b_tma: TMA,
    p,
):
    """
    TMA loads two input matrices from global memory to shared memory. It performs the following operations:

       - tma.load a_shared_memory[0] at coordinate [0, 0]  (Loads 128x64)
       - tma.load b_shared_memory[0] at coordinate [0, 0]  (Loads 64x64)
       - tma.load b_shared_memory[0] at coordinate [64, 0] (Loads 64x64)

       mbarrier.arrive ta_count = 128x64xf16 + 64x128xf16
    """

    size_tma_a = get_type_size(a_tma.tma_memref)
    size_tma_b = get_type_size(b_tma.tma_memref)
    ta_count = size_tma_a + (size_tma_b * 2)

    off_b = size_tma_a
    off_b2 = off_b + size_tma_b
    a_elem_ty = a_tma.tma_memref.element_type
    b_elem_ty = b_tma.tma_memref.element_type
    a = get_dynamic_shared_memory(a_tma.tma_memref.shape, a_elem_ty)
    b1 = get_dynamic_shared_memory(b_tma.tma_memref.shape, b_elem_ty, off_b)
    b2 = get_dynamic_shared_memory(b_tma.tma_memref.shape, b_elem_ty, off_b2)

    mbar_group[0].arrive(ta_count, predicate=p)

    a_tma.load(a, mbar_group[0], coords=[0, 0], predicate=p)
    b_tma.load(b1, mbar_group[0], coords=[0, 0], predicate=p)
    b_tma.load(b2, mbar_group[0], coords=[64, 0], predicate=p)


@NVDSL.mlir_func
def gemm_128_128_64(a, b, d):
    token_ty = gpu.AsyncTokenType.get()
    t1 = gpu.wait(token_ty, [])
    a_dev, t2 = gpu.alloc(a.type, token_ty, [t1], [], [])
    b_dev, t3 = gpu.alloc(b.type, token_ty, [t2], [], [])
    d_dev, t4 = gpu.alloc(d.type, token_ty, [t3], [], [])
    t5 = gpu.memcpy(token_ty, [t4], a_dev, a)
    t6 = gpu.memcpy(token_ty, [t5], b_dev, b)
    t7 = gpu.wait(token_ty, [t6])

    sw = nvgpu.TensorMapSwizzleKind.SWIZZLE_128B
    a_tma = TMA([128, 64], a.type, swizzle=sw)
    b_tma = TMA([64, 64], b.type, swizzle=sw)
    a_tma.create_descriptor(a_dev)
    b_tma.create_descriptor(b_dev)
    a_size = get_type_size(a.type)
    b_size = get_type_size(b.type)
    smem_size_in_bytes = a_size + b_size

    @NVDSL.mlir_gpu_launch(grid=(1, 1, 1), block=(128, 1, 1), smem=smem_size_in_bytes)
    def gemm_tma_kernel():
        tidx = gpu.thread_id(gpu.Dimension.x)

        mbar_group = Mbarriers(number_of_barriers=1)
        isThread0 = tidx == 0

        mbar_group[0].init(1, predicate=isThread0)
        a_tma.prefetch(predicate=isThread0)
        b_tma.prefetch(predicate=isThread0)

        a_smem = get_dynamic_shared_memory((M, K), T.f16())
        b_smem = get_dynamic_shared_memory((K, N), T.f16(), offset=a_size)

        # 1. TMA Load for two input matrices
        tma_load(mbar_group, a_tma, b_tma, isThread0)

        # 2. All threads wait TMA load completion
        mbar_group[0].try_wait()

        # 3. Performs Tensor Core GEMM 128x128x64 by warpgroup
        A = WGMMAMatrix(WGMMAType.Descriptor, [M, K], desc=a_tma, smem=a_smem)
        B = WGMMAMatrix(WGMMAType.Descriptor, [K, N], desc=b_tma, smem=b_smem)
        D = WGMMAMatrix(WGMMAType.Accumulator, shape=[M, N], ty=T.f32())

        # Matrix Multiply
        D += A @ B

        # 4. Stores fragmented registers to global memory by warpgroup
        D.store_accumulator(d_dev)

    gemm_tma_kernel()

    t8 = gpu.memcpy(token_ty, [t7], d, d_dev)
    gpu.wait(None, [t8])


# Python pass arguments to MLIR
M = 128
N = 128
K = 64
a = np.random.randn(M, K).astype(np.float16)
b = np.random.randn(K, N).astype(np.float16)
d = np.zeros((M, N), np.float32)
gemm_128_128_64(a, b, d)

ref_d = a.astype(np.float16) @ b.astype(np.float16)
np.testing.assert_allclose(d, ref_d, rtol=5e-03, atol=1e-01)
print("PASS")
# CHECK-NOT: Mismatched elements
