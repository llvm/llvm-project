# RUN: env SUPPORT_LIB=%mlir_cuda_runtime \
# RUN: sh -c 'if [[ "%mlir_run_cuda_sm90_tests" == "1" ]]; \
# RUN: then %PYTHON %s | FileCheck %s; \
# RUN: else export MLIR_NVDSL_PRINT_IR=1; \
# RUN: %PYTHON %s | FileCheck %s --check-prefix=DUMPIR; fi'


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
    t1 = gpu.wait([])
    a_dev, t2 = gpu.alloc(a.type, token_ty, [t1], [], [])
    b_dev, t3 = gpu.alloc(b.type, token_ty, [t2], [], [])
    d_dev, t4 = gpu.alloc(d.type, token_ty, [t3], [], [])
    t5 = gpu.memcpy(token_ty, [t4], a_dev, a)
    t6 = gpu.memcpy(token_ty, [t5], b_dev, b)
    t7 = gpu.wait([t6])

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
    gpu.wait([t8])


# Python pass arguments to MLIR
M = 128
N = 128
K = 64
a = np.random.randn(M, K).astype(np.float16)
b = np.random.randn(K, N).astype(np.float16)
d = np.zeros((M, N), np.float32)
gemm_128_128_64(a, b, d)

if os.getenv("MLIR_NVDSL_PRINT_IR") != "1":
    # Verify MLIR program with reference computation in python
    ref_d = a.astype(np.float16) @ b.astype(np.float16)
    np.testing.assert_allclose(d, ref_d, rtol=5e-03, atol=1e-01)
    print("PASS")
# CHECK-NOT: Mismatched elements
# CHECK: PASS

# DUMPIR:   func.func @gemm_128_128_64(%{{.*}}: memref<128x64xf16>, %{{.*}}: memref<64x128xf16>, %[[ARG2:.*]]: memref<128x128xf32>) attributes {llvm.emit_c_interface} {
# DUMPIR:     %[[C128:.*]] = arith.constant 128 : index
# DUMPIR:     %[[C64:.*]] = arith.constant 64 : index
# DUMPIR:     %[[TMA0:.*]] = nvgpu.tma.create.descriptor %{{.*}} box[%[[C128]], %[[C64]]] : memref<*xf16> -> <tensor = memref<128x64xf16, 3>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none>
# DUMPIR:     %[[CAST1:.*]] = memref.cast %{{.*}} : memref<64x128xf16> to memref<*xf16>
# DUMPIR:     %[[C64_5:.*]] = arith.constant 64 : index
# DUMPIR:     %[[C64_6:.*]] = arith.constant 64 : index
# DUMPIR:     %[[TMA1:.*]] = nvgpu.tma.create.descriptor %[[CAST1]] box[%[[C64_5]], %[[C64_6]]] : memref<*xf16> -> <tensor = memref<64x64xf16, 3>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none>
# DUMPIR:       %[[THREADID:.*]] = gpu.thread_id  x
# DUMPIR:       %[[MB:.*]] = nvgpu.mbarrier.create -> <memorySpace = #gpu.address_space<workgroup>>
# DUMPIR:       %[[C0:.*]] = arith.constant 0 : index
# DUMPIR:       %[[EQ:.*]] = arith.cmpi eq, %[[THREADID]], %[[C0]] : index
# DUMPIR:       %[[C0_12:.*]] = arith.constant 0 : index
# DUMPIR:       %[[C1_13:.*]] = arith.constant 1 : index
# DUMPIR:       nvgpu.mbarrier.init %[[MB]][%[[C0_12]]], %[[C1_13]], predicate = %[[EQ]] : <memorySpace = #gpu.address_space<workgroup>>
# DUMPIR:       nvgpu.tma.prefetch.descriptor %[[TMA0]], predicate = %[[EQ]] : <tensor = memref<128x64xf16, 3>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none>
# DUMPIR:       nvgpu.tma.prefetch.descriptor %[[TMA1]], predicate = %[[EQ]] : <tensor = memref<64x64xf16, 3>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none>
# DUMPIR:       %[[DSM0:.*]] = gpu.dynamic_shared_memory : memref<?xi8, #gpu.address_space<workgroup>>
# DUMPIR:       %[[C0_14:.*]] = arith.constant 0 : index
# DUMPIR:       %[[VIEW:.*]] = memref.view %[[DSM0]][%[[C0_14]]][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<128x64xf16, #gpu.address_space<workgroup>>
# DUMPIR:       %[[DSM1:.*]] = gpu.dynamic_shared_memory : memref<?xi8, #gpu.address_space<workgroup>>
# DUMPIR:       %[[C16384:.*]] = arith.constant 16384 : index
# DUMPIR:       %[[VIEW_15:.*]] = memref.view %[[DSM1]][%[[C16384]]][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<64x128xf16, #gpu.address_space<workgroup>>
# DUMPIR:       %[[DSM2:.*]] = gpu.dynamic_shared_memory : memref<?xi8, #gpu.address_space<workgroup>>
# DUMPIR:       %[[C0_16:.*]] = arith.constant 0 : index
# DUMPIR:       %[[VIEW_17:.*]] = memref.view %[[DSM2]][%[[C0_16]]][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<128x64xf16, #gpu.address_space<workgroup>>
# DUMPIR:       %[[DSM3:.*]] = gpu.dynamic_shared_memory : memref<?xi8, #gpu.address_space<workgroup>>
# DUMPIR:       %[[C16384_18:.*]] = arith.constant 16384 : index
# DUMPIR:       %[[VIEW_19:.*]] = memref.view %[[DSM3]][%[[C16384_18]]][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<64x64xf16, #gpu.address_space<workgroup>>
# DUMPIR:       %[[DSM4:.*]] = gpu.dynamic_shared_memory : memref<?xi8, #gpu.address_space<workgroup>>
# DUMPIR:       %[[C24576:.*]] = arith.constant 24576 : index
# DUMPIR:       %[[VIEW_20:.*]] = memref.view %[[DSM4]][%[[C24576]]][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<64x64xf16, #gpu.address_space<workgroup>>
# DUMPIR:       %[[C0_21:.*]] = arith.constant 0 : index
# DUMPIR:       %[[C32768:.*]] = arith.constant 32768 : index
# DUMPIR:       nvgpu.mbarrier.arrive.expect_tx %[[MB]][%[[C0_21]]], %[[C32768]], predicate = %[[EQ]] : <memorySpace = #gpu.address_space<workgroup>>
# DUMPIR:       %[[C0_22:.*]] = arith.constant 0 : index
# DUMPIR:       %[[C0_23:.*]] = arith.constant 0 : index
# DUMPIR:       %[[C0_24:.*]] = arith.constant 0 : index
# DUMPIR:       nvgpu.tma.async.load %[[TMA0]][%[[C0_23]], %[[C0_24]]], %[[MB]][%[[C0_22]]] to %[[VIEW_17]], predicate = %[[EQ]] : <tensor = memref<128x64xf16, 3>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none>, <memorySpace = #gpu.address_space<workgroup>> -> memref<128x64xf16, #gpu.address_space<workgroup>>
# DUMPIR:       %[[C0_25:.*]] = arith.constant 0 : index
# DUMPIR:       %[[C0_26:.*]] = arith.constant 0 : index
# DUMPIR:       %[[C0_27:.*]] = arith.constant 0 : index
# DUMPIR:       nvgpu.tma.async.load %[[TMA1]][%[[C0_26]], %[[C0_27]]], %[[MB]][%[[C0_25]]] to %[[VIEW_19]], predicate = %[[EQ]] : <tensor = memref<64x64xf16, 3>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none>, <memorySpace = #gpu.address_space<workgroup>> -> memref<64x64xf16, #gpu.address_space<workgroup>>
# DUMPIR:       %[[C0_28:.*]] = arith.constant 0 : index
# DUMPIR:       %[[C64_29:.*]] = arith.constant 64 : index
# DUMPIR:       %[[C0_30:.*]] = arith.constant 0 : index
# DUMPIR:       nvgpu.tma.async.load %[[TMA1]][%[[C64_29]], %[[C0_30]]], %[[MB]][%[[C0_28]]] to %[[VIEW_20]], predicate = %[[EQ]] : <tensor = memref<64x64xf16, 3>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none>, <memorySpace = #gpu.address_space<workgroup>> -> memref<64x64xf16, #gpu.address_space<workgroup>>
# DUMPIR:       %[[C0_31:.*]] = arith.constant 0 : index
# DUMPIR:       %[[C10000000:.*]] = arith.constant 10000000 : index
# DUMPIR:       %[[FALSE:.*]] = arith.constant false
# DUMPIR:       nvgpu.mbarrier.try_wait.parity %[[MB]][%[[C0_31]]], %[[FALSE]], %[[C10000000]] : <memorySpace = #gpu.address_space<workgroup>>
# DUMPIR:       %[[WG_ACC:.*]] = nvgpu.warpgroup.mma.init.accumulator -> <fragmented = vector<128x128xf32>>
# DUMPIR:       %[[GEN0:.*]] = nvgpu.warpgroup.generate.descriptor %[[VIEW]], %[[TMA0]] : memref<128x64xf16, #gpu.address_space<workgroup>>, <tensor = memref<128x64xf16, 3>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none> -> <tensor = memref<128x64xf16, #gpu.address_space<workgroup>>>
# DUMPIR:       %[[GEN1:.*]] = nvgpu.warpgroup.generate.descriptor %[[VIEW_15]], %[[TMA1]] : memref<64x128xf16, #gpu.address_space<workgroup>>, <tensor = memref<64x64xf16, 3>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none> -> <tensor = memref<64x128xf16, #gpu.address_space<workgroup>>>
# DUMPIR:       %[[MMA:.*]] = nvgpu.warpgroup.mma %[[GEN0]], %[[GEN1]], %[[WG_ACC]] {transposeB} : <tensor = memref<128x64xf16, #gpu.address_space<workgroup>>>, <tensor = memref<64x128xf16, #gpu.address_space<workgroup>>>, <fragmented = vector<128x128xf32>> -> <fragmented = vector<128x128xf32>>
# DUMPIR:       nvgpu.warpgroup.mma.store %[[MMA]], %{{.*}} : <fragmented = vector<128x128xf32>> to memref<128x128xf32>
# DUMPIR:       gpu.terminator
# DUMPIR:     }
# DUMPIR:     %[[CPY3:.*]] = gpu.memcpy async [%{{.*}}] %[[ARG2]], %{{.*}} : memref<128x128xf32>, memref<128x128xf32>
# DUMPIR:     gpu.wait async [%[[CPY3]]]
# DUMPIR:     return
# DUMPIR:   }
