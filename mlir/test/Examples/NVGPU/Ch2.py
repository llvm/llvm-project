# RUN: env SUPPORT_LIB=%mlir_cuda_runtime \
# RUN: sh -c 'if [[ "%mlir_run_cuda_sm90_tests" == "1" ]]; \
# RUN: then %PYTHON %s | FileCheck %s; \
# RUN: else export MLIR_NVDSL_PRINT_IR=1; \
# RUN: %PYTHON %s | FileCheck %s --check-prefix=DUMPIR; fi'


# ===----------------------------------------------------------------------===//
#  Chapter 2 : 2D Saxpy with TMA
# ===----------------------------------------------------------------------===//
#
# This program demonstrates 2D Saxpy. It is same as Chapter 1,
# but it loads data using TMA (Tensor Memory Accelerator)
#
# This chapter introduces demonstrates:
#  1. Computes 2D SAXPY in the same way as Ch1.py but loads data using TMA
#  2. Create and initialize 1 asynchronous transactional barrier (mbarrier)
#  3. Thread-0 Load request data load from TMA for each thread block
#  4. Each thread block loads <1x32xf32> for x and y.
#  5. Wait for completion of TMA load with mbarrier
#
# ===----------------------------------------------------------------------===//

from mlir import ir
from mlir.dialects import nvgpu, scf, arith, memref, vector, gpu
from tools.nvdsl import *
from mlir import runtime as rt
from mlir.extras import types as T
import numpy as np


@NVDSL.mlir_func
def saxpy(x, y, alpha):
    token_ty = gpu.AsyncTokenType.get()
    t1 = gpu.wait([])
    x_dev, t2 = gpu.alloc(x.type, token_ty, [t1], [], [])
    y_dev, t3 = gpu.alloc(y.type, token_ty, [t2], [], [])
    t4 = gpu.memcpy(token_ty, [t3], x_dev, x)
    t5 = gpu.memcpy(token_ty, [t4], y_dev, y)
    t6 = gpu.wait([t5])

    x_tma = TMA([1, N], x.type)
    y_tma = TMA([1, N], y.type)
    x_tma.create_descriptor(x_dev)
    y_tma.create_descriptor(y_dev)
    sz_x = get_type_size(x_tma.tma_memref)
    sz_y = get_type_size(x_tma.tma_memref)
    sz = sz_x + sz_y

    @NVDSL.mlir_gpu_launch(grid=(M, 1, 1), block=(N, 1, 1), smem=sz)
    def saxpy_tma_kernel():
        bidx = gpu.block_id(gpu.Dimension.x)
        tidx = gpu.thread_id(gpu.Dimension.x)
        isThread0 = tidx == 0

        # 1. Create and initialize asynchronous transactional barrier (mbarrier)
        mbar_group = Mbarriers(number_of_barriers=1)
        mbar_group[0].init(1, predicate=isThread0)

        # 2. Execute Tensor Memory Accelerator (TMA) Load
        x_smem = get_dynamic_shared_memory([1, N], T.f32())
        y_smem = get_dynamic_shared_memory([1, N], T.f32(), offset=sz_x)
        x_tma.load(x_smem, mbar_group[0], coords=[0, bidx], predicate=isThread0)
        y_tma.load(y_smem, mbar_group[0], coords=[0, bidx], predicate=isThread0)
        mbar_group[0].arrive(txcount=sz, predicate=isThread0)

        # 3. Wait for completion of TMA load with mbarrier
        mbar_group[0].try_wait()

        x_val = memref.load(x_smem, [const(0), tidx])
        y_val = memref.load(y_smem, [const(0), tidx])

        # SAXPY: y[i] += a * x[i];
        y_val += x_val * alpha

        memref.store(y_val, y_dev, [bidx, tidx])

    saxpy_tma_kernel()

    t7 = gpu.memcpy(token_ty, [t6], y, y_dev)
    gpu.wait([t7])


# 3. Pass numpy arrays to MLIR
M = 256
N = 32
alpha = 2.0
x = np.random.randn(M, N).astype(np.float32)
y = np.ones((M, N), np.float32)
saxpy(x, y, alpha)

if os.getenv("MLIR_NVDSL_PRINT_IR") != "1":
    #  4. Verify MLIR with reference computation
    ref = np.ones((M, N), np.float32)
    ref += x * alpha
    np.testing.assert_allclose(y, ref, rtol=5e-03, atol=1e-01)
    print("PASS")
# CHECK-NOT: Mismatched elements
# CHECK: PASS

# DUMPIR:   func.func @saxpy(%{{.*}}: memref<256x32xf32>, %[[ARG1:.*]]: memref<256x32xf32>, %[[ARG2:.*]]: f32) attributes {llvm.emit_c_interface} {
# DUMPIR:     %[[WAIT0:.*]] = gpu.wait async
# DUMPIR:     %[[MEMREF:.*]], %[[ASYNC0:.*]] = gpu.alloc async [%[[WAIT0]]] () : memref<256x32xf32>
# DUMPIR:     %[[CAST:.*]] = memref.cast %[[MEMREF]] : memref<256x32xf32> to memref<*xf32>
# DUMPIR:     %[[C1:.*]] = arith.constant 1 : index
# DUMPIR:     %[[C32:.*]] = arith.constant 32 : index
# DUMPIR:     %[[TMA0:.*]] = nvgpu.tma.create.descriptor %[[CAST]] box[%[[C1]], %[[C32]]] : memref<*xf32> -> <tensor = memref<1x32xf32, 3>, swizzle = none, l2promo = none, oob = zero, interleave = none>
# DUMPIR:       %[[C0:.*]] = arith.constant 0 : index
# DUMPIR:       %[[EQ:.*]] = arith.cmpi eq, %{{.*}}, %[[C0]] : index
# DUMPIR:       %[[MB:.*]] = nvgpu.mbarrier.create -> <memorySpace = #gpu.address_space<workgroup>>
# DUMPIR:       %[[C0_10:.*]] = arith.constant 0 : index
# DUMPIR:       %[[C1_11:.*]] = arith.constant 1 : index
# DUMPIR:       nvgpu.mbarrier.init %[[MB]][%[[C0_10]]], %[[C1_11]], predicate = %[[EQ]] : <memorySpace = #gpu.address_space<workgroup>>
# DUMPIR:       %[[DSM0:.*]] = gpu.dynamic_shared_memory : memref<?xi8, #gpu.address_space<workgroup>>
# DUMPIR:       %[[C0_12:.*]] = arith.constant 0 : index
# DUMPIR:       %[[VIEW:.*]] = memref.view %[[DSM0]][%[[C0_12]]][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<1x32xf32, #gpu.address_space<workgroup>>
# DUMPIR:       %[[DSM1:.*]] = gpu.dynamic_shared_memory : memref<?xi8, #gpu.address_space<workgroup>>
# DUMPIR:       %[[C128:.*]] = arith.constant 128 : index
# DUMPIR:       %[[VIEW_13:.*]] = memref.view %[[DSM1]][%[[C128]]][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<1x32xf32, #gpu.address_space<workgroup>>
# DUMPIR:       nvgpu.tma.async.load %[[TMA0]][%{{.*}}, %{{.*}}], %[[MB]][%{{.*}}] to %[[VIEW]], predicate = %[[EQ]] : <tensor = memref<1x32xf32, 3>, swizzle = none, l2promo = none, oob = zero, interleave = none>, <memorySpace = #gpu.address_space<workgroup>> -> memref<1x32xf32, #gpu.address_space<workgroup>>
# DUMPIR:       nvgpu.mbarrier.arrive.expect_tx %[[MB]][%{{.*}}], %{{.*}}, predicate = %[[EQ]] : <memorySpace = #gpu.address_space<workgroup>>
# DUMPIR:       %[[C0_20:.*]] = arith.constant 0 : index
# DUMPIR:       %[[C10000000:.*]] = arith.constant 10000000 : index
# DUMPIR:       %[[FALSE:.*]] = arith.constant false
# DUMPIR:       nvgpu.mbarrier.try_wait.parity %[[MB]][%[[C0_20]]], %[[FALSE]], %[[C10000000]] : <memorySpace = #gpu.address_space<workgroup>>
# DUMPIR:       %[[C0_21:.*]] = arith.constant 0 : index
# DUMPIR:       %[[LD0:.*]] = memref.load %[[VIEW]][%[[C0_21]], %{{.*}}] : memref<1x32xf32, #gpu.address_space<workgroup>>
# DUMPIR:       %[[C0_22:.*]] = arith.constant 0 : index
# DUMPIR:       %[[LD1:.*]] = memref.load %[[VIEW_13]][%[[C0_22]], %{{.*}}] : memref<1x32xf32, #gpu.address_space<workgroup>>
# DUMPIR:       memref.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<256x32xf32>
# DUMPIR:     %[[MEMCPY3:.*]] = gpu.memcpy async [%{{.*}}] %[[ARG1]], %{{.*}} : memref<256x32xf32>, memref<256x32xf32>
# DUMPIR:     %{{.*}} = gpu.wait async [%[[MEMCPY3]]]
# DUMPIR:     return
# DUMPIR:   }
