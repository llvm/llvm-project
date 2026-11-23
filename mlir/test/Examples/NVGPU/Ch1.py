# RUN: env SUPPORT_LIB=%mlir_cuda_runtime \
# RUN: sh -c 'if [[ "%mlir_run_cuda_sm90_tests" == "1" ]]; \
# RUN: then %PYTHON %s | FileCheck %s; \
# RUN: else export MLIR_NVDSL_PRINT_IR=1; \
# RUN: %PYTHON %s | FileCheck %s --check-prefix=DUMPIR; fi'


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
    t1 = gpu.wait([])
    x_dev, t2 = gpu.alloc(x.type, token_ty, [t1], [], [])
    y_dev, t3 = gpu.alloc(y.type, token_ty, [t2], [], [])
    t4 = gpu.memcpy(token_ty, [t3], x_dev, x)
    t5 = gpu.memcpy(token_ty, [t4], y_dev, y)
    t6 = gpu.wait([t5])

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
    gpu.wait([t7])


# 3. Pass numpy arrays to MLIR
M = 256
N = 32
alpha = 2.0
x = np.random.randn(M, N).astype(np.float32)
y = np.ones((M, N), np.float32)

saxpy(x, y, alpha)

if os.getenv("MLIR_NVDSL_PRINT_IR") != "1":
    # 4. Verify MLIR with reference computation
    ref = np.ones((M, N), np.float32)
    ref += x * alpha
    np.testing.assert_allclose(y, ref, rtol=5e-03, atol=1e-01)
    print("PASS")
# CHECK-NOT: Mismatched elements
# CHECK: PASS

# DUMPIR:   func.func @saxpy(%[[ARG0:.*]]: memref<256x32xf32>, %[[ARG1:.*]]: memref<256x32xf32>, %[[ARG2:.*]]: f32) attributes {llvm.emit_c_interface} {
# DUMPIR:     %[[WAIT0:.*]] = gpu.wait async
# DUMPIR:     %[[MEMREF:.*]], %[[ASYNC0:.*]] = gpu.alloc async [%[[WAIT0]]] () : memref<256x32xf32>
# DUMPIR:     %[[MEMREF0:.*]], %[[ASYNC1:.*]] = gpu.alloc async [%[[ASYNC0]]] () : memref<256x32xf32>
# DUMPIR:     %[[MEMCPY1:.*]] = gpu.memcpy async [%[[ASYNC1]]] %[[MEMREF]], %[[ARG0]] : memref<256x32xf32>, memref<256x32xf32>
# DUMPIR:     %[[MEMCPY2:.*]] = gpu.memcpy async [%[[MEMCPY1]]] %[[MEMREF0]], %[[ARG1]] : memref<256x32xf32>, memref<256x32xf32>
# DUMPIR:     %[[WAIT1:.*]] = gpu.wait async [%[[MEMCPY2]]]
# DUMPIR:       %[[LD0:.*]] = memref.load %[[MEMREF]][%{{.*}}, %{{.*}}] : memref<256x32xf32>
# DUMPIR:       %[[LD1:.*]] = memref.load %[[MEMREF0]][%{{.*}}, %{{.*}}] : memref<256x32xf32>
# DUMPIR:       %[[MUL:.*]] = arith.mulf %[[LD0]], %[[ARG2]] : f32
# DUMPIR:       %[[ADD:.*]] = arith.addf %[[LD1]], %[[MUL]] : f32
# DUMPIR:       memref.store %[[ADD]], %[[MEMREF0]][%{{.*}}, %{{.*}}] : memref<256x32xf32>
# DUMPIR:       gpu.terminator
# DUMPIR:     %[[MEMCPY3:.*]] = gpu.memcpy async [%[[WAIT1]]] %[[ARG1]], %[[MEMREF0]] : memref<256x32xf32>, memref<256x32xf32>
# DUMPIR:     %[[WAIT2:.*]] = gpu.wait async [%[[MEMCPY3]]]
# DUMPIR:     return
# DUMPIR:   }
