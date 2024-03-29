# RUN: env SUPPORT_LIB=%mlir_cuda_runtime \
# RUN:   %PYTHON %s | FileCheck %s

# ===----------------------------------------------------------------------===//
#  Chapter 0 : Hello World
# ===----------------------------------------------------------------------===//
#
# This program demonstrates Hello World
#
# This chapter introduces demonstrates:
#   1. Build MLIR function with arguments
#   2. Build MLIR GPU kernel
#   3. Print from a GPU thread
#   4. Pass arguments, JIT compile and run the MLIR function
#
# ===----------------------------------------------------------------------===//


from mlir.dialects import gpu
from tools.nvdsl import *


# 1. Build function with arguments
@NVDSL.mlir_func
def main(alpha):
    # 2. Build GPU kernel
    @NVDSL.mlir_gpu_launch(grid=(1, 1, 1), block=(4, 1, 1))
    def kernel():
        tidx = gpu.thread_id(gpu.Dimension.x)
        myValue = alpha + tidx
        # Print from a GPU thread
        gpu.printf("GPU thread %llu has %llu\n", [tidx, myValue])

    # 3. Call the GPU kernel
    kernel()


# 4. Pass arguments, JIT compile and run the MLIR function
alpha = 100
main(alpha)


# CHECK: GPU thread 0 has 100
# CHECK: GPU thread 1 has 101
# CHECK: GPU thread 2 has 102
# CHECK: GPU thread 3 has 103
