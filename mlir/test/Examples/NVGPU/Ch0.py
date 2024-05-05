# RUN: env SUPPORT_LIB=%mlir_cuda_runtime \
# RUN:   %PYTHON %s | FileCheck %s

# ===----------------------------------------------------------------------===//
#  Chapter 0 : Hello World
# ===----------------------------------------------------------------------===//
#
# This program demonstrates Hello World:
#   1. Build MLIR function with arguments
#   2. Build MLIR GPU kernel
#   3. Print from a GPU thread
#   4. Pass arguments, JIT compile and run the MLIR function
#
# ===----------------------------------------------------------------------===//


from mlir.dialects import gpu
from tools.nvdsl import *


# 1. The decorator generates a MLIR func.func.
# Everything inside the Python function becomes the body of the func.
# The decorator also translates `alpha` to an `index` type.
@NVDSL.mlir_func
def main(alpha):
    # 2. The decorator generates a MLIR gpu.launch.
    # Everything inside the Python function becomes the body of the gpu.launch.
    # This allows for late outlining of the GPU kernel, enabling optimizations
    # like constant folding from host to device.
    @NVDSL.mlir_gpu_launch(grid=(1, 1, 1), block=(4, 1, 1))
    def kernel():
        tidx = gpu.thread_id(gpu.Dimension.x)
        # + operator generates arith.addi
        myValue = alpha + tidx
        # Print from a GPU thread
        gpu.printf("GPU thread %llu has %llu\n", [tidx, myValue])

    # 3. Call the GPU kernel
    kernel()


alpha = 100
# 4. The `mlir_func` decorator JIT compiles the IR and executes the MLIR function.
main(alpha)


# CHECK: GPU thread 0 has 100
# CHECK: GPU thread 1 has 101
# CHECK: GPU thread 2 has 102
# CHECK: GPU thread 3 has 103
