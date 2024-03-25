# RUN: %PYTHON %s | FileCheck %s

from mlir.ir import *
import mlir.dialects.gpu as gpu
import mlir.dialects.gpu.passes
from mlir.passmanager import *


def run(f):
    print("\nTEST:", f.__name__)
    with Context(), Location.unknown():
        f()
    return f


# CHECK-LABEL: testGPUPass
#       CHECK: SUCCESS
@run
def testGPUPass():
    PassManager.parse("any(gpu-kernel-outlining)")
    print("SUCCESS")


# CHECK-LABEL: testMMAElementWiseAttr
@run
def testMMAElementWiseAttr():
    module = Module.create()
    with InsertionPoint(module.body):
        gpu.BlockDimOp(gpu.Dimension.y)
    # CHECK: %block_dim_y = gpu.block_dim  y
    print(module)
    pass
