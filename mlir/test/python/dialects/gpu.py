# RUN: %PYTHON %s | FileCheck %s

from mlir.ir import *
import mlir.dialects.gpu
import mlir.dialects.gpu.passes
from mlir.passmanager import *


def run(f):
    print("\nTEST:", f.__name__)
    f()


def testGPUPass():
    with Context() as context:
        PassManager.parse("any(gpu-kernel-outlining)")
    print("SUCCESS")


# CHECK-LABEL: testGPUPass
#       CHECK: SUCCESS
run(testGPUPass)
