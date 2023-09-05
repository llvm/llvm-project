# RUN: %PYTHON %s | FileCheck %s
# This is just a smoke test that the dialect is functional.

from mlir.ir import *
from mlir.dialects import amdgpu, arith, memref


def constructAndPrintInModule(f):
    print("\nTEST:", f.__name__)
    with Context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            f()
        print(module)
    return f


# CHECK-LABEL: testSmoke
@constructAndPrintInModule
def testSmoke():
    # CHECK: amdgpu.lds_barrier
    amdgpu.LDSBarrierOp()
