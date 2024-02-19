# RUN: %PYTHON %s | FileCheck %s

from mlir.ir import *
from mlir.dialects.openmp import *


def constructAndPrintInModule(f):
    print("\nTEST:", f.__name__)
    with Context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            f()
        print(module)
    return f


# CHECK-LABEL: test_barrier
# CHECK: module {
# CHECK:   omp.barrier
# CHECK: }
@constructAndPrintInModule
def test_barrier():
    barrier = BarrierOp()
