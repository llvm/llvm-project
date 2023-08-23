# RUN: %PYTHON %s | FileCheck %s
# This is just a smoke test that the dialect is functional.

from mlir.ir import *
from mlir.dialects import nvvm


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
    # CHECK: nvvm.cp.async.wait.group 5
    nvvm.CpAsyncWaitGroupOp(5)
