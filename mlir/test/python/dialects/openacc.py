from mlir.ir import *
from mlir.dialects import openacc

def run(f):
    print("\nTEST:", f.__name__)
    with Context(), Location.unknown():
        f()
    return f

@run
def testOpenACCKernel():
    module = Module.create()
    with InsertionPoint(module.body):
        openacc.KernelOp(
            openacc.KernelType.parallel,
            openacc.KernelModifier.seq,
            openacc.KernelModifier.seq,
        )
    print(module)
