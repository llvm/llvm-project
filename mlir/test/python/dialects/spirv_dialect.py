# RUN: %PYTHON %s | FileCheck %s

from mlir.ir import *
import mlir.dialects.spirv as spirv


def run(f):
    print("\nTEST:", f.__name__)
    f()


# CHECK-LABEL: TEST: testConstantOp
@run
def testConstantOps():
    with Context() as ctx, Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            spirv.ConstantOp(
                value=FloatAttr.get(F32Type.get(), 42.42), constant=F32Type.get()
            )
        # CHECK:         %cst_f32 = spirv.Constant 4.242000e+01 : f32
        print(module)
