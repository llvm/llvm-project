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
            i32 = IntegerType.get_signless(32)
            spirv.ConstantOp(value=IntegerAttr.get(i32, 42), constant=i32)
        # CHECK: spirv.Constant 42 : i32
        print(module)
