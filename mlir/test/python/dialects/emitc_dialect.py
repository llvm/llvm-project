# RUN: %PYTHON %s | FileCheck %s

from mlir.ir import *
import mlir.dialects.emitc as emitc


def run(f):
    print("\nTEST:", f.__name__)
    with Context() as ctx, Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            f(ctx)
        print(module)


# CHECK-LABEL: TEST: testConstantOp
@run
def testConstantOp(ctx):
    i32 = IntegerType.get_signless(32)
    a = emitc.ConstantOp(result=i32, value=IntegerAttr.get(i32, 42))
    # CHECK: %{{.*}} = "emitc.constant"() <{value = 42 : i32}> : () -> i32


# CHECK-LABEL: TEST: testAddOp
@run
def testAddOp(ctx):
    i32 = IntegerType.get_signless(32)
    lhs = emitc.ConstantOp(result=i32, value=IntegerAttr.get(i32, 0))
    rhs = emitc.ConstantOp(result=i32, value=IntegerAttr.get(i32, 0))
    a = emitc.AddOp(i32, lhs, rhs)
    # CHECK: %{{.*}} = emitc.add %{{.*}}, %{{.*}} : (i32, i32) -> i32
