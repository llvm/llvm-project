# RUN: %PYTHON %s | FileCheck %s

from mlir.ir import *
import mlir.dialects.func as func
import mlir.dialects.arith as arith


def run(f):
    print("\nTEST:", f.__name__)
    f()


# CHECK-LABEL: TEST: testConstantOp
@run
def testConstantOps():
    with Context() as ctx, Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            arith.ConstantOp(value=42.42, result=F32Type.get())
        # CHECK:         %cst = arith.constant 4.242000e+01 : f32
        print(module)


# CHECK-LABEL: TEST: testFastMathFlags
@run
def testFastMathFlags():
    with Context() as ctx, Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            a = arith.ConstantOp(value=42.42, result=F32Type.get())
            r = arith.AddFOp(
                a, a, fastmath=arith.FastMathFlags.nnan | arith.FastMathFlags.ninf
            )
            # CHECK: %0 = arith.addf %cst, %cst fastmath<nnan,ninf> : f32
            print(r)


# CHECK-LABEL: TEST: testArithValueBuilder
@run
def testArithValueBuilder():
    with Context() as ctx, Location.unknown():
        module = Module.create()
        f32_t = F32Type.get()

        with InsertionPoint(module.body):
            a = arith.constant(value=FloatAttr.get(f32_t, 42.42))
            # CHECK: %cst = arith.constant 4.242000e+01 : f32
            print(a)
