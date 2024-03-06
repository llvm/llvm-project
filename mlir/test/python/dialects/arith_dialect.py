# RUN: %PYTHON %s | FileCheck %s
from functools import partialmethod

from mlir.ir import *
import mlir.dialects.arith as arith
import mlir.dialects.func as func


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


# CHECK-LABEL: TEST: testArithValue
@run
def testArithValue():
    def _binary_op(lhs, rhs, op: str) -> "ArithValue":
        op = op.capitalize()
        if arith._is_float_type(lhs.type) and arith._is_float_type(rhs.type):
            op += "F"
        elif arith._is_integer_like_type(lhs.type) and arith._is_integer_like_type(
            lhs.type
        ):
            op += "I"
        else:
            raise NotImplementedError(f"Unsupported '{op}' operands: {lhs}, {rhs}")

        op = getattr(arith, f"{op}Op")
        return op(lhs, rhs).result

    @register_value_caster(F16Type.static_typeid)
    @register_value_caster(F32Type.static_typeid)
    @register_value_caster(F64Type.static_typeid)
    @register_value_caster(IntegerType.static_typeid)
    class ArithValue(Value):
        def __init__(self, v):
            super().__init__(v)

        __add__ = partialmethod(_binary_op, op="add")
        __sub__ = partialmethod(_binary_op, op="sub")
        __mul__ = partialmethod(_binary_op, op="mul")

        def __str__(self):
            return super().__str__().replace(Value.__name__, ArithValue.__name__)

    with Context() as ctx, Location.unknown():
        module = Module.create()
        f16_t = F16Type.get()
        f32_t = F32Type.get()
        f64_t = F64Type.get()

        with InsertionPoint(module.body):
            a = arith.constant(f16_t, 42.42)
            # CHECK: ArithValue(%cst = arith.constant 4.240
            print(a)

            b = a + a
            # CHECK: ArithValue(%0 = arith.addf %cst, %cst : f16)
            print(b)

            a = arith.constant(f32_t, 42.42)
            b = a - a
            # CHECK: ArithValue(%1 = arith.subf %cst_0, %cst_0 : f32)
            print(b)

            a = arith.constant(f64_t, 42.42)
            b = a * a
            # CHECK: ArithValue(%2 = arith.mulf %cst_1, %cst_1 : f64)
            print(b)
