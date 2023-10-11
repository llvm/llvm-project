# RUN: %PYTHON %s | FileCheck %s
from functools import partialmethod

from mlir.ir import *
import mlir.dialects.arith as arith
from mlir.dialects._ods_common import maybe_cast


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
    def _binary_op(lhs, rhs, op: str):
        op = op.capitalize()
        if arith._is_float_type(lhs.type):
            op += "F"
        elif arith._is_integer_like_type(lhs.type):
            op += "I"
        else:
            raise NotImplementedError(f"Unsupported '{op}' operands: {lhs}, {rhs}")

        op = getattr(arith, f"{op}Op")
        return maybe_cast(op(lhs, rhs).result)

    @register_value_caster(F16Type.static_typeid)
    @register_value_caster(F32Type.static_typeid)
    @register_value_caster(F64Type.static_typeid)
    @register_value_caster(IntegerType.static_typeid)
    class ArithValue(Value):
        __add__ = partialmethod(_binary_op, op="add")
        __sub__ = partialmethod(_binary_op, op="sub")
        __mul__ = partialmethod(_binary_op, op="mul")

        def __str__(self):
            return super().__str__().replace("Value", "ArithValue")

    @register_value_caster(IntegerType.static_typeid, priority=0)
    class ArithValue1(Value):
        __mul__ = partialmethod(_binary_op, op="mul")

        def __str__(self):
            return super().__str__().replace("Value", "ArithValue1")

    @register_value_caster(IntegerType.static_typeid, priority=0)
    def no_op_caster(val):
        print("no_op_caster", val)
        return None

    with Context() as ctx, Location.unknown():
        module = Module.create()
        f16_t = F16Type.get()
        f32_t = F32Type.get()
        f64_t = F64Type.get()
        i32 = IntegerType.get_signless(32)

        with InsertionPoint(module.body):
            a = arith.constant(value=FloatAttr.get(f16_t, 42.42))
            b = a + a
            # CHECK: ArithValue(%0 = arith.addf %cst, %cst : f16)
            print(b)

            a = arith.constant(value=FloatAttr.get(f32_t, 42.42))
            b = a - a
            # CHECK: ArithValue(%1 = arith.subf %cst_0, %cst_0 : f32)
            print(b)

            a = arith.constant(value=FloatAttr.get(f64_t, 42.42))
            b = a * a
            # CHECK: ArithValue(%2 = arith.mulf %cst_1, %cst_1 : f64)
            print(b)

            # CHECK: no_op_caster Value(%c1_i32 = arith.constant 1 : i32)
            a = arith.constant(value=IntegerAttr.get(i32, 1))
            b = a * a
            # CHECK: no_op_caster Value(%3 = arith.muli %c1_i32, %c1_i32 : i32)
            # CHECK: ArithValue1(%3 = arith.muli %c1_i32, %c1_i32 : i32)
            print(b)
