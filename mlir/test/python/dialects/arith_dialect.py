# RUN: %PYTHON %s | FileCheck %s
from functools import partialmethod

from mlir.ir import *
import mlir.dialects.arith as arith
import mlir.dialects.func as func
from array import array


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


# CHECK-LABEL: TEST: testArrayConstantConstruction
@run
def testArrayConstantConstruction():
    with Context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            i32_array = array("i", [1, 2, 3, 4])
            i32 = IntegerType.get_signless(32)
            vec_i32 = VectorType.get([2, 2], i32)
            arith.constant(vec_i32, i32_array)
            arith.ConstantOp(vec_i32, DenseIntElementsAttr.get(i32_array, type=vec_i32))

            # "q" is the equivalent of `long long` in C and requires at least
            # 64 bit width integers on both Linux and Windows.
            i64_array = array("q", [5, 6, 7, 8])
            i64 = IntegerType.get_signless(64)
            vec_i64 = VectorType.get([1, 4], i64)
            arith.constant(vec_i64, i64_array)
            arith.ConstantOp(vec_i64, DenseIntElementsAttr.get(i64_array, type=vec_i64))

            f32_array = array("f", [1.0, 2.0, 3.0, 4.0])
            f32 = F32Type.get()
            vec_f32 = VectorType.get([4, 1], f32)
            arith.constant(vec_f32, f32_array)
            arith.ConstantOp(vec_f32, DenseFPElementsAttr.get(f32_array, type=vec_f32))

            f64_array = array("d", [1.0, 2.0, 3.0, 4.0])
            f64 = F64Type.get()
            vec_f64 = VectorType.get([2, 1, 2], f64)
            arith.constant(vec_f64, f64_array)
            arith.ConstantOp(vec_f64, DenseFPElementsAttr.get(f64_array, type=vec_f64))

        # CHECK-COUNT-2: arith.constant dense<[{{\[}}1, 2], [3, 4]]> : vector<2x2xi32>
        # CHECK-COUNT-2: arith.constant dense<[{{\[}}5, 6, 7, 8]]> : vector<1x4xi64>
        # CHECK-COUNT-2: arith.constant dense<[{{\[}}1.000000e+00], [2.000000e+00], [3.000000e+00], [4.000000e+00]]> : vector<4x1xf32>
        # CHECK-COUNT-2: arith.constant dense<[{{\[}}[1.000000e+00, 2.000000e+00]], [{{\[}}3.000000e+00, 4.000000e+00]]]> : vector<2x1x2xf64>
        print(module)
