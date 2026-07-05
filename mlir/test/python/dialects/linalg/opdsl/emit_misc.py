# RUN: %PYTHON %s | FileCheck %s

from mlir.ir import *
from mlir.dialects import builtin
from mlir.dialects import func
from mlir.dialects import linalg

from mlir.dialects.linalg.opdsl.lang import *

# This tests miscellaneous features of the emitter that are not tested by the
# fill, matmul, convolution, or pooling tests. The features include:
# - constant defined in the body
# - fix/predefined types
# - some math/arith functions, including abs, ceil, exp, floor, log, and negf
# - custom op names.


@linalg_structured_op
def test_const(O=TensorDef(F32, S.M, S.N, output=True)):
    O[D.m, D.n] = TypeFn.cast_unsigned(F32, const(42)) + TypeFn.cast_unsigned(
        F32, const(2.3283064e-10)
    )


@linalg_structured_op
def test_index(O=TensorDef(I32, S.M, S.N, output=True)):
    O[D.m, D.n] = TypeFn.cast_signed(I32, index(D.m)) + TypeFn.cast_signed(
        I32, index(D.n)
    )


@linalg_structured_op
def elemwise_unary_poly(
    I=TensorDef(T),
    O=TensorDef(U, output=True),
    fun=UnaryFnAttrDef(default=UnaryFn.exp),
    cast=TypeFnAttrDef(default=TypeFn.cast_signed),
):
    O[None] = fun(cast(U, I[None]))


@linalg_structured_op(op_name="custom_op_name")
def non_default_op_name(I=TensorDef(T, S.N), O=TensorDef(T, S.N, output=True)):
    O[D.n] = I[D.n]


with Context() as ctx, Location.unknown():
    module = Module.create()
    f32 = F32Type.get()
    c32 = ComplexType.get(f32)
    i32 = IntegerType.get_signless(32)
    with InsertionPoint(module.body):

        # CHECK-LABEL: @test_f32_const
        # CHECK-DAG:    %[[CST0:.+]] = arith.constant 42 : i64
        # CHECK-DAG:    %[[CST0_CAST:.+]] = arith.uitofp %[[CST0]] : i64 to f32
        # CHECK-DAG:    %[[CST1:.+]] = arith.constant 2.3283063999999999E-10 : f64
        # CHECK-DAG:    %[[CST1_CAST:.+]] = arith.truncf %[[CST1]] : f64 to f32
        # CHECK-DAG:    %[[SUM:.+]] = arith.addf %[[CST0_CAST]], %[[CST1_CAST]] : f32
        # CHECK-NEXT:   linalg.yield %[[SUM]] : f32
        @func.FuncOp.from_py_func(RankedTensorType.get((4, 16), f32))
        def test_f32_const(init_result):
            return test_const(outs=[init_result])

        # CHECK-LABEL: @test_i32_index
        # CHECK-DAG:    %[[IDX0:.+]] = linalg.index 0 : index
        # CHECK-DAG:    %[[IDX1:.+]] = linalg.index 1 : index
        # CHECK-DAG:    %[[IDX0_CAST:.+]] = arith.index_cast %[[IDX0]] : index to i32
        # CHECK-DAG:    %[[IDX1_CAST:.+]] = arith.index_cast %[[IDX1]] : index to i32
        # CHECK-DAG:    %[[SUM:.+]] = arith.addi %[[IDX0_CAST]], %[[IDX1_CAST]] : i32
        # CHECK-NEXT:   linalg.yield %[[SUM]] : i32
        @func.FuncOp.from_py_func(RankedTensorType.get((4, 16), i32))
        def test_i32_index(init_result):
            return test_index(outs=[init_result])

        # CHECK-LABEL: @test_f32_elemwise_exp
        # CHECK:      ^{{.*}}(%[[IN:.+]]: f32, %[[OUT:.+]]: f32)
        # CHECK-NEXT:   %[[EXP:.+]] = math.exp %[[IN]] : f32
        # CHECK-NEXT:   linalg.yield %[[EXP]] : f32
        # CHECK-NEXT: -> tensor<4x16xf32>
        @func.FuncOp.from_py_func(
            RankedTensorType.get((4, 16), f32), RankedTensorType.get((4, 16), f32)
        )
        def test_f32_elemwise_exp(input, init_result):
            return elemwise_unary_poly(input, outs=[init_result], fun=UnaryFn.exp)

        # CHECK-LABEL: @test_f32_elemwise_log
        # CHECK:      ^{{.*}}(%[[IN:.+]]: f32, %[[OUT:.+]]: f32)
        # CHECK-NEXT:   %[[LOG:.+]] = math.log %[[IN]] : f32
        # CHECK-NEXT:   linalg.yield %[[LOG]] : f32
        # CHECK-NEXT: -> tensor<4x16xf32>
        @func.FuncOp.from_py_func(
            RankedTensorType.get((4, 16), f32), RankedTensorType.get((4, 16), f32)
        )
        def test_f32_elemwise_log(input, init_result):
            return elemwise_unary_poly(input, outs=[init_result], fun=UnaryFn.log)

        # CHECK-LABEL: @test_f32_elemwise_abs
        # CHECK:      ^{{.*}}(%[[IN:.+]]: f32, %[[OUT:.+]]: f32)
        # CHECK-NEXT:   %[[EXP:.+]] = math.absf %[[IN]] : f32
        # CHECK-NEXT:   linalg.yield %[[EXP]] : f32
        # CHECK-NEXT: -> tensor<4x16xf32>
        @func.FuncOp.from_py_func(
            RankedTensorType.get((4, 16), f32), RankedTensorType.get((4, 16), f32)
        )
        def test_f32_elemwise_abs(input, init_result):
            return elemwise_unary_poly(input, outs=[init_result], fun=UnaryFn.abs)

        # CHECK-LABEL: @test_f32_elemwise_ceil
        # CHECK:      ^{{.*}}(%[[IN:.+]]: f32, %[[OUT:.+]]: f32)
        # CHECK-NEXT:   %[[EXP:.+]] = math.ceil %[[IN]] : f32
        # CHECK-NEXT:   linalg.yield %[[EXP]] : f32
        # CHECK-NEXT: -> tensor<4x16xf32>
        @func.FuncOp.from_py_func(
            RankedTensorType.get((4, 16), f32), RankedTensorType.get((4, 16), f32)
        )
        def test_f32_elemwise_ceil(input, init_result):
            return elemwise_unary_poly(input, outs=[init_result], fun=UnaryFn.ceil)

        # CHECK-LABEL: @test_f32_elemwise_floor
        # CHECK:      ^{{.*}}(%[[IN:.+]]: f32, %[[OUT:.+]]: f32)
        # CHECK-NEXT:   %[[EXP:.+]] = math.floor %[[IN]] : f32
        # CHECK-NEXT:   linalg.yield %[[EXP]] : f32
        # CHECK-NEXT: -> tensor<4x16xf32>
        @func.FuncOp.from_py_func(
            RankedTensorType.get((4, 16), f32), RankedTensorType.get((4, 16), f32)
        )
        def test_f32_elemwise_floor(input, init_result):
            return elemwise_unary_poly(input, outs=[init_result], fun=UnaryFn.floor)

        # CHECK-LABEL: @test_f32_elemwise_neg
        # CHECK:      ^{{.*}}(%[[IN:.+]]: f32, %[[OUT:.+]]: f32)
        # CHECK-NEXT:   %[[EXP:.+]] = arith.negf %[[IN]] : f32
        # CHECK-NEXT:   linalg.yield %[[EXP]] : f32
        # CHECK-NEXT: -> tensor<4x16xf32>
        @func.FuncOp.from_py_func(
            RankedTensorType.get((4, 16), f32), RankedTensorType.get((4, 16), f32)
        )
        def test_f32_elemwise_neg(input, init_result):
            return elemwise_unary_poly(input, outs=[init_result], fun=UnaryFn.negf)

        # CHECK-LABEL: @test_c32_elemwise_neg
        # CHECK:      ^{{.*}}(%[[IN:.+]]: complex<f32>, %[[OUT:.+]]: complex<f32>)
        # CHECK-NEXT:   %[[EXP:.+]] = complex.neg %[[IN]] : complex<f32>
        # CHECK-NEXT:   linalg.yield %[[EXP]] : complex<f32>
        # CHECK-NEXT: -> tensor<4x16xcomplex<f32>>
        @func.FuncOp.from_py_func(
            RankedTensorType.get((4, 16), c32), RankedTensorType.get((4, 16), c32)
        )
        def test_c32_elemwise_neg(input, init_result):
            return elemwise_unary_poly(input, outs=[init_result], fun=UnaryFn.negf)

        # Just check that we don't assert out on name mismatch.
        # CHECK-LABEL: @test_non_default_op_name
        @func.FuncOp.from_py_func(
            RankedTensorType.get((42,), f32), RankedTensorType.get((42,), f32)
        )
        def test_non_default_op_name(input, init_result):
            return non_default_op_name(input, outs=[init_result])


print(module)
