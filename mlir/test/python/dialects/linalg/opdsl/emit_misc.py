# RUN: %PYTHON %s | FileCheck %s

from mlir.ir import *
from mlir.dialects import builtin
from mlir.dialects import linalg
from mlir.dialects import std

from mlir.dialects.linalg.opdsl.lang import *

# This tests miscellaneous features of the emitter that are not tested by the
# matmul, convolution, or, pooling tests. The features include:
# - constant defined in the body
# - fix/predefined types
# - exponential functions
# - custom op names.


@linalg_structured_op
def fill_rng_poly(
    min=ScalarDef(F64),
    max=ScalarDef(F64),
    seed=ScalarDef(I32),
    O=TensorDef(T, S.M, S.N, output=True)):
  multiplier = TypeFn.cast(I32, const(1103515245))
  increment = TypeFn.cast(I32, const(12345))
  rand1 = (TypeFn.cast(I32, index(D.m)) + seed) * multiplier + increment
  rand2 = (TypeFn.cast(I32, index(D.n)) + rand1) * multiplier + increment
  inv_range = TypeFn.cast(F64, const(2.3283064e-10))
  offset = TypeFn.cast(F64, const(2147483647))
  scaling = (max - min) * inv_range
  O[D.m, D.n] = TypeFn.cast(T,
                            (offset + TypeFn.cast(F64, rand2)) * scaling + min)


@linalg_structured_op
def soft_plus_poly(
    I=TensorDef(T, S.M, S.N), O=TensorDef(U, S.M, S.N, output=True)):
  O[D.m, D.n] = ArithFn.log(
      TypeFn.cast(U, const(1.0)) + TypeFn.cast(U, ArithFn.exp(I[D.m, D.n])))


@linalg_structured_op(op_name="custom_op_name")
def non_default_op_name(I=TensorDef(T, S.N), O=TensorDef(T, S.N, output=True)):
  O[D.n] = I[D.n]


with Context() as ctx, Location.unknown():
  module = Module.create()
  f32 = F32Type.get()
  f64 = F64Type.get()
  i32 = IntegerType.get_signless(32)
  with InsertionPoint(module.body):

    # CHECK-LABEL: @test_i32_fill_rng
    # CHECK:      ^{{.*}}(%[[MIN:.+]]: f64, %[[MAX:.+]]: f64, %[[SEED:.+]]: i32, %{{.*}}
    # CHECK-DAG:    %[[IDX0:.+]] = linalg.index 0 : index
    # CHECK-DAG:    %[[IDX0_CAST:.+]] = arith.index_cast %[[IDX0]] : index to i32
    # CHECK-DAG:    %[[RND0:.+]] = arith.addi %[[IDX0_CAST]], %[[SEED]] : i32
    # CHECK-DAG:    %[[CST0:.+]] = arith.constant 1103515245 : i64
    # CHECK-DAG:    %[[CST0_CAST:.+]] = arith.trunci %[[CST0]] : i64 to i32
    # Skip the remaining random number computation and match the scaling logic.
    # CHECK-DAG:    %[[DIFF:.+]] = arith.subf %[[MAX]], %[[MIN]] : f64
    # CHECK-DAG:    %[[CST3:.+]] = arith.constant 2.3283063999999999E-10 : f64
    # CHECK-DAG:    %[[FACT:.+]] = arith.mulf %[[DIFF]], %[[CST3]] : f64
    # CHECK-DAG:    %[[RND4:.+]] = arith.mulf %{{.+}}, %[[FACT]] : f64
    # CHECK-DAG:    %[[RND5:.+]] = arith.addf %[[RND4]], %[[MIN]] : f64
    # CHECK-DAG:    %{{.*}} = arith.fptosi %[[RND5]] : f64 to i32
    @builtin.FuncOp.from_py_func(f64, f64, i32,
                                 RankedTensorType.get((4, 16), i32))
    def test_i32_fill_rng(min, max, seed, init_result):
      return fill_rng_poly(min, max, seed, outs=[init_result])

    # CHECK-LABEL: @test_f32_soft_plus
    # CHECK:      ^{{.*}}(%[[IN:.+]]: f32, %[[OUT:.+]]: f32)
    # CHECK-NEXT:   %[[C1:.+]] = arith.constant 1.000000e+00 : f64
    # CHECK-NEXT:   %[[C1_CAST:.+]] = arith.truncf %[[C1]] : f64 to f32
    # CHECK-NEXT:   %[[EXP:.+]] = math.exp %[[IN]] : f32
    # CHECK-NEXT:   %[[SUM:.+]] = arith.addf %[[C1_CAST]], %[[EXP]] : f32
    # CHECK-NEXT:   %[[LOG:.+]] = math.log %[[SUM]] : f32
    # CHECK-NEXT:   linalg.yield %[[LOG]] : f32
    # CHECK-NEXT: -> tensor<4x16xf32>
    @builtin.FuncOp.from_py_func(
        RankedTensorType.get((4, 16), f32), RankedTensorType.get((4, 16), f32))
    def test_f32_soft_plus(input, init_result):
      return soft_plus_poly(input, outs=[init_result])

    # Just check that we don't assert out on name mismatch.
    # CHECK-LABEL: @test_non_default_op_name
    @builtin.FuncOp.from_py_func(
        RankedTensorType.get((42,), f32), RankedTensorType.get((42,), f32))
    def test_non_default_op_name(input, init_result):
      return non_default_op_name(input, outs=[init_result])


print(module)
