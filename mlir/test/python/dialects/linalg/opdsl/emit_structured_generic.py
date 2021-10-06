# RUN: %PYTHON %s | FileCheck %s

from mlir.ir import *
from mlir.dialects import builtin
from mlir.dialects import linalg
from mlir.dialects import std

from mlir.dialects.linalg.opdsl.lang import *

T1 = TV.T1
T2 = TV.T2


@linalg_structured_op
def matmul_mono(
    A=TensorDef(T, S.M, S.K),
    B=TensorDef(T, S.K, S.N),
    C=TensorDef(T, S.M, S.N, output=True)):
  domain(D.m, D.n, D.k)
  C[D.m, D.n] += A[D.m, D.k] * B[D.k, D.n]


@linalg_structured_op
def matmul_poly(
    A=TensorDef(T1, S.M, S.K),
    B=TensorDef(T2, S.K, S.N),
    C=TensorDef(U, S.M, S.N, output=True)):
  domain(D.m, D.n, D.k)
  C[D.m, D.n] += cast(U, A[D.m, D.k]) * cast(U, B[D.k, D.n])


@linalg_structured_op
def conv_poly(
    I=TensorDef(T1, S.N, S.IH, S.IW, S.C),
    K=TensorDef(T2, S.KH, S.KW, S.C),
    O=TensorDef(U, S.N, S.OH, S.OW, S.C, output=True),
    strides=AttributeDef(S.SH, S.SW),
    dilations=AttributeDef(S.DH, S.DW)):
  domain(D.n, D.oh, D.ow, D.kh, D.kw, D.c)
  O[D.n, D.oh, D.ow, D.c] += cast(
      U, I[D.n, D.oh * S.SH + D.kh * S.DH, D.ow * S.SW + D.kw * S.DW,
           D.c]) * cast(U, K[D.kh, D.kw, D.c])


@linalg_structured_op
def pooling_max_poly(
    I=TensorDef(T1, S.N, S.H, S.W, S.C),
    K=TensorDef(T2, S.KH, S.KW, index_dims=[D.kh, D.kw]),
    O=TensorDef(U, S.N, S.OH, S.OW, S.C, output=True),
    strides=AttributeDef(S.SH, S.SW),
    dilations=AttributeDef(S.DH, S.DW)):
  domain(D.n, D.oh, D.ow, D.kh, D.kw, D.c)
  O[D.n, D.oh, D.ow, D.c] = ReduceFn.max(D.kh, D.kw)(
      cast(U, I[D.n, D.oh * S.SH + D.kh * S.DH, D.ow * S.SW + D.kw * S.DW,
                D.c]))


@linalg_structured_op
def pooling_min_poly(
    I=TensorDef(T1, S.N, S.H, S.W, S.C),
    K=TensorDef(T2, S.KH, S.KW, index_dims=[D.kh, D.kw]),
    O=TensorDef(U, S.N, S.OH, S.OW, S.C, output=True),
    strides=AttributeDef(S.SH, S.SW),
    dilations=AttributeDef(S.DH, S.DW)):
  domain(D.n, D.oh, D.ow, D.kh, D.kw, D.c)
  O[D.n, D.oh, D.ow, D.c] = ReduceFn.min(D.kh, D.kw)(
      cast(U, I[D.n, D.oh * S.SH + D.kh * S.DH, D.ow * S.SW + D.kw * S.DW,
                D.c]))


@linalg_structured_op
def fill_rng_poly(
    min=ScalarDef(F64),
    max=ScalarDef(F64),
    seed=ScalarDef(I32),
    O=TensorDef(T, S.M, S.N, output=True)):
  multiplier = cast(I32, const(1103515245))
  increment = cast(I32, const(12345))
  rand1 = (cast(I32, index(D.m)) + seed) * multiplier + increment
  rand2 = (cast(I32, index(D.n)) + rand1) * multiplier + increment
  inv_range = cast(F64, const(2.3283064e-10))
  offset = cast(F64, const(2147483647))
  scaling = (max - min) * inv_range
  O[D.m, D.n] = cast(T, (offset + cast(F64, rand2)) * scaling + min)


@linalg_structured_op
def soft_plus_poly(
    I=TensorDef(T, S.M, S.N), O=TensorDef(U, S.M, S.N, output=True)):
  O[D.m, D.n] = \
      PrimFn.log(cast(U, const(1.0)) + cast(U, PrimFn.exp(I[D.m, D.n])))


with Context() as ctx, Location.unknown():
  module = Module.create()
  f16 = F16Type.get()
  f32 = F32Type.get()
  f64 = F64Type.get()
  i8 = IntegerType.get_signless(8)
  i16 = IntegerType.get_signless(16)
  i32 = IntegerType.get_signless(32)
  with InsertionPoint(module.body):

    # Multiplication indexing maps. We verify only the indexing maps of the
    # first multiplication and then do additional tests on casting and body
    # generation behavior.
    # CHECK: #[[$MUL_MAP_A:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
    # CHECK: #[[$MUL_MAP_B:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
    # CHECK: #[[$MUL_MAP_C:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>

    # Convolution indexing maps.
    # CHECK: #[[$CONV_MAP_I:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1 * 2 + d3, d2 * 4 + d4 * 2, d5)>
    # CHECK: #[[$CONV_MAP_K:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5)>
    # CHECK: #[[$CONV_MAP_O:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d5)>

    # Pooling indexing maps.
    # CHECK: #[[$POOL_MAP_K:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4)>

    # CHECK-LABEL: func @test_matmul_mono
    # CHECK-SAME:  %[[A:.+]]: tensor<4x16xf32>
    # CHECK-SAME: %[[B:.+]]: tensor<16x8xf32>

    # CHECK: %[[INITC:.+]] = linalg.init_tensor [4, 8] : tensor<4x8xf32>
    # CHECK: linalg.generic
    # CHECK-SAME: indexing_maps = [#[[$MUL_MAP_A]], #[[$MUL_MAP_B]], #[[$MUL_MAP_C]]]
    # CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
    # CHECK-SAME: ins(%[[A]], %[[B]]
    # CHECK-SAME: outs(%[[INITC]]

    @builtin.FuncOp.from_py_func(
        RankedTensorType.get((4, 16), f32), RankedTensorType.get((16, 8), f32))
    def test_matmul_mono(lhs, rhs):
      init_result = linalg.InitTensorOp([4, 8], f32)
      return matmul_mono(lhs, rhs, outs=[init_result.result])

    # CHECK-LABEL: @test_i8i8i32_matmul
    # CHECK:      ^{{.*}}(%[[A_ARG:.+]]: i8, %[[B_ARG:.+]]: i8, %[[C_ARG:.+]]: i32)
    # CHECK-NEXT:   %[[A_CAST:.+]] = sexti %[[A_ARG]] : i8 to i32
    # CHECK-NEXT:   %[[B_CAST:.+]] = sexti %[[B_ARG]] : i8 to i32
    # CHECK-NEXT:   %[[MUL:.+]] = muli %[[A_CAST]], %[[B_CAST]] : i32
    # CHECK-NEXT:   %[[ADD:.+]] = addi %[[C_ARG]], %[[MUL]] : i32
    # CHECK-NEXT:   linalg.yield %[[ADD]] : i32
    # CHECK-NEXT: -> tensor<4x8xi32>
    @builtin.FuncOp.from_py_func(
        RankedTensorType.get((4, 16), i8), RankedTensorType.get((16, 8), i8),
        RankedTensorType.get((4, 8), i32))
    def test_i8i8i32_matmul(lhs, rhs, init_result):
      return matmul_poly(lhs, rhs, outs=[init_result])

    # CHECK-LABEL: @test_i8i16i32_matmul
    # CHECK:      ^{{.*}}(%[[A_ARG:.+]]: i8, %[[B_ARG:.+]]: i16, %[[C_ARG:.+]]: i32)
    # CHECK-NEXT:   %[[A_CAST:.+]] = sexti %[[A_ARG]] : i8 to i32
    # CHECK-NEXT:   %[[B_CAST:.+]] = sexti %[[B_ARG]] : i16 to i32
    # CHECK-NEXT:   %[[MUL:.+]] = muli %[[A_CAST]], %[[B_CAST]] : i32
    # CHECK-NEXT:   %[[ADD:.+]] = addi %[[C_ARG]], %[[MUL]] : i32
    # CHECK-NEXT:   linalg.yield %[[ADD]] : i32
    # CHECK-NEXT: -> tensor<4x8xi32>
    @builtin.FuncOp.from_py_func(
        RankedTensorType.get((4, 16), i8), RankedTensorType.get((16, 8), i16),
        RankedTensorType.get((4, 8), i32))
    def test_i8i16i32_matmul(lhs, rhs, init_result):
      return matmul_poly(lhs, rhs, outs=[init_result])

    # CHECK-LABEL: @test_i32i32i16_matmul
    # CHECK:      ^{{.*}}(%[[A_ARG:.+]]: i32, %[[B_ARG:.+]]: i32, %[[C_ARG:.+]]: i16)
    # CHECK-NEXT:   %[[A_CAST:.+]] = trunci %[[A_ARG]] : i32 to i16
    # CHECK-NEXT:   %[[B_CAST:.+]] = trunci %[[B_ARG]] : i32 to i16
    # CHECK-NEXT:   %[[MUL:.+]] = muli %[[A_CAST]], %[[B_CAST]] : i16
    # CHECK-NEXT:   %[[ADD:.+]] = addi %[[C_ARG]], %[[MUL]] : i16
    # CHECK-NEXT:   linalg.yield %[[ADD]] : i16
    # CHECK-NEXT: -> tensor<4x8xi16>
    @builtin.FuncOp.from_py_func(
        RankedTensorType.get((4, 16), i32), RankedTensorType.get((16, 8), i32),
        RankedTensorType.get((4, 8), i16))
    def test_i32i32i16_matmul(lhs, rhs, init_result):
      return matmul_poly(lhs, rhs, outs=[init_result])

    # CHECK-LABEL: @test_i8i8f32_matmul
    # CHECK:      ^{{.*}}(%[[A_ARG:.+]]: i8, %[[B_ARG:.+]]: i8, %[[C_ARG:.+]]: f32)
    # CHECK-NEXT:   %[[A_CAST:.+]] = sitofp %[[A_ARG]] : i8 to f32
    # CHECK-NEXT:   %[[B_CAST:.+]] = sitofp %[[B_ARG]] : i8 to f32
    # CHECK-NEXT:   %[[MUL:.+]] = mulf %[[A_CAST]], %[[B_CAST]] : f32
    # CHECK-NEXT:   %[[ADD:.+]] = addf %[[C_ARG]], %[[MUL]] : f32
    # CHECK-NEXT:   linalg.yield %[[ADD]] : f32
    # CHECK-NEXT: -> tensor<4x8xf32>
    @builtin.FuncOp.from_py_func(
        RankedTensorType.get((4, 16), i8), RankedTensorType.get((16, 8), i8),
        RankedTensorType.get((4, 8), f32))
    def test_i8i8f32_matmul(lhs, rhs, init_result):
      return matmul_poly(lhs, rhs, outs=[init_result])

    # CHECK-LABEL: @test_f16f16f32_matmul
    # CHECK:      ^{{.*}}(%[[A_ARG:.+]]: f16, %[[B_ARG:.+]]: f16, %[[C_ARG:.+]]: f32)
    # CHECK-NEXT:   %[[A_CAST:.+]] = fpext %[[A_ARG]] : f16 to f32
    # CHECK-NEXT:   %[[B_CAST:.+]] = fpext %[[B_ARG]] : f16 to f32
    # CHECK-NEXT:   %[[MUL:.+]] = mulf %[[A_CAST]], %[[B_CAST]] : f32
    # CHECK-NEXT:   %[[ADD:.+]] = addf %[[C_ARG]], %[[MUL]] : f32
    # CHECK-NEXT:   linalg.yield %[[ADD]] : f32
    # CHECK-NEXT: -> tensor<4x8xf32>
    @builtin.FuncOp.from_py_func(
        RankedTensorType.get((4, 16), f16), RankedTensorType.get((16, 8), f16),
        RankedTensorType.get((4, 8), f32))
    def test_f16f16f32_matmul(lhs, rhs, init_result):
      return matmul_poly(lhs, rhs, outs=[init_result])

    # CHECK-LABEL: @test_f64f64f32_matmul
    # CHECK:      ^{{.*}}(%[[A_ARG:.+]]: f64, %[[B_ARG:.+]]: f64, %[[C_ARG:.+]]: f32)
    # CHECK-NEXT:   %[[A_CAST:.+]] = fptrunc %[[A_ARG]] : f64 to f32
    # CHECK-NEXT:   %[[B_CAST:.+]] = fptrunc %[[B_ARG]] : f64 to f32
    # CHECK-NEXT:   %[[MUL:.+]] = mulf %[[A_CAST]], %[[B_CAST]] : f32
    # CHECK-NEXT:   %[[ADD:.+]] = addf %[[C_ARG]], %[[MUL]] : f32
    # CHECK-NEXT:   linalg.yield %[[ADD]] : f32
    # CHECK-NEXT: -> tensor<4x8xf32>
    @builtin.FuncOp.from_py_func(
        RankedTensorType.get((4, 16), f64), RankedTensorType.get((16, 8), f64),
        RankedTensorType.get((4, 8), f32))
    def test_f64f64f32_matmul(lhs, rhs, init_result):
      return matmul_poly(lhs, rhs, outs=[init_result])

    # CHECK-LABEL: @test_f32i32_conv
    # CHECK: linalg.generic
    # CHECK-SAME: indexing_maps = [#[[$CONV_MAP_I]], #[[$CONV_MAP_K]], #[[$CONV_MAP_O]]]
    # CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "parallel"]
    # CHECK:      ^{{.*}}(%[[IN:.+]]: f32, %[[FILTER:.+]]: f32, %[[OUT:.+]]: i32)
    # CHECK-NEXT:   %[[IN_CAST:.+]] = fptosi %[[IN:.+]] : f32 to i32
    # CHECK-NEXT:   %[[FILTER_CAST:.+]] = fptosi %[[FILTER:.+]] : f32 to i32
    # CHECK-NEXT:   %[[PROD:.+]] = muli %[[IN_CAST]], %[[FILTER_CAST]] : i32
    # CHECK-NEXT:   %[[SUM:.+]] = addi %[[OUT]], %[[PROD]] : i32
    # CHECK-NEXT:   linalg.yield %[[SUM]] : i32
    # CHECK-NEXT: -> tensor<2x4xi32>
    @builtin.FuncOp.from_py_func(
        RankedTensorType.get((4, 16), f32), RankedTensorType.get((2, 2, 1),
                                                                 f32),
        RankedTensorType.get((2, 4), i32))
    def test_f32i32_conv(input, filter, init_result):
      return conv_poly(
          input, filter, outs=[init_result], strides=[2, 4], dilations=[1, 2])

    # CHECK-LABEL: @test_f32i32_max_pooling
    # CHECK: linalg.generic
    # CHECK-SAME: indexing_maps = [#[[$CONV_MAP_I]], #[[$POOL_MAP_K]], #[[$CONV_MAP_O]]]
    # CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "parallel"]
    # CHECK:      ^{{.*}}(%[[IN:.+]]: f32, %[[SHAPE:.+]]: f32, %[[OUT:.+]]: i32)
    # CHECK-NEXT:   %[[IN_CAST:.+]] = fptosi %[[IN:.+]] : f32 to i32
    # CHECK-NEXT:   %[[MAX:.+]] = maxsi %[[OUT]], %[[IN_CAST:.+]] : i32
    # CHECK-NEXT:   linalg.yield %[[MAX]] : i32
    # CHECK-NEXT: -> tensor<2x4xi32>
    @builtin.FuncOp.from_py_func(
        RankedTensorType.get((4, 16), f32), RankedTensorType.get((2, 2), f32),
        RankedTensorType.get((2, 4), i32))
    def test_f32i32_max_pooling(input, shape, init_result):
      return pooling_max_poly(
          input, shape, outs=[init_result], strides=[2, 4], dilations=[1, 2])

    # CHECK-LABEL: @test_f32f32_max_pooling
    # CHECK: linalg.generic
    # CHECK-SAME: indexing_maps = [#[[$CONV_MAP_I]], #[[$POOL_MAP_K]], #[[$CONV_MAP_O]]]
    # CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "parallel"]
    # CHECK:      ^{{.*}}(%[[IN:.+]]: f32, %[[SHAPE:.+]]: f32, %[[OUT:.+]]: f32)
    # CHECK-NEXT:   %[[MAX:.+]] = maxf %[[OUT]], %[[IN:.+]] : f32
    # CHECK-NEXT:   linalg.yield %[[MAX]] : f32
    # CHECK-NEXT: -> tensor<2x4xf32>
    @builtin.FuncOp.from_py_func(
        RankedTensorType.get((4, 16), f32), RankedTensorType.get((2, 2), f32),
        RankedTensorType.get((2, 4), f32))
    def test_f32f32_max_pooling(input, shape, init_result):
      return pooling_max_poly(
          input, shape, outs=[init_result], strides=[2, 4], dilations=[1, 2])

    # CHECK-LABEL: @test_f32i32_min_pooling
    # CHECK:   = minsi
    @builtin.FuncOp.from_py_func(
        RankedTensorType.get((4, 16), f32), RankedTensorType.get((2, 2), f32),
        RankedTensorType.get((2, 4), i32))
    def test_f32i32_min_pooling(input, shape, init_result):
      return pooling_min_poly(
          input, shape, outs=[init_result], strides=[2, 4], dilations=[1, 2])

    # CHECK-LABEL: @test_f32f32_min_pooling
    # CHECK:   = minf
    @builtin.FuncOp.from_py_func(
        RankedTensorType.get((4, 16), f32), RankedTensorType.get((2, 2), f32),
        RankedTensorType.get((2, 4), f32))
    def test_f32f32_min_pooling(input, shape, init_result):
      return pooling_min_poly(
          input, shape, outs=[init_result], strides=[2, 4], dilations=[1, 2])

    # CHECK-LABEL: @test_i32_fill_rng
    # CHECK:      ^{{.*}}(%[[MIN:.+]]: f64, %[[MAX:.+]]: f64, %[[SEED:.+]]: i32, %{{.*}}
    # CHECK-DAG:    %[[IDX0:.+]] = linalg.index 0 : index
    # CHECK-DAG:    %[[IDX0_CAST:.+]] = index_cast %[[IDX0]] : index to i32
    # CHECK-DAG:    %[[RND0:.+]] = addi %[[IDX0_CAST]], %[[SEED]] : i32
    # CHECK-DAG:    %[[CST0:.+]] = constant 1103515245 : i64
    # CHECK-DAG:    %[[CST0_CAST:.+]] = trunci %[[CST0]] : i64 to i32
    # Skip the remaining random number computation and match the scaling logic.
    # CHECK-DAG:    %[[DIFF:.+]] = subf %[[MAX]], %[[MIN]] : f64
    # CHECK-DAG:    %[[CST3:.+]] = constant 2.3283063999999999E-10 : f64
    # CHECK-DAG:    %[[FACT:.+]] = mulf %[[DIFF]], %[[CST3]] : f64
    # CHECK-DAG:    %[[RND4:.+]] = mulf %{{.+}}, %[[FACT]] : f64
    # CHECK-DAG:    %[[RND5:.+]] = addf %[[RND4]], %[[MIN]] : f64
    # CHECK-DAG:    %{{.*}} = fptosi %[[RND5]] : f64 to i32
    @builtin.FuncOp.from_py_func(f64, f64, i32,
                                 RankedTensorType.get((4, 16), i32))
    def test_i32_fill_rng(min, max, seed, init_result):
      return fill_rng_poly(min, max, seed, outs=[init_result])

    # CHECK-LABEL: @test_f32_soft_plus
    # CHECK:      ^{{.*}}(%[[IN:.+]]: f32, %[[OUT:.+]]: f32)
    # CHECK-NEXT:   %[[C1:.+]] = constant 1.000000e+00 : f64
    # CHECK-NEXT:   %[[C1_CAST:.+]] = fptrunc %[[C1]] : f64 to f32
    # CHECK-NEXT:   %[[EXP:.+]] = math.exp %[[IN]] : f32
    # CHECK-NEXT:   %[[SUM:.+]] = addf %[[C1_CAST]], %[[EXP]] : f32
    # CHECK-NEXT:   %[[LOG:.+]] = math.log %[[SUM]] : f32
    # CHECK-NEXT:   linalg.yield %[[LOG]] : f32
    # CHECK-NEXT: -> tensor<4x16xf32>
    @builtin.FuncOp.from_py_func(
        RankedTensorType.get((4, 16), f32), RankedTensorType.get((4, 16), f32))
    def test_f32_soft_plus(input, init_result):
      return soft_plus_poly(input, outs=[init_result])


print(module)
