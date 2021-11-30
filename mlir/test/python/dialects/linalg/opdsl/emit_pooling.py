# RUN: %PYTHON %s | FileCheck %s

from mlir.ir import *
from mlir.dialects import builtin
from mlir.dialects import linalg
from mlir.dialects import std

from mlir.dialects.linalg.opdsl.lang import *

T1 = TV.T1
T2 = TV.T2


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
def pooling_max_unsigned_poly(
    I=TensorDef(T1, S.N, S.H, S.W, S.C),
    K=TensorDef(T2, S.KH, S.KW, index_dims=[D.kh, D.kw]),
    O=TensorDef(U, S.N, S.OH, S.OW, S.C, output=True),
    strides=AttributeDef(S.SH, S.SW),
    dilations=AttributeDef(S.DH, S.DW)):
  domain(D.n, D.oh, D.ow, D.kh, D.kw, D.c)
  O[D.n, D.oh, D.ow, D.c] = ReduceFn.max_unsigned(D.kh, D.kw)(
      cast_unsigned(
          U, I[D.n, D.oh * S.SH + D.kh * S.DH, D.ow * S.SW + D.kw * S.DW, D.c]))


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
def pooling_min_unsigned_poly(
    I=TensorDef(T1, S.N, S.H, S.W, S.C),
    K=TensorDef(T2, S.KH, S.KW, index_dims=[D.kh, D.kw]),
    O=TensorDef(U, S.N, S.OH, S.OW, S.C, output=True),
    strides=AttributeDef(S.SH, S.SW),
    dilations=AttributeDef(S.DH, S.DW)):
  domain(D.n, D.oh, D.ow, D.kh, D.kw, D.c)
  O[D.n, D.oh, D.ow, D.c] = ReduceFn.min_unsigned(D.kh, D.kw)(
      cast_unsigned(
          U, I[D.n, D.oh * S.SH + D.kh * S.DH, D.ow * S.SW + D.kw * S.DW, D.c]))


with Context() as ctx, Location.unknown():
  module = Module.create()
  f32 = F32Type.get()
  i32 = IntegerType.get_signless(32)
  with InsertionPoint(module.body):

    # Pooling indexing maps.
    # CHECK: #[[$POOL_MAP_I:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1 * 2 + d3, d2 * 4 + d4 * 2, d5)>
    # CHECK: #[[$POOL_MAP_K:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4)>
    # CHECK: #[[$POOL_MAP_O:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d5)>

    # CHECK-LABEL: @test_f32i32_max_pooling
    # CHECK: linalg.generic
    # CHECK-SAME: indexing_maps = [#[[$POOL_MAP_I]], #[[$POOL_MAP_K]], #[[$POOL_MAP_O]]]
    # CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "parallel"]
    # CHECK:      ^{{.*}}(%[[IN:.+]]: f32, %[[SHAPE:.+]]: f32, %[[OUT:.+]]: i32)
    # CHECK-NEXT:   %[[IN_CAST:.+]] = arith.fptosi %[[IN:.+]] : f32 to i32
    # CHECK-NEXT:   %[[MAX:.+]] = arith.maxsi %[[OUT]], %[[IN_CAST:.+]] : i32
    # CHECK-NEXT:   linalg.yield %[[MAX]] : i32
    # CHECK-NEXT: -> tensor<1x2x4x1xi32>
    @builtin.FuncOp.from_py_func(
        RankedTensorType.get((1, 4, 16, 1), f32),
        RankedTensorType.get((2, 2), f32),
        RankedTensorType.get((1, 2, 4, 1), i32))
    def test_f32i32_max_pooling(input, shape, init_result):
      return pooling_max_poly(
          input, shape, outs=[init_result], strides=[2, 4], dilations=[1, 2])

    # CHECK-LABEL: @test_f32i32_max_unsigned_pooling
    # CHECK:   = arith.fptoui
    # CHECK:   = arith.maxui
    @builtin.FuncOp.from_py_func(
        RankedTensorType.get((1, 4, 16, 1), f32),
        RankedTensorType.get((2, 2), f32),
        RankedTensorType.get((1, 2, 4, 1), i32))
    def test_f32i32_max_unsigned_pooling(input, shape, init_result):
      return pooling_max_unsigned_poly(
          input, shape, outs=[init_result], strides=[2, 4], dilations=[1, 2])

    # CHECK-LABEL: @test_f32f32_max_pooling
    # CHECK: linalg.generic
    # CHECK-SAME: indexing_maps = [#[[$POOL_MAP_I]], #[[$POOL_MAP_K]], #[[$POOL_MAP_O]]]
    # CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "parallel"]
    # CHECK:      ^{{.*}}(%[[IN:.+]]: f32, %[[SHAPE:.+]]: f32, %[[OUT:.+]]: f32)
    # CHECK-NEXT:   %[[MAX:.+]] = arith.maxf %[[OUT]], %[[IN:.+]] : f32
    # CHECK-NEXT:   linalg.yield %[[MAX]] : f32
    # CHECK-NEXT: -> tensor<1x2x4x1xf32>
    @builtin.FuncOp.from_py_func(
        RankedTensorType.get((1, 4, 16, 1), f32),
        RankedTensorType.get((2, 2), f32),
        RankedTensorType.get((1, 2, 4, 1), f32))
    def test_f32f32_max_pooling(input, shape, init_result):
      return pooling_max_poly(
          input, shape, outs=[init_result], strides=[2, 4], dilations=[1, 2])

    # CHECK-LABEL: @test_f32i32_min_pooling
    # CHECK:   = arith.fptosi
    # CHECK:   = arith.minsi
    @builtin.FuncOp.from_py_func(
        RankedTensorType.get((1, 4, 16, 1), f32),
        RankedTensorType.get((2, 2), f32),
        RankedTensorType.get((1, 2, 4, 1), i32))
    def test_f32i32_min_pooling(input, shape, init_result):
      return pooling_min_poly(
          input, shape, outs=[init_result], strides=[2, 4], dilations=[1, 2])

    # CHECK-LABEL: @test_f32i32_min_unsigned_pooling
    # CHECK:   = arith.fptoui
    # CHECK:   = arith.minui
    @builtin.FuncOp.from_py_func(
        RankedTensorType.get((1, 4, 16, 1), f32),
        RankedTensorType.get((2, 2), f32),
        RankedTensorType.get((1, 2, 4, 1), i32))
    def test_f32i32_min_unsigned_pooling(input, shape, init_result):
      return pooling_min_unsigned_poly(
          input, shape, outs=[init_result], strides=[2, 4], dilations=[1, 2])

    # CHECK-LABEL: @test_f32f32_min_pooling
    # CHECK:   = arith.minf
    @builtin.FuncOp.from_py_func(
        RankedTensorType.get((1, 4, 16, 1), f32),
        RankedTensorType.get((2, 2), f32),
        RankedTensorType.get((1, 2, 4, 1), f32))
    def test_f32f32_min_pooling(input, shape, init_result):
      return pooling_min_poly(
          input, shape, outs=[init_result], strides=[2, 4], dilations=[1, 2])


print(module)
