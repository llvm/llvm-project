# RUN: %PYTHON %s | FileCheck %s

from typing import Optional, Sequence

from mlir.ir import *
from mlir.dialects import builtin
from mlir.dialects import linalg
from mlir.dialects import std

from mlir.dialects.linalg.opdsl.lang import *


@linalg_structured_op
def matmul_mono(A=TensorDef(T, S.M, S.K),
                B=TensorDef(T, S.K, S.N),
                C=TensorDef(T, S.M, S.N, output=True)):
  C[D.m, D.n] += A[D.m, D.k] * B[D.k, D.n]


@linalg_structured_op
def matmul_poly(A=TensorDef(TV.T1, S.M, S.K),
                B=TensorDef(TV.T2, S.K, S.N),
                C=TensorDef(U, S.M, S.N, output=True)):
  C[D.m, D.n] += cast(U, A[D.m, D.k]) * cast(U, B[D.k, D.n])


with Context() as ctx, Location.unknown():
  module = Module.create()
  f16 = F16Type.get()
  f32 = F32Type.get()
  f64 = F64Type.get()
  i8 = IntegerType.get_signless(8)
  i16 = IntegerType.get_signless(16)
  i32 = IntegerType.get_signless(32)
  with InsertionPoint(module.body):

    # Note that these all have the same indexing maps. We verify the first and
    # then do more permutation tests on casting and body generation
    # behavior.
    # CHECK: #[[$MAPA:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
    # CHECK: #[[$MAPB:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
    # CHECK: #[[$MAPC:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>

    # CHECK-LABEL: func @test_matmul_mono
    # CHECK-SAME:  %[[A:.+]]: tensor<4x16xf32>
    # CHECK-SAME: %[[B:.+]]: tensor<16x8xf32>

    # CHECK: %[[INITC:.+]] = linalg.init_tensor [4, 8] : tensor<4x8xf32>
    # CHECK: linalg.generic
    # CHECK-SAME: indexing_maps = [#[[$MAPA]], #[[$MAPB]], #[[$MAPC]]]
    # CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]
    # CHECK-SAME: ins(%[[A]], %[[B]]
    # CHECK-SAME: outs(%[[INITC]]

    @builtin.FuncOp.from_py_func(RankedTensorType.get((4, 16), f32),
                                 RankedTensorType.get((16, 8), f32))
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
    @builtin.FuncOp.from_py_func(RankedTensorType.get((4, 16), i8),
                                 RankedTensorType.get((16, 8), i8),
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
    @builtin.FuncOp.from_py_func(RankedTensorType.get((4, 16), i8),
                                 RankedTensorType.get((16, 8), i16),
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
    @builtin.FuncOp.from_py_func(RankedTensorType.get((4, 16), i32),
                                 RankedTensorType.get((16, 8), i32),
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
    @builtin.FuncOp.from_py_func(RankedTensorType.get((4, 16), i8),
                                 RankedTensorType.get((16, 8), i8),
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
    @builtin.FuncOp.from_py_func(RankedTensorType.get((4, 16), f16),
                                 RankedTensorType.get((16, 8), f16),
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
    @builtin.FuncOp.from_py_func(RankedTensorType.get((4, 16), f64),
                                 RankedTensorType.get((16, 8), f64),
                                 RankedTensorType.get((4, 8), f32))
    def test_f64f64f32_matmul(lhs, rhs, init_result):
      return matmul_poly(lhs, rhs, outs=[init_result])


print(module)
