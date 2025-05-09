// RUN: mlir-opt %s -linalg-generalize-named-ops -split-input-file | FileCheck %s
// CHECK: #[[IDENTITY:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
//
// CHECK: @unary_exp(%[[A:.+]]: tensor<8x16x32xf32>, %[[B:.+]]: tensor<8x16x32xf32>)
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[IDENTITY]], #[[IDENTITY]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel"]
// CHECK-SAME:  ins(%[[A]]
// CHECK-SAME: outs(%[[B]]
//
// CHECK: ^{{.*}}(%[[A_ARG:.+]]: f32, %[[B_ARG:.+]]: f32)
// CHECK:   %[[EXP:.+]] = math.exp %[[A_ARG]] : f32
// CHECK:   linalg.yield %[[EXP]] : f32
//
func.func @unary_exp(%A : tensor<8x16x32xf32>, %B: tensor<8x16x32xf32>) ->  tensor<8x16x32xf32> {
  %r = linalg.elementwise
      kind=#linalg.elementwise_kind<exp>
      ins(%A : tensor<8x16x32xf32>)
      outs(%B: tensor<8x16x32xf32>) -> tensor<8x16x32xf32>
  return %r : tensor<8x16x32xf32>
}
// -----
// CHECK-DAG: #[[IDENTITY:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-DAG: #[[PROJECTION:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
//
// CHECK: @unary_transpose_broadcast_tanh(%[[A:.+]]: tensor<32x16xf32>, %[[B:.+]]: tensor<8x16x32xf32>)
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[PROJECTION]], #[[IDENTITY]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel"]
// CHECK-SAME:  ins(%[[A]]
// CHECK-SAME: outs(%[[B]]
//
// CHECK: ^{{.*}}(%[[A_ARG:.+]]: f32, %[[B_ARG:.+]]: f32)
// CHECK:   %[[TANH:.+]] = math.tanh %[[A_ARG]] : f32
// CHECK:   linalg.yield %[[TANH]] : f32
//
func.func @unary_transpose_broadcast_tanh(%A : tensor<32x16xf32>, %B: tensor<8x16x32xf32>) ->  tensor<8x16x32xf32> {
  %r = linalg.elementwise
      kind=#linalg.elementwise_kind<tanh>
      indexing_maps = [affine_map<(d0, d1, d2) -> (d2, d1)>,
                       affine_map<(d0, d1, d2) -> (d0, d1, d2)>]
      ins(%A : tensor<32x16xf32>)
      outs(%B: tensor<8x16x32xf32>) -> tensor<8x16x32xf32>
  return %r : tensor<8x16x32xf32>
}
// -----
// CHECK: #[[MAP:.+]] = affine_map<(d0, d1) -> (d0, d1)>
//
// CHECK: @binary_div_on_memrefs(%[[A:.+]]: memref<16x8xf32>, %[[B:.+]]: memref<16x8xf32>, %[[C:.+]]: memref<16x8xf32>)
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP]], #[[MAP]], #[[MAP]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel"]
// CHECK-SAME:  ins(%[[A]], %[[B]]
// CHECK-SAME: outs(%[[C]]
//
// CHECK: ^{{.*}}(%[[A_ARG:.+]]: f32, %[[B_ARG:.+]]: f32, %[[C_ARG:.+]]: f32)
// CHECK:   %[[DIV:.+]] = arith.divf %[[A_ARG]], %[[B_ARG]] : f32
// CHECK:   linalg.yield %[[DIV]] : f32
//
func.func @binary_div_on_memrefs(%A : memref<16x8xf32>, %B: memref<16x8xf32>, %C: memref<16x8xf32>) {
  linalg.elementwise
      kind=#linalg.elementwise_kind<div>
      ins(%A, %B: memref<16x8xf32>, memref<16x8xf32>)
      outs(%C: memref<16x8xf32>)
  return
}
// -----
// CHECK: #[[MAP:.+]] = affine_map<(d0, d1) -> (d0, d1)>
//
// CHECK: @binary_mul_on_tensors(%[[A:.+]]: tensor<16x8xf32>, %[[B:.+]]: tensor<16x8xf32>, %[[C:.+]]: tensor<16x8xf32>)
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP]], #[[MAP]], #[[MAP]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel"]
// CHECK-SAME:  ins(%[[A]], %[[B]]
// CHECK-SAME: outs(%[[C]]
//
// CHECK: ^{{.*}}(%[[A_ARG:.+]]: f32, %[[B_ARG:.+]]: f32, %[[C_ARG:.+]]: f32)
// CHECK:   %[[MUL:.+]] = arith.mulf %[[A_ARG]], %[[B_ARG]] : f32
// CHECK:   linalg.yield %[[MUL]] : f32
//
func.func @binary_mul_on_tensors(%A : tensor<16x8xf32>, %B: tensor<16x8xf32>, %C: tensor<16x8xf32>) ->  tensor<16x8xf32> {
  %r = linalg.elementwise
      kind=#linalg.elementwise_kind<mul>
      ins(%A, %B: tensor<16x8xf32>, tensor<16x8xf32>)
      outs(%C: tensor<16x8xf32>) -> tensor<16x8xf32>
  return %r : tensor<16x8xf32>
}
// -----
// CHECK-DAG: #[[IDENTITY:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: #[[TRANSPOSE:.+]] = affine_map<(d0, d1) -> (d1, d0)>
//
// CHECK: @binary_transpose_a(%[[A:.+]]: tensor<8x16xf32>, %[[B:.+]]: tensor<16x8xf32>, %[[C:.+]]: tensor<16x8xf32>)
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[TRANSPOSE]], #[[IDENTITY]], #[[IDENTITY]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel"]
// CHECK-SAME:  ins(%[[A]], %[[B]]
// CHECK-SAME: outs(%[[C]]
//
// CHECK: ^{{.*}}(%[[A_ARG:.+]]: f32, %[[B_ARG:.+]]: f32, %[[C_ARG:.+]]: f32)
// CHECK:   %[[SUB:.+]] = arith.subf %[[A_ARG]], %[[B_ARG]] : f32
// CHECK:   linalg.yield %[[SUB]] : f32
//
func.func @binary_transpose_a(%A : tensor<8x16xf32>, %B: tensor<16x8xf32>, %C: tensor<16x8xf32>) ->  tensor<16x8xf32> {
  %r = linalg.elementwise
      kind=#linalg.elementwise_kind<sub>
      indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>,
                       affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>]
      ins(%A, %B: tensor<8x16xf32>, tensor<16x8xf32>)
      outs(%C: tensor<16x8xf32>) -> tensor<16x8xf32>
  return %r : tensor<16x8xf32>
}
// -----
// CHECK-DAG: #[[IDENTITY:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: #[[TRANSPOSE:.+]] = affine_map<(d0, d1) -> (d1, d0)>
// CHECK-DAG: #[[BROADCAST:.+]] = affine_map<(d0, d1) -> (d0)>
//
// CHECK: @binary_transpose_a_broadcast_b(%[[A:.+]]: tensor<8x16xf32>, %[[B:.+]]: tensor<16xf32>, %[[C:.+]]: tensor<16x8xf32>)
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[TRANSPOSE]], #[[BROADCAST]], #[[IDENTITY]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel"]
// CHECK-SAME:  ins(%[[A]], %[[B]]
// CHECK-SAME: outs(%[[C]]
//
// CHECK: ^{{.*}}(%[[A_ARG:.+]]: f32, %[[B_ARG:.+]]: f32, %[[C_ARG:.+]]: f32)
// CHECK:   %[[ADD:.+]] = arith.addf %[[A_ARG]], %[[B_ARG]] : f32
// CHECK:   linalg.yield %[[ADD]] : f32
//
func.func @binary_transpose_a_broadcast_b(%A : tensor<8x16xf32>, %B: tensor<16xf32>, %C: tensor<16x8xf32>) ->  tensor<16x8xf32> {
  %r = linalg.elementwise
      kind=#linalg.elementwise_kind<add>
      indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>,
                       affine_map<(d0, d1) -> (d0)>,
                       affine_map<(d0, d1) -> (d0, d1)>]
      ins(%A, %B: tensor<8x16xf32>, tensor<16xf32>)
      outs(%C: tensor<16x8xf32>) -> tensor<16x8xf32>
  return %r : tensor<16x8xf32>
}
// -----
// CHECK-DAG: #[[IDENTITY:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-DAG: #[[PROJECTION:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
//
// CHECK: @ternary(%[[A:.+]]: tensor<32x16xi1>, %[[B:.+]]: tensor<8x16x32xf32>, %[[C:.+]]: tensor<8x16x32xf32>, %[[D:.+]]: tensor<8x16x32xf32>)
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[PROJECTION]], #[[IDENTITY]], #[[IDENTITY]], #[[IDENTITY]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel"]
//
// CHECK-SAME:  ins(%[[A]], %[[B]], %[[C]]
// CHECK-SAME: outs(%[[D]]
//
// CHECK: ^{{.*}}(%[[A_ARG:.+]]: i1, %[[B_ARG:.+]]: f32, %[[C_ARG:.+]]: f32, %[[D_ARG:.+]]: f32)
// CHECK:   %[[SELECTED:.+]] = arith.select %[[A_ARG]], %[[B_ARG]], %[[C_ARG]] : f32
// CHECK:   linalg.yield %[[SELECTED]] : f32
//
func.func @ternary(%A : tensor<32x16xi1>, %B: tensor<8x16x32xf32>, %C : tensor<8x16x32xf32>, %D : tensor<8x16x32xf32>) ->  tensor<8x16x32xf32> {
  %r = linalg.elementwise
      kind=#linalg.elementwise_kind<select>
      indexing_maps = [affine_map<(d0, d1, d2) -> (d2, d1)>,
                       affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1, d2)>]
      ins(%A, %B, %C : tensor<32x16xi1>, tensor<8x16x32xf32>, tensor<8x16x32xf32>)
      outs(%D: tensor<8x16x32xf32>) -> tensor<8x16x32xf32>
  return %r : tensor<8x16x32xf32>
}
