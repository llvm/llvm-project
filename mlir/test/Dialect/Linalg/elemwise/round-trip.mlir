// RUN: mlir-opt %s -split-input-file | FileCheck %s

// CHECK: @unary_exp(%[[A:.+]]: tensor<8x16x32xf32>, %[[B:.+]]: tensor<8x16x32xf32>)
// CHECK: %{{.*}} = linalg.elemwise func_type=#linalg.elemwise_fn<exp>
// CHECK-SAME         ins(%[[A:.+]] : tensor<8x16x32xf32>) outs(%[[B:.+]] : tensor<8x16x32xf32>)
//
func.func @unary_exp(%A : tensor<8x16x32xf32>, %B: tensor<8x16x32xf32>) ->  tensor<8x16x32xf32> {
  %r = linalg.elemwise
               func_type=#linalg.elemwise_fn<exp>
               ins(%A : tensor<8x16x32xf32>)
               outs(%B: tensor<8x16x32xf32>) -> tensor<8x16x32xf32>
  return %r : tensor<8x16x32xf32>
}

// -----

// CHECK-DAG: #[[IDENTITY:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-DAG: #[[PROJECTION:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
//
// CHECK: @unary_transpose_broadcast_tanh(%[[A:.+]]: tensor<?x16xf32>, %[[B:.+]]: tensor<8x16x?xf32>) ->  tensor<8x16x?xf32> {
// CHECK: {{.*}} = linalg.elemwise func_type=#linalg.elemwise_fn<tanh>
// CHECK-SAME:       indexing_maps = [#[[PROJECTION]], #[[IDENTITY]]]
// CHECK-SAME:       ins(%[[A]] : tensor<?x16xf32>) outs(%[[B]] : tensor<8x16x?xf32>) -> tensor<8x16x?xf32>
//
func.func @unary_transpose_broadcast_tanh(%A : tensor<?x16xf32>, %B: tensor<8x16x?xf32>) ->  tensor<8x16x?xf32> {
  %r = linalg.elemwise
               func_type=#linalg.elemwise_fn<tanh>
               indexing_maps = [affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>]
               ins(%A : tensor<?x16xf32>)
               outs(%B: tensor<8x16x?xf32>) -> tensor<8x16x?xf32>
  return %r : tensor<8x16x?xf32>
}

// -----

// CHECK: @binary_mul(%[[A:.+]]: tensor<16x8xf32>, %[[B:.+]]: tensor<16x8xf32>, %[[C:.+]]: tensor<16x8xf32>) ->  tensor<16x8xf32> {
// CHECK: {{.*}} = linalg.elemwise func_type=#linalg.elemwise_fn<mul> ins(%[[A]], %[[B]] : tensor<16x8xf32>, tensor<16x8xf32>) outs(%[[C]] : tensor<16x8xf32>) -> tensor<16x8xf32>
//
func.func @binary_mul(%A : tensor<16x8xf32>, %B : tensor<16x8xf32>, %C : tensor<16x8xf32>) ->  tensor<16x8xf32> {
  %r = linalg.elemwise
               func_type=#linalg.elemwise_fn<mul>
               ins(%A, %B: tensor<16x8xf32>, tensor<16x8xf32>)
               outs(%C: tensor<16x8xf32>) -> tensor<16x8xf32>
  return %r : tensor<16x8xf32>
}

// -----

// CHECK: @multiple_dims(%[[A]]: tensor<1x2x3x4x5xi32>, %[[B:.+]]: tensor<1x2x3x4x5xi32>, %[[C:.+]]: tensor<1x2x3x4x5xi32>) -> tensor<1x2x3x4x5xi32> {
// CHECK: {{.*}} = linalg.elemwise func_type=#linalg.elemwise_fn<mul> ins(%[[A]], %[[B]] : tensor<1x2x3x4x5xi32>, tensor<1x2x3x4x5xi32>) outs(%[[C]] : tensor<1x2x3x4x5xi32>) -> tensor<1x2x3x4x5xi32>
//
func.func @multiple_dims(%arg0 : tensor<1x2x3x4x5xi32>, %arg1 : tensor<1x2x3x4x5xi32>, %arg2 : tensor<1x2x3x4x5xi32>) ->  tensor<1x2x3x4x5xi32> {
  %r = linalg.elemwise
               func_type=#linalg.elemwise_fn<mul>
               ins(%arg0, %arg1: tensor<1x2x3x4x5xi32>, tensor<1x2x3x4x5xi32>)
               outs(%arg2: tensor<1x2x3x4x5xi32>) -> tensor<1x2x3x4x5xi32>
  return %r : tensor<1x2x3x4x5xi32>
}

// -----

// CHECK: @redundant_maps
// CHECK-NOT: indexing_maps
//
#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
func.func @redundant_maps(%arg0 : tensor<1x2x3x4x5xi32>, %arg1 : tensor<1x2x3x4x5xi32>, %arg2 : tensor<1x2x3x4x5xi32>) ->  tensor<1x2x3x4x5xi32> {
  %r = linalg.elemwise   func_type=#linalg.elemwise_fn<mul>
         indexing_maps = [#map, #map, #map]
         ins(%arg0, %arg1: tensor<1x2x3x4x5xi32>, tensor<1x2x3x4x5xi32>)
         outs(%arg2: tensor<1x2x3x4x5xi32>) -> tensor<1x2x3x4x5xi32>
  return %r : tensor<1x2x3x4x5xi32>
}
