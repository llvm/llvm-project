// RUN: mlir-opt %s -split-input-file | FileCheck %s
//
// Note - the functions are named @{unary|binary}_{identity|transpose|broadcast|transpose_a|...}_{exp|mul|div|..}

// CHECK: @unary_identity_exp(%[[A:.+]]: tensor<8x16x32xf32>, %[[B:.+]]: tensor<8x16x32xf32>)
// CHECK: %{{.*}} = linalg.elementwise kind=#linalg.elementwise_kind<exp>
// CHECK-SAME         ins(%[[A:.+]] : tensor<8x16x32xf32>) outs(%[[B:.+]] : tensor<8x16x32xf32>)
//
func.func @unary_identity_exp(%A : tensor<8x16x32xf32>, %B: tensor<8x16x32xf32>) ->  tensor<8x16x32xf32> {
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
// CHECK: @unary_projection_tanh(%[[A:.+]]: tensor<?x16xf32>,
// CHECK-SAME:                            %[[B:.+]]: tensor<8x16x?xf32>) ->  tensor<8x16x?xf32> {
// CHECK: {{.*}} = linalg.elementwise kind=#linalg.elementwise_kind<tanh>
// CHECK-SAME:       indexing_maps = [#[[PROJECTION]], #[[IDENTITY]]]
// CHECK-SAME:       ins(%[[A]] : tensor<?x16xf32>) outs(%[[B]] : tensor<8x16x?xf32>) -> tensor<8x16x?xf32>
//
func.func @unary_projection_tanh(%A: tensor<?x16xf32>,
                                          %B: tensor<8x16x?xf32>) ->  tensor<8x16x?xf32> {
  %r = linalg.elementwise
      kind=#linalg.elementwise_kind<tanh>
      indexing_maps = [affine_map<(d0, d1, d2) -> (d2, d1)>,
                       affine_map<(d0, d1, d2) -> (d0, d1, d2)>]
      ins(%A : tensor<?x16xf32>)
      outs(%B: tensor<8x16x?xf32>) -> tensor<8x16x?xf32>
  return %r : tensor<8x16x?xf32>
}

// -----

// CHECK: @binary_identity_div(%[[A:.+]]: tensor<16x8xf32>, %[[B:.+]]: tensor<16x8xf32>,
// CHECK-SAME:        %[[C:.+]]: tensor<16x8xf32>) ->  tensor<16x8xf32> {
// CHECK: {{.*}} = linalg.elementwise
// CHECK-SAME:       kind=#linalg.elementwise_kind<div>
// CHECK-SAME:       ins(%[[A]], %[[B]] : tensor<16x8xf32>, tensor<16x8xf32>)
// CHECK-SAME:       outs(%[[C]] : tensor<16x8xf32>) -> tensor<16x8xf32>
//
func.func @binary_identity_div(%A: tensor<16x8xf32>, %B: tensor<16x8xf32>,
                      %C: tensor<16x8xf32>) ->  tensor<16x8xf32> {
  %r = linalg.elementwise
      kind=#linalg.elementwise_kind<div>
      ins(%A, %B: tensor<16x8xf32>, tensor<16x8xf32>)
      outs(%C: tensor<16x8xf32>) -> tensor<16x8xf32>
  return %r : tensor<16x8xf32>
}

// -----

// CHECK: @binary_identity_mul_5Di(%[[A]]: tensor<1x2x3x4x5xi32>,
// CHECK-SAME:                     %[[B:.+]]: tensor<1x2x3x4x5xi32>,
// CHECK-SAME:                     %[[C:.+]]: tensor<1x2x3x4x5xi32>) -> tensor<1x2x3x4x5xi32> {
// CHECK: {{.*}} = linalg.elementwise
// CHECK-SAME:       kind=#linalg.elementwise_kind<mul>
// CHECK-SAME:       ins(%[[A]], %[[B]] : tensor<1x2x3x4x5xi32>, tensor<1x2x3x4x5xi32>)
// CHECK-SAME:       outs(%[[C]] : tensor<1x2x3x4x5xi32>) -> tensor<1x2x3x4x5xi32>
//
func.func @binary_identity_mul_5Di(%A: tensor<1x2x3x4x5xi32>, %B: tensor<1x2x3x4x5xi32>,
                                   %C: tensor<1x2x3x4x5xi32>) ->  tensor<1x2x3x4x5xi32> {
  %r = linalg.elementwise
      kind=#linalg.elementwise_kind<mul>
      ins(%A, %B: tensor<1x2x3x4x5xi32>, tensor<1x2x3x4x5xi32>)
      outs(%C: tensor<1x2x3x4x5xi32>) -> tensor<1x2x3x4x5xi32>
  return %r : tensor<1x2x3x4x5xi32>
}

// -----

// CHECK: @redundant_maps
// CHECK-NOT: indexing_maps
//
#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
func.func @redundant_maps(%A: tensor<1x2x3x4x5xi32>, %B: tensor<1x2x3x4x5xi32>,
                          %C: tensor<1x2x3x4x5xi32>) ->  tensor<1x2x3x4x5xi32> {
  %r = linalg.elementwise
      kind=#linalg.elementwise_kind<mul>
      indexing_maps = [#map, #map, #map]
      ins(%A, %B: tensor<1x2x3x4x5xi32>, tensor<1x2x3x4x5xi32>)
      outs(%C: tensor<1x2x3x4x5xi32>) -> tensor<1x2x3x4x5xi32>
  return %r : tensor<1x2x3x4x5xi32>
}
