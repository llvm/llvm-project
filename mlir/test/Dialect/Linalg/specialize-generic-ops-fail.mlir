// RUN: mlir-opt %s -split-input-file --linalg-specialize-generic-ops | FileCheck %s

#map = affine_map<(d0, d1, d2) -> (d1, d0)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// This test checks that linalg.generic does not get incorrectly specialized to transform or broadcast.
// CHECK-LABEL: @transpose_and_broadcast
// CHECK: linalg.generic
func.func @transpose_and_broadcast(%arg0: tensor<7x8xf32>, %arg1: tensor<8x7x9xf32>) -> tensor<8x7x9xf32> {
  %0 = linalg.generic
        {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%arg0 : tensor<7x8xf32>) outs(%arg1 : tensor<8x7x9xf32>) {
        ^bb0(%in: f32, %out: f32):
           linalg.yield %in : f32
  } -> tensor<8x7x9xf32>
  return %0 : tensor<8x7x9xf32>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
// Verifies that the pass crashes when trying to specialize a linalg.generic op with 
// a reduction iterator when there is no valid reduction operation in the contraction body.
// CHECK-LABEL: @specialize_reduction
func.func private @specialize_reduction(%arg0: tensor<1x31x8xi32>) -> tensor<31x31xi32> {
  %c-2351_i32 = arith.constant -2351 : i32
  %c0_i32 = arith.constant 0 : i32
  %0 = tensor.empty() : tensor<31x8xi32>
  %1 = linalg.generic 
        {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]}
        outs(%0 : tensor<31x8xi32>) {
    ^bb0(%out: i32):
      linalg.yield %c-2351_i32 : i32
  } -> tensor<31x8xi32>
  %2 = tensor.empty() : tensor<31x31xi32>
  %3 = linalg.generic 
        {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]}
        outs(%2 : tensor<31x31xi32>) {
    ^bb0(%out: i32):
      linalg.yield %c0_i32 : i32
  } -> tensor<31x31xi32>
  %4 = linalg.generic 
        {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"]}
        ins(%1, %1 : tensor<31x8xi32>, tensor<31x8xi32>) 
        outs(%3 : tensor<31x31xi32>) {
    ^bb0(%in: i32, %in_0: i32, %out: i32):
      linalg.yield %out : i32
  } -> tensor<31x31xi32>
  return %4 : tensor<31x31xi32>
}