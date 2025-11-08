// RUN: mlir-opt %s -split-input-file --linalg-specialize-generic-ops | FileCheck %s

#map = affine_map<(d0, d1, d2) -> (d1, d0)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// This test checks that linalg.generic does not get incorrectly specialized to transform or broadcast.
// CHECK-LABEL: @transpose_and_broadcast
// CHECK: linalg.generic
func.func @transpose_and_broadcast(%arg0: tensor<7x8xf32>, %arg1: tensor<8x7x9xf32>) -> tensor<8x7x9xf32> {
  %res = linalg.generic {
    indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%arg0 : tensor<7x8xf32>) outs(%arg1 : tensor<8x7x9xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<8x7x9xf32>
  return %res : tensor<8x7x9xf32>
}

// -----

#map = affine_map<(d0) -> (d0)>
// CHECK-LABEL: @neither_permutation_nor_broadcast
// CHECK: linalg.generic
func.func @neither_permutation_nor_broadcast(%init : tensor<8xi32>) -> tensor<8xi32> {
  %res = linalg.generic {
    indexing_maps = [#map], iterator_types = ["parallel"]
  } outs(%init: tensor<8xi32>) {
  ^bb0(%out: i32):
    linalg.yield %out: i32
  } -> tensor<8xi32>
  return %res : tensor<8xi32>
}

// -----

#map = affine_map<(d0) -> (d0)>
// CHECK-LABEL: func @not_copy
//  CHECK-NOT:    linalg.copy
//      CHECK:    linalg.generic
func.func @not_copy(%input: tensor<8xi32>, %init: tensor<8xi32>) -> tensor<8xi32> {
  %c0_i32 = arith.constant 0 : i32
  %res = linalg.generic {
    indexing_maps = [#map, #map], iterator_types = ["parallel"]
  } ins(%input: tensor<8xi32>) outs(%init: tensor<8xi32>) {
  ^bb0(%in: i32, %out: i32):
    linalg.yield %c0_i32 : i32
  } -> tensor<8xi32>
  return %res : tensor<8xi32>
}
