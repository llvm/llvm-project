// RUN: mlir-opt %s -split-input-file -linalg-morph-ops=generic-to-named \
// RUN: | FileCheck %s --check-prefixes=CHECK,NAMED
// RUN: mlir-opt %s -split-input-file -linalg-morph-ops=generic-to-category \
// RUN: | FileCheck %s --check-prefixes=CHECK,CATEGORY

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

// -----

#map3 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map4 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map5 = affine_map<(d0, d1, d2) -> (d0, d1)>
// This test checks that linalg.generic with a negf between mulf and addf
// does not get incorrectly specialized to matmul.
// CHECK-LABEL: @contraction_with_negf
// NAMED-NOT:    linalg.matmul
// CATEGORY-NOT: linalg.contract
// CHECK:        linalg.generic
func.func @contraction_with_negf(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>, %arg2: tensor<3x3xf32>) -> tensor<3x3xf32> {
  %0 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<3x3xf32>, tensor<3x3xf32>) outs(%arg2 : tensor<3x3xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %1 = arith.mulf %in, %in_0 : f32
    %2 = arith.negf %1 : f32
    %3 = arith.addf %out, %2 : f32
    linalg.yield %3 : f32
  } -> tensor<3x3xf32>
  return %0 : tensor<3x3xf32>
}

// -----

#map3 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map4 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map5 = affine_map<(d0, d1, d2) -> (d0, d1)>
// This test checks that a cast chain changing the input semantics does not get
// ignored when matching contractions.
// CHECK-LABEL: @contraction_with_rounding_cast_chain
// NAMED-NOT:    linalg.matmul
// CATEGORY-NOT: linalg.contract
// CHECK:        linalg.generic
func.func @contraction_with_rounding_cast_chain(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>, %arg2: tensor<3x3xf32>) -> tensor<3x3xf32> {
  %0 = linalg.generic {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<3x3xf32>, tensor<3x3xf32>) outs(%arg2 : tensor<3x3xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %1 = arith.fptosi %in : f32 to i32
    %2 = arith.sitofp %1 : i32 to f32
    %3 = arith.fptosi %in_0 : f32 to i32
    %4 = arith.sitofp %3 : i32 to f32
    %5 = arith.mulf %2, %4 : f32
    %6 = arith.addf %out, %5 : f32
    linalg.yield %6 : f32
  } -> tensor<3x3xf32>
  return %0 : tensor<3x3xf32>
}
