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


// -----

// CHECK-LABEL: @scalar_input
// CHECK: linalg.generic
#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
func.func @scalar_input(%arg0: tensor<128xf32>, %arg1: f32) -> tensor<128xf32> {
  %1 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<128xf32>, f32) outs(%arg0 : tensor<128xf32>) {
  ^bb0(%in: f32, %in_1: f32, %out: f32):
    %2 = arith.addf %in_1, %in_1 : f32
    linalg.yield %2 : f32
  } -> tensor<128xf32>
  return %1 : tensor<128xf32>
}

// -----

// linalg.reduce requires the same number of inputs and inits.
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
// CHECK-LABEL: @reduce_mismatched_ins_outs
// CHECK-NOT: linalg.reduce
// CHECK: linalg.generic
func.func @reduce_mismatched_ins_outs(%A: tensor<4x8xf32>, %B: tensor<4x8xf32>,
    %init: tensor<4xf32>) -> tensor<4xf32> {
  %0 = linalg.generic {
    indexing_maps = [#map, #map, #map1], iterator_types = ["parallel", "reduction"]
  } ins(%A, %B : tensor<4x8xf32>, tensor<4x8xf32>) outs(%init : tensor<4xf32>) {
  ^bb0(%a: f32, %b: f32, %acc: f32):
    %1 = arith.addf %a, %b : f32
    %2 = arith.addf %acc, %1 : f32
    linalg.yield %2 : f32
  } -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// linalg.reduce requires the init shape to be the input shape with the reduced
// dimensions dropped, in order. A permuted output map is not a reduce.
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d0)>
// CHECK-LABEL: @reduce_permuted_output
// CHECK-NOT: linalg.reduce
// CHECK: linalg.generic
func.func @reduce_permuted_output(%A: tensor<4x8x16xf32>, %init: tensor<8x4xf32>) -> tensor<8x4xf32> {
  %0 = linalg.generic {
    indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%A : tensor<4x8x16xf32>) outs(%init : tensor<8x4xf32>) {
  ^bb0(%a: f32, %acc: f32):
    %1 = arith.addf %acc, %a : f32
    linalg.yield %1 : f32
  } -> tensor<8x4xf32>
  return %0 : tensor<8x4xf32>
}

// -----

// linalg.generic accepts a dynamic input dim matching a static init dim, but
// linalg.reduce requires the shapes to match exactly.
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
// CHECK-LABEL: @reduce_dynamic_input_static_init
// CHECK-NOT: linalg.reduce
// CHECK: linalg.generic
func.func @reduce_dynamic_input_static_init(%A: tensor<?x8xf32>, %init: tensor<4xf32>) -> tensor<4xf32> {
  %0 = linalg.generic {
    indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]
  } ins(%A : tensor<?x8xf32>) outs(%init : tensor<4xf32>) {
  ^bb0(%a: f32, %acc: f32):
    %1 = arith.addf %acc, %a : f32
    linalg.yield %1 : f32
  } -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
// CHECK-LABEL: @reduce_static_input_dynamic_init
// CHECK-NOT: linalg.reduce
// CHECK: linalg.generic
func.func @reduce_static_input_dynamic_init(%A: tensor<4x8xf32>, %init: tensor<?xf32>) -> tensor<?xf32> {
  %0 = linalg.generic {
    indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]
  } ins(%A : tensor<4x8xf32>) outs(%init : tensor<?xf32>) {
  ^bb0(%a: f32, %acc: f32):
    %1 = arith.addf %acc, %a : f32
    linalg.yield %1 : f32
  } -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

// linalg.reduce requires all inputs to have exactly the same shape.
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
// CHECK-LABEL: @reduce_multi_input_shape_mismatch
// CHECK-NOT: linalg.reduce
// CHECK: linalg.generic
func.func @reduce_multi_input_shape_mismatch(%A: tensor<?x8xf32>, %B: tensor<4x8xf32>,
    %initA: tensor<?xf32>, %initB: tensor<?xf32>) -> (tensor<?xf32>, tensor<?xf32>) {
  %0:2 = linalg.generic {
    indexing_maps = [#map, #map, #map1, #map1], iterator_types = ["parallel", "reduction"]
  } ins(%A, %B : tensor<?x8xf32>, tensor<4x8xf32>) outs(%initA, %initB : tensor<?xf32>, tensor<?xf32>) {
  ^bb0(%a: f32, %b: f32, %accA: f32, %accB: f32):
    %1 = arith.addf %accA, %a : f32
    %2 = arith.addf %accB, %b : f32
    linalg.yield %1, %2 : f32, f32
  } -> (tensor<?xf32>, tensor<?xf32>)
  return %0#0, %0#1 : tensor<?xf32>, tensor<?xf32>
}
