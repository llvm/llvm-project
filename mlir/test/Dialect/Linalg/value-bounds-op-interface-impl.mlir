// RUN: mlir-opt %s -pass-pipeline='builtin.module(func.func(test-affine-reify-value-bounds))' -verify-diagnostics \
// RUN:     -split-input-file | FileCheck %s

// CHECK-LABEL: func @linalg_fill(
//  CHECK-SAME:     %[[t:.*]]: tensor<?xf32>
//       CHECK:   %[[c0:.*]] = arith.constant 0 : index
//       CHECK:   %[[dim:.*]] = tensor.dim %[[t]], %[[c0]]
//       CHECK:   return %[[dim]]
func.func @linalg_fill(%t: tensor<?xf32>, %f: f32) -> index {
  %0 = linalg.fill ins(%f : f32) outs(%t : tensor<?xf32>) -> tensor<?xf32>
  %1 = "test.reify_bound"(%0) {dim = 0} : (tensor<?xf32>) -> (index)
  return %1 : index
}

// -----

#accesses = [
  affine_map<(i, j, k) -> (j, i)>,
  affine_map<(i, j, k) -> (i, k, i + j)>
]

#trait = {
  indexing_maps = #accesses,
  iterator_types = ["parallel", "parallel", "parallel"]
}

// CHECK-LABEL: func @linalg_index(
//  CHECK-SAME:     %[[arg0:.*]]: memref<?x?xf32>
func.func @linalg_index(%arg0: memref<?x?xf32>,
                        %arg1: memref<?x5x?xf32>) {
  linalg.generic #trait
                 ins(%arg0 : memref<?x?xf32>)
                 outs(%arg1 : memref<?x5x?xf32>)
  {
    ^bb(%a: f32, %b: f32):
      // CHECK: %[[c1:.*]] = arith.constant 1 : index
      // CHECK: %[[ub_0:.*]] = memref.dim %[[arg0]], %[[c1]]
      // CHECK: "test.some_use"(%[[ub_0]])
      %0 = linalg.index 0 : index
      %ub_0 = "test.reify_bound"(%0) {type = "UB"} : (index) -> (index)
      "test.some_use"(%ub_0) : (index) -> ()

      // CHECK: %[[c0:.*]] = arith.constant 0 : index
      // CHECK: "test.some_use"(%[[c0]])
      %lb_0 = "test.reify_bound"(%0) {type = "LB"} : (index) -> (index)
      "test.some_use"(%lb_0) : (index) -> ()

      // CHECK: %[[c0:.*]] = arith.constant 0 : index
      // CHECK: %[[ub_1:.*]] = memref.dim %[[arg0]], %[[c0]]
      // CHECK: "test.some_use"(%[[ub_1]])
      %1 = linalg.index 1 : index
      %ub_1 = "test.reify_bound"(%1) {type = "UB"} : (index) -> (index)
      "test.some_use"(%ub_1) : (index) -> ()

      // CHECK: %[[ub_2:.*]] = arith.constant 5 : index
      // CHECK: "test.some_use"(%[[ub_2]])
      %2 = linalg.index 2 : index
      %ub_2 = "test.reify_bound"(%2) {type = "UB"} : (index) -> (index)
      "test.some_use"(%ub_2) : (index) -> ()

      linalg.yield %b : f32
  }
  return
}

// -----

// CHECK-LABEL: func @linalg_index_scalar_input(
func.func @linalg_index_scalar_input(%scalar: i32, %out: tensor<4xi32>) -> tensor<4xi32> {
  %result = linalg.generic {
      indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%scalar : i32) outs(%out : tensor<4xi32>) {
  ^bb0(%in: i32, %old: i32):
    %i = linalg.index 0 : index
    // CHECK: %[[UB:.*]] = arith.constant 4 : index
    // CHECK: %[[VALUE:.*]] = arith.index_cast %[[UB]] : index to i32
    // CHECK: linalg.yield %[[VALUE]] : i32
    %ub = "test.reify_bound"(%i) {type = "UB"} : (index) -> index
    %value = arith.index_cast %ub : index to i32
    linalg.yield %value : i32
  } -> tensor<4xi32>
  return %result : tensor<4xi32>
}

// -----

// CHECK-LABEL: func @linalg_index_scalar_input_dynamic(
// CHECK-SAME: %[[OUT:[^ ,:]+]]: tensor<?x?xi32>
func.func @linalg_index_scalar_input_dynamic(%input: tensor<?xi32>, %scalar: i32,
                                           %rank_zero: tensor<i32>,
                                           %out: tensor<?x?xi32>) -> tensor<?x?xi32> {
  %result = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0)>,
                       affine_map<(d0, d1) -> ()>,
                       affine_map<(d0, d1) -> ()>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]}
      ins(%input, %scalar, %rank_zero : tensor<?xi32>, i32, tensor<i32>)
      outs(%out : tensor<?x?xi32>) {
  ^bb0(%in: i32, %s: i32, %r: i32, %old: i32):
    %i = linalg.index 1 : index
    // CHECK: %[[C1:.*]] = arith.constant 1 : index
    // CHECK: %[[UB:.*]] = tensor.dim %[[OUT]], %[[C1]]
    // CHECK: %[[VALUE:.*]] = arith.index_cast %[[UB]] : index to i32
    // CHECK: linalg.yield %[[VALUE]] : i32
    %ub = "test.reify_bound"(%i) {type = "UB"} : (index) -> index
    %value = arith.index_cast %ub : index to i32
    linalg.yield %value : i32
  } -> tensor<?x?xi32>
  return %result : tensor<?x?xi32>
}
