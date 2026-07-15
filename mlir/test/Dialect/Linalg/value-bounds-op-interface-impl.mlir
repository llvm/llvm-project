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
