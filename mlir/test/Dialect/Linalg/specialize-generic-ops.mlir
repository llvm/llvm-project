// RUN: mlir-opt %s -split-input-file --linalg-specialize-generic-ops | FileCheck %s

#umap = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func.func @unary_op_exp(%A: tensor<?x?x?xf32>, %Out: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %0 = linalg.generic
          {indexing_maps = [#umap, #umap], iterator_types = ["parallel", "parallel","parallel"]}
          ins(%A : tensor<?x?x?xf32>) outs(%Out : tensor<?x?x?xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1 = math.exp %in : f32
    linalg.yield %1 : f32
  } -> tensor<?x?x?xf32>
  return %0 : tensor<?x?x?xf32>
}

// CHECK-LABEL: unary_op_exp
// CHECK-SAME: %[[A:.+]]: tensor<?x?x?xf32>, %[[Out:.+]]: tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
// CHECK-NOT: linalg.generic
// CHECK: linalg.exp ins(%[[A]] : tensor<?x?x?xf32>) outs(%[[Out]] : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @binary_op_div(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %Out: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.generic
         {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]}
         ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>) outs(%Out : tensor<?x?xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %1 = arith.divf %in, %in_0 : f32
    linalg.yield %1 : f32
  } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: binary_op_div
// CHECK-SAME: %[[A:.+]]: tensor<?x?xf32>, %[[B:.+]]: tensor<?x?xf32>,  %[[Out:.+]]: tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK-NOT: linalg.generic
// CHECK: linalg.div ins(%[[A]], %[[B]] : tensor<?x?xf32>, tensor<?x?xf32>) outs(%[[Out]] : tensor<?x?xf32>) -> tensor<?x?xf32>
