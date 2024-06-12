// RUN: mlir-opt %s -split-input-file --linalg-specialize-generic-ops | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @specialize_div(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.generic 
         {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]}
         ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg2 : tensor<?x?xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %1 = arith.divf %in, %in_0 : f32
    linalg.yield %1 : f32
  } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: specialize_div
// CHECK-SAME: %[[ARG0:.+]]: tensor<?x?xf32>, %[[ARG1:.+]]: tensor<?x?xf32>,  %[[ARG2:.+]]: tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK-NOT: linalg.generic
// CHECK: linalg.div ins(%[[ARG0]], %[[ARG1]] : tensor<?x?xf32>, tensor<?x?xf32>) outs(%[[ARG2]] : tensor<?x?xf32>) -> tensor<?x?xf32>

// -----

#umap = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func.func @specialize_exp(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %0 = linalg.generic
          {indexing_maps = [#umap, #umap], iterator_types = ["parallel", "parallel","parallel"]}
          ins(%arg0 : tensor<?x?x?xf32>) outs(%arg1 : tensor<?x?x?xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1 = math.exp %in : f32
    linalg.yield %1 : f32
  } -> tensor<?x?x?xf32>
  return %0 : tensor<?x?x?xf32>
}

// CHECK-LABEL: specialize_exp
// CHECK-SAME: %[[ARG0:.+]]: tensor<?x?x?xf32>, %[[ARG1:.+]]: tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
// CHECK-NOT: linalg.generic
// CHECK: linalg.exp ins(%[[ARG0]] : tensor<?x?x?xf32>) outs(%[[ARG1]] : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
