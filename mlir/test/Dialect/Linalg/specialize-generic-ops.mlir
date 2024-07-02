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

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @op_matmul(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %Out: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.generic
         {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]}
         ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>) outs(%Out : tensor<?x?xf32>) {
   ^bb0(%in: f32, %in_0: f32, %out: f32):
     %1 = arith.mulf %in, %in_0 : f32
     %2 = arith.addf %out, %1 : f32
     linalg.yield %2 : f32
   } -> tensor<?x?xf32>
   return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: op_matmul
// CHECK-SAME: %[[A:.+]]: tensor<?x?xf32>, %[[B:.+]]: tensor<?x?xf32>,  %[[Out:.+]]: tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK-NOT: linalg.generic
// CHECK: linalg.matmul ins(%[[A]], %[[B]] : tensor<?x?xf32>, tensor<?x?xf32>) outs(%[[Out]] : tensor<?x?xf32>) -> tensor<?x?xf32>

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
func.func @op_batch_matmul(%A: tensor<2x16x8xf32>, %B: tensor<2x8x16xf32>, %Out: tensor<2x16x16xf32>) -> tensor<2x16x16xf32> {
  %0 = linalg.generic
           {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
           ins(%A, %B : tensor<2x16x8xf32>, tensor<2x8x16xf32>) outs(%Out : tensor<2x16x16xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %1 = arith.mulf %in, %in_0 : f32
    %2 = arith.addf %out, %1 : f32
    linalg.yield %2 : f32
  } -> tensor<2x16x16xf32>
  return %0 : tensor<2x16x16xf32>
}

// CHECK-LABEL: op_batch_matmul
// CHECK-SAME: %[[A:.+]]: tensor<2x16x8xf32>, %[[B:.+]]: tensor<2x8x16xf32>,  %[[Out:.+]]: tensor<2x16x16xf32>) -> tensor<2x16x16xf32>
// CHECK-NOT: linalg.generic
// CHECK: linalg.batch_matmul ins(%[[A]], %[[B]] : tensor<2x16x8xf32>, tensor<2x8x16xf32>) outs(%[[Out]] : tensor<2x16x16xf32>) -> tensor<2x16x16xf32>

// -----

// This is a multi-reduction linalg.generic and cannot be lifted to matrix multiply
#mapA = affine_map<(m, n, k1, k2) -> (m, k1, k2)>
#mapB = affine_map<(m, n, k1, k2) -> (k2, k1, n)>
#mapC = affine_map<(m, n, k1, k2) -> (m, n)>
func.func @negative_op_multi_reduction(%A: tensor<10x20x30xf32>,
                                       %B: tensor<30x20x40xf32>,
                                       %C: tensor<10x40xf32>) -> tensor<10x40xf32> {
  %0 = linalg.generic
           {indexing_maps = [#mapA, #mapB, #mapC],
            iterator_types = ["parallel", "parallel", "reduction", "reduction"]}
  ins(%A, %B : tensor<10x20x30xf32>, tensor<30x20x40xf32>)
  outs(%C : tensor<10x40xf32>) {
  ^bb0(%a: f32, %b: f32, %c: f32):
    %1 = arith.mulf %a, %b : f32
    %2 = arith.addf %c, %1 : f32
    linalg.yield %2 : f32
  } -> tensor<10x40xf32>
  return %0 : tensor<10x40xf32>
}

// CHECK-LABEL: negative_op_multi_reduction
// CHECK: linalg.generic

// -----

// TODO: matvec
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
#map2 = affine_map<(d0, d1) -> (d0)>
func.func @op_matvec(%A: tensor<?x?xf32>, %B: tensor<?xf32>, %Out: tensor<?xf32>) -> tensor<?xf32> {
  %0 = linalg.generic
          {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "reduction"]}
          ins(%A, %B : tensor<?x?xf32>, tensor<?xf32>) outs(%Out : tensor<?xf32>) {
      ^bb0(%in: f32, %in_0: f32, %out: f32):
        %1 = arith.mulf %in, %in_0 : f32
        %2 = arith.addf %out, %1 : f32
        linalg.yield %2 : f32
  } -> tensor<?xf32>
  return %0 : tensor<?xf32>
}
// CHECK-LABEL: op_matvec
// CHECK: linalg.generic
