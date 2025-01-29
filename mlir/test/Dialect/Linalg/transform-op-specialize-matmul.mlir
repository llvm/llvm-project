// RUN: mlir-opt --transform-interpreter --split-input-file --verify-diagnostics %s | FileCheck %s

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @specialize_matmul(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.generic
          {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]}
          ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg2 : tensor<?x?xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %0 = arith.mulf %in, %in_0 : f32
      %1 = arith.addf %out, %0 : f32
      linalg.yield %1 : f32
    } -> tensor<?x?xf32>
 return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: @specialize_matmul
// CHECK-SAME: %[[ARG0:.+]]: tensor<?x?xf32>, %[[ARG1:.+]]: tensor<?x?xf32>, %[[ARG2:.+]]: tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK-NOT: linalg.generic
// CHECK: linalg.matmul ins(%[[ARG0]], %[[ARG1]] : tensor<?x?xf32>, tensor<?x?xf32>) outs(%[[ARG2]] : tensor<?x?xf32>) -> tensor<?x?xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match interface{LinalgOp} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.specialize %0 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

#map = affine_map<(d0, d1, d2) -> (d2, d0)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @matmul_transpose_a(%arg0: memref<5x3xf32>, %arg1: memref<5x7xf32>, %arg2: memref<3x7xf32>) {
  linalg.generic
     {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]}
     ins(%arg0, %arg1 : memref<5x3xf32>, memref<5x7xf32>) outs(%arg2 : memref<3x7xf32>) {
      ^bb0(%in: f32, %in_0: f32, %out: f32):
       %0 = arith.mulf %in, %in_0 : f32
       %1 = arith.addf %out, %0 : f32
       linalg.yield %1 : f32
  }
  return
}

// CHECK-LABEL: @matmul_transpose_a
// CHECK-SAME: %[[ARG0:.+]]: memref<5x3xf32>, %[[ARG1:.+]]: memref<5x7xf32>, %[[ARG2:.+]]: memref<3x7xf32>) {
// CHECK-NOT: linalg.generic
// CHECK: linalg.matmul_transpose_a ins(%[[ARG0]], %[[ARG1]] : memref<5x3xf32>, memref<5x7xf32>) outs(%[[ARG2]] : memref<3x7xf32>)

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match interface{LinalgOp} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.specialize %0 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func.func @matmul_transpose_b(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.generic
          {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]}
          ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg2 : tensor<?x?xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %1 = arith.mulf %in, %in_0 : f32
    %2 = arith.addf %out, %1 : f32
    linalg.yield %2 : f32
  } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: @matmul_transpose_b
// CHECK-SAME: %[[ARG0:.+]]: tensor<?x?xf32>, %[[ARG1:.+]]: tensor<?x?xf32>, %[[ARG2:.+]]: tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK-NOT: linalg.generic
// CHECK: linalg.matmul_transpose_b ins(%[[ARG0]], %[[ARG1]] : tensor<?x?xf32>, tensor<?x?xf32>) outs(%[[ARG2]] : tensor<?x?xf32>) -> tensor<?x?xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match interface{LinalgOp} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.specialize %0 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
func.func @batch_matmul(%arg0: tensor<2x16x8xf32>, %arg1: tensor<2x8x16xf32>, %arg2: tensor<2x16x16xf32>) -> tensor<2x16x16xf32> {
  %0 = linalg.generic
           {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
           ins(%arg0, %arg1 : tensor<2x16x8xf32>, tensor<2x8x16xf32>) outs(%arg2 : tensor<2x16x16xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %1 = arith.mulf %in, %in_0 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
  } -> tensor<2x16x16xf32>
  return %0 : tensor<2x16x16xf32>
}

// CHECK-LABEL: @batch_matmul
// CHECK-SAME: %[[ARG0:.+]]: tensor<2x16x8xf32>, %[[ARG1:.+]]: tensor<2x8x16xf32>, %[[ARG2:.+]]: tensor<2x16x16xf32>) -> tensor<2x16x16xf32>
// CHECK-NOT: linalg.generic
// CHECK: linalg.batch_matmul ins(%[[ARG0]], %[[ARG1]] : tensor<2x16x8xf32>, tensor<2x8x16xf32>) outs(%[[ARG2]] : tensor<2x16x16xf32>) -> tensor<2x16x16xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match interface{LinalgOp} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.specialize %0 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
func.func @batch_matmul_transpose_b(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>, %arg2: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %0 = linalg.generic
       {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
       ins(%arg0, %arg1 : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%arg2 : tensor<?x?x?xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
      %1 = arith.mulf %in, %in_0 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
  } -> tensor<?x?x?xf32>
  return %0 : tensor<?x?x?xf32>
}

// CHECK-LABEL: @batch_matmul_transpose_b
// CHECK-SAME: %[[ARG0:.+]]: tensor<?x?x?xf32>, %[[ARG1:.+]]: tensor<?x?x?xf32>, %[[ARG2:.+]]: tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
// CHECK-NOT: linalg.generic
// CHECK: linalg.batch_matmul_transpose_b ins(%[[ARG0]], %[[ARG1]] : tensor<?x?x?xf32>, tensor<?x?x?xf32>) outs(%[[ARG2]] : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match interface{LinalgOp} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.specialize %0 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
