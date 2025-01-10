// RUN: mlir-opt -split-input-file -test-tensor-transform-patterns=test-expand-shape-bubbling %s | FileCheck %s

func.func @bubble_parallel_reshapes(%arg0: tensor<?x?x?x?xf32>, %s0: index, %s1: index, %s2: index, %s3: index) -> tensor<?x?x?x?xf32> {
  %collapse = tensor.collapse_shape %arg0 [[0], [1, 2], [3]] : tensor<?x?x?x?xf32> into tensor<?x?x?xf32>
  %expand = tensor.expand_shape %collapse [[0], [1], [2, 3]]
              output_shape [%s0, %s1, %s2, %s3] : tensor<?x?x?xf32> into tensor<?x?x?x?xf32>
  return %expand : tensor<?x?x?x?xf32>
}
//      CHECK: func @bubble_parallel_reshapes
// CHECK-SAME:   %[[ARG0:.+]]: tensor<?x?x?x?xf32>
// CHECK-SAME:   %[[S0:.+]]: index, %[[S1:.+]]: index, %[[S2:.+]]: index, %[[S3:.+]]: index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//  CHECK-DAG:   %[[DIM1:.+]] = tensor.dim %[[ARG0]], %[[C1]] : tensor<?x?x?x?xf32>
//  CHECK-DAG:   %[[DIM2:.+]] = tensor.dim %[[ARG0]], %[[C2]] : tensor<?x?x?x?xf32>
//      CHECK:   %[[EXPAND:.+]] = tensor.expand_shape %[[ARG0]] {{\[}}[0], [1], [2], [3, 4]]
// CHECK-SAME:       output_shape [%[[S0]], %[[DIM1]], %[[DIM2]], %[[S2]], %[[S3]]] : tensor<?x?x?x?xf32> into tensor<?x?x?x?x?xf32>
//      CHECK:   %[[COLLAPSE:.+]] = tensor.collapse_shape %[[EXPAND]] {{\[}}[0], [1, 2], [3], [4]] : tensor<?x?x?x?x?xf32> into tensor<?x?x?x?xf32>
//      CHECK:   return %[[COLLAPSE]]

// -----

func.func @bubble_parallel_reshapes2(%arg0: tensor<?x2x2x6xf32>, %s0: index, %s1: index) -> tensor<?x4x2x3xf32> {
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %collapse = tensor.collapse_shape %arg0 [[0], [1, 2], [3]] : tensor<?x2x2x6xf32> into tensor<?x4x6xf32>
  %expand = tensor.expand_shape %collapse [[0], [1], [2, 3]]
              output_shape [%s0, %s1, %c2, %c3] : tensor<?x4x6xf32> into tensor<?x4x2x3xf32>
  return %expand : tensor<?x4x2x3xf32>
}
//      CHECK: func @bubble_parallel_reshapes2
// CHECK-SAME:   %[[ARG0:.+]]: tensor<?x2x2x6xf32>
// CHECK-SAME:   %[[S0:.+]]: index, %[[S1:.+]]: index
//  CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//  CHECK-DAG:   %[[C3:.+]] = arith.constant 3 : index
//      CHECK:   %[[EXPAND:.+]] = tensor.expand_shape %[[ARG0]] {{\[}}[0], [1], [2], [3, 4]]
// CHECK-SAME:       output_shape [%[[S0]], 2, 2, %[[C2]], %[[C3]]] : tensor<?x2x2x6xf32> into tensor<?x2x2x2x3xf32>
//      CHECK:   %[[COLLAPSE:.+]] = tensor.collapse_shape %[[EXPAND]] {{\[}}[0], [1, 2], [3], [4]] : tensor<?x2x2x2x3xf32> into tensor<?x4x2x3xf32>
//      CHECK:   return %[[COLLAPSE]]

// -----

func.func @no_bubble_full_intersecting_reshapes(%arg0: tensor<?x?x?x?xf32>, %s0: index, %s1: index, %s2: index, %s3: index) -> tensor<?x?x?x?xf32> {
  %collapse = tensor.collapse_shape %arg0 [[0], [1, 2], [3]] : tensor<?x?x?x?xf32> into tensor<?x?x?xf32>
  %expand = tensor.expand_shape %collapse [[0], [1, 2], [3]]
              output_shape [%s0, %s1, %s2, %s3] : tensor<?x?x?xf32> into tensor<?x?x?x?xf32>
  return %expand : tensor<?x?x?x?xf32>
}
//      CHECK: func @no_bubble_full_intersecting_reshapes
// CHECK-SAME:   %[[ARG0:.+]]: tensor<?x?x?x?xf32>
//      CHECK:   %[[COLLAPSE:.+]] = tensor.collapse_shape %[[ARG0]] {{\[}}[0], [1, 2], [3]]
//      CHECK:   %[[EXPAND:.+]] = tensor.expand_shape %[[COLLAPSE]] {{\[}}[0], [1, 2], [3]]
//      CHECK:   return %[[EXPAND]]

// -----

func.func @no_bubble_partial_intersecting_reshapes(%arg0: tensor<?x?x?x?xf32>, %s0: index, %s1: index, %s2: index, %s3: index) -> tensor<?x?x?x?xf32> {
  %collapse = tensor.collapse_shape %arg0 [[0, 1, 2], [3]] : tensor<?x?x?x?xf32> into tensor<?x?xf32>
  %expand = tensor.expand_shape %collapse [[0, 1], [2, 3]]
              output_shape [%s0, %s1, %s2, %s3] : tensor<?x?xf32> into tensor<?x?x?x?xf32>
  return %expand : tensor<?x?x?x?xf32>
}
//      CHECK: func @no_bubble_partial_intersecting_reshapes
// CHECK-SAME:   %[[ARG0:.+]]: tensor<?x?x?x?xf32>
//      CHECK:   %[[COLLAPSE:.+]] = tensor.collapse_shape %[[ARG0]] {{\[}}[0, 1, 2], [3]]
//      CHECK:   %[[EXPAND:.+]] = tensor.expand_shape %[[COLLAPSE]] {{\[}}[0, 1], [2, 3]]
//      CHECK:   return %[[EXPAND]]
