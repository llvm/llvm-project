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

// -----

func.func @no_bubble_0d_tensor_reshapes(%arg0: tensor<1x1xf32>) -> tensor<1x1x1xf32> {
  %collapse = tensor.collapse_shape %arg0 [] : tensor<1x1xf32> into tensor<f32>
  %expand = tensor.expand_shape %collapse []
              output_shape [1, 1, 1] : tensor<f32> into tensor<1x1x1xf32>
  return %expand : tensor<1x1x1xf32>
}
//      CHECK: func @no_bubble_0d_tensor_reshapes
// CHECK-SAME:   %[[ARG0:.+]]: tensor<1x1xf32>
//      CHECK:   %[[COLLAPSE:.+]] = tensor.collapse_shape %[[ARG0]] {{\[}}]
//      CHECK:   %[[EXPAND:.+]] = tensor.expand_shape %[[COLLAPSE]] {{\[}}]
//      CHECK:   return %[[EXPAND]]

// -----

// Test the case where the reassocation indices in the collapse and expand
// are of same size.
func.func @bubble_expand_match_non_unit_size_reassocation(
      %arg0 : tensor<4x?x4x32x4x?xf16>, %arg1 : index, %arg2 : index) -> tensor<4x?x4x128x?x32xf16> {
  %collapsed = tensor.collapse_shape %arg0 [[0, 1, 2], [3, 4], [5]]
      : tensor<4x?x4x32x4x?xf16> into tensor<?x128x?xf16>
  %expanded = tensor.expand_shape %collapsed [[0, 1, 2], [3], [4, 5]] output_shape [4, %arg1, 4, 128, %arg2, 32]
      : tensor<?x128x?xf16> into tensor<4x?x4x128x?x32xf16>
  return %expanded : tensor<4x?x4x128x?x32xf16>
}
//      CHECK: func @bubble_expand_match_non_unit_size_reassocation
// CHECK-SAME:     %[[ARG0:.+]]: tensor<4x?x4x32x4x?xf16>
// CHECK-SAME:     %[[ARG1:[a-zA-z0-9]+]]: index
// CHECK-SAME:     %[[ARG2:[a-zA-z0-9]+]]: index
//      CHECK:   %[[EXPANDED:.+]] = tensor.expand_shape %[[ARG0]]
// CHECK-SAME:       {{\[}}[0], [1], [2], [3], [4], [5, 6]{{\]}}
// CHECK-SAME:       [4, %[[ARG1]], 4, 32, 4, %[[ARG2]], 32]
//      CHECK:   %[[COLLAPSED:.+]] = tensor.collapse_shape %[[EXPANDED]]
// CHECK-SAME:       {{\[}}[0], [1], [2], [3, 4], [5], [6]{{\]}}
//      CHECK:   return %[[COLLAPSED]]

// -----

// Test the case where the trailing collapse isnt needed.
func.func @no_collapse_generated(
      %arg0 : tensor<4x?x4x128x?xf16>, %arg1 : index, %arg2 : index) -> tensor<4x?x4x128x?x32xf16> {
  %collapsed = tensor.collapse_shape %arg0 [[0, 1, 2], [3], [4]]
      : tensor<4x?x4x128x?xf16> into tensor<?x128x?xf16>
  %expanded = tensor.expand_shape %collapsed [[0, 1, 2], [3], [4, 5]] output_shape [4, %arg1, 4, 128, %arg2, 32]
      : tensor<?x128x?xf16> into tensor<4x?x4x128x?x32xf16>
  return %expanded : tensor<4x?x4x128x?x32xf16>
}
//      CHECK: func @no_collapse_generated
//      CHECK:   %[[EXPANDED:.+]] = tensor.expand_shape 
//      CHECK:   return %[[EXPANDED]]

// -----

// Test the case where the leading expand isnt needed.
func.func @no_expand_generated(
      %arg0 : tensor<4x?x4x128x?x?x?xf16>, %arg1 : index, %arg2 : index, %arg3 : index) -> tensor<4x?x4x128x?x?xf16> {
  %collapsed = tensor.collapse_shape %arg0 [[0, 1, 2], [3], [4], [5, 6]]
      : tensor<4x?x4x128x?x?x?xf16> into tensor<?x128x?x?xf16>
  %expanded = tensor.expand_shape %collapsed [[0, 1, 2], [3], [4], [5]] output_shape [4, %arg1, 4, 128, %arg2, %arg3]
      : tensor<?x128x?x?xf16> into tensor<4x?x4x128x?x?xf16>
  return %expanded : tensor<4x?x4x128x?x?xf16>
}
//      CHECK: func @no_expand_generated
//      CHECK:   %[[EXPANDED:.+]] = tensor.collapse_shape
//      CHECK:   return %[[EXPANDED]]
