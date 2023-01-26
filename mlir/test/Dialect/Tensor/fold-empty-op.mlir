// RUN: mlir-opt -split-input-file -test-tensor-transform-patterns=test-empty-op-folding  %s | FileCheck %s

func.func @empty_reshape_expansion(%arg0 : index) -> tensor<2x3x5x4x?x7xf32> {
  %0 = tensor.empty(%arg0) : tensor<6x5x?xf32>
  %1 = tensor.expand_shape %0 [[0, 1], [2], [3, 4, 5]]
      : tensor<6x5x?xf32> into tensor<2x3x5x4x?x7xf32>
  return %1 : tensor<2x3x5x4x?x7xf32>
}
//      CHECK: #[[MAP:.+]] = affine_map<()[s0] -> (s0 floordiv 28)>
//      CHECK: func @empty_reshape_expansion
// CHECK-SAME:     %[[ARG0:.+]]: index
// CHECK:        %[[OLD_INIT:.+]] = tensor.empty(%{{.*}}) : tensor<6x5x?xf32>
// CHECK-NEXT:   %[[DIM:.*]] = tensor.dim %[[OLD_INIT]]
// CHECK-NEXT:   %[[D:.+]] = affine.apply #[[MAP]]()[%[[DIM]]]
// CHECK-NEXT:   %[[INIT:.+]] = tensor.empty(%[[D]])
// CHECK-NEXT:   return %[[INIT]]

// -----

func.func @empty_reshape_collapse(%arg0 : index) -> tensor<6x5x?xf32> {
  %0 = tensor.empty(%arg0) : tensor<2x3x5x4x?x7xf32>
  %1 = tensor.collapse_shape %0 [[0, 1], [2], [3, 4, 5]]
      : tensor<2x3x5x4x?x7xf32> into tensor<6x5x?xf32>
  return %1 : tensor<6x5x?xf32>
}
//      CHECK: #[[MAP:.+]] = affine_map<()[s0] -> (s0 * 28)>
//      CHECK: func @empty_reshape_collapse
// CHECK-SAME:     %[[ARG0:.+]]: index
// CHECK:        %[[OLD_INIT:.+]] = tensor.empty(%{{.*}}) : tensor<2x3x5x4x?x7xf32>
// CHECK-NEXT:   %[[DIM:.*]] = tensor.dim %[[OLD_INIT]]
// CHECK-NEXT:   %[[D:.+]] = affine.apply #[[MAP]]()[%[[DIM]]]
// CHECK-NEXT:   %[[INIT:.+]] = tensor.empty(%[[D]])
// CHECK-NEXT:   return %[[INIT]]

// -----

func.func @fold_empty_tensor_with_slice
  (%arg0 : index, %arg1 : index) -> tensor<5x?x20xf32>
{
  %0 = tensor.empty(%arg0) : tensor<?x10x40xf32>
  %1 = tensor.extract_slice %0[0, 0, 0] [5, %arg1, 20] [1, 1, 1]
    : tensor<?x10x40xf32> to tensor<5x?x20xf32>
  return %1 : tensor<5x?x20xf32>
}
//      CHECK: func @fold_empty_tensor_with_slice
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: index
//      CHECK:   %[[T0:.+]] = tensor.empty(%[[ARG1]])
//      CHECK:   return %[[T0]]

// -----

// CHECK-LABEL: func @rank_reducing_empty_tensor_extract
func.func @rank_reducing_empty_tensor_extract(%sz : index, %idx : index) -> tensor<2xf32> {
  // CHECK: tensor.empty() : tensor<2xf32>
  %a = tensor.empty(%sz) : tensor<?x2xf32>

  // CHECK-NOT: extract
  %r = tensor.extract_slice %a[%idx, 0] [1, 2] [1, 1] : tensor<?x2xf32> to tensor<2xf32>
  return %r: tensor<2xf32>
}
