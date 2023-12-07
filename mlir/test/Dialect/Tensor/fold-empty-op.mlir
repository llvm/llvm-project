// RUN: mlir-opt -split-input-file -test-transform-dialect-interpreter %s | FileCheck %s

transform.sequence failures(propagate) {
^bb1(%func_op: !transform.op<"func.func">):
  transform.apply_patterns to %func_op {
    transform.apply_patterns.tensor.fold_tensor_empty
  } : !transform.op<"func.func">
}

// CHECK: #[[$MAP:.+]] = affine_map<()[s0] -> (s0 floordiv 28)>
// CHECK: #[[$MAP2:.+]] = affine_map<()[s0] -> (s0 * 28)>

func.func @empty_reshape_expansion(%arg0 : index) -> tensor<2x3x5x4x?x7xf32> {
  %0 = tensor.empty(%arg0) : tensor<6x5x?xf32>
  %1 = tensor.expand_shape %0 [[0, 1], [2], [3, 4, 5]]
      : tensor<6x5x?xf32> into tensor<2x3x5x4x?x7xf32>
  return %1 : tensor<2x3x5x4x?x7xf32>
}
// CHECK-LABEL: func @empty_reshape_expansion
// CHECK-SAME:     %[[ARG0:.+]]: index
// CHECK:        %[[OLD_INIT:.+]] = tensor.empty(%{{.*}}) : tensor<6x5x?xf32>
// CHECK-NEXT:   %[[DIM:.*]] = tensor.dim %[[OLD_INIT]]
// CHECK-NEXT:   %[[D:.+]] = affine.apply #[[$MAP]]()[%[[DIM]]]
// CHECK-NEXT:   %[[INIT:.+]] = tensor.empty(%[[D]])
// CHECK-NEXT:   return %[[INIT]]

func.func @empty_reshape_collapse(%arg0 : index) -> tensor<6x5x?xf32> {
  %0 = tensor.empty(%arg0) : tensor<2x3x5x4x?x7xf32>
  %1 = tensor.collapse_shape %0 [[0, 1], [2], [3, 4, 5]]
      : tensor<2x3x5x4x?x7xf32> into tensor<6x5x?xf32>
  return %1 : tensor<6x5x?xf32>
}
// CHECK-LABEL: func @empty_reshape_collapse
// CHECK-SAME:     %[[ARG0:.+]]: index
// CHECK:        %[[OLD_INIT:.+]] = tensor.empty(%{{.*}}) : tensor<2x3x5x4x?x7xf32>
// CHECK-NEXT:   %[[DIM:.*]] = tensor.dim %[[OLD_INIT]]
// CHECK-NEXT:   %[[D:.+]] = affine.apply #[[$MAP2]]()[%[[DIM]]]
// CHECK-NEXT:   %[[INIT:.+]] = tensor.empty(%[[D]])
// CHECK-NEXT:   return %[[INIT]]

func.func @fold_empty_tensor_with_slice
  (%arg0 : index, %arg1 : index) -> tensor<5x?x20xf32>
{
  %0 = tensor.empty(%arg0) : tensor<?x10x40xf32>
  %1 = tensor.extract_slice %0[0, 0, 0] [5, %arg1, 20] [1, 1, 1]
    : tensor<?x10x40xf32> to tensor<5x?x20xf32>
  return %1 : tensor<5x?x20xf32>
}
// CHECK-LABEL: func @fold_empty_tensor_with_slice
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: index
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]+]]: index
//      CHECK:   %[[T0:.+]] = tensor.empty(%[[ARG1]])
//      CHECK:   return %[[T0]]

// CHECK-LABEL: func @rank_reducing_empty_tensor_extract
func.func @rank_reducing_empty_tensor_extract(%sz : index, %idx : index) -> tensor<2xf32> {
  // CHECK: tensor.empty() : tensor<2xf32>
  %a = tensor.empty(%sz) : tensor<?x2xf32>

  // CHECK-NOT: extract
  %r = tensor.extract_slice %a[%idx, 0] [1, 2] [1, 1] : tensor<?x2xf32> to tensor<2xf32>
  return %r: tensor<2xf32>
}

// -----

transform.sequence failures(propagate) {
^bb1(%func_op: !transform.op<"func.func">):
  transform.apply_patterns to %func_op {
    transform.apply_patterns.tensor.fold_tensor_empty
        {fold_single_use_only = true}
  } : !transform.op<"func.func">
}

func.func @double_use_of_tensor_empty(%arg0: index, %arg1: index)
    -> (tensor<5x?x20xf32>, tensor<5x?x20xf32>)
{
  %0 = tensor.empty(%arg0) : tensor<?x10x40xf32>
  %1 = tensor.extract_slice %0[0, 0, 0] [5, %arg1, 20] [1, 1, 1]
    : tensor<?x10x40xf32> to tensor<5x?x20xf32>
  %2 = tensor.extract_slice %0[1, 1, 1] [5, %arg1, 20] [1, 1, 1]
    : tensor<?x10x40xf32> to tensor<5x?x20xf32>
  return %1, %2 : tensor<5x?x20xf32>, tensor<5x?x20xf32>
}
// CHECK-LABEL: func @double_use_of_tensor_empty(
//       CHECK:   tensor.empty{{.*}} : tensor<?x10x40xf32>
//       CHECK:   tensor.extract_slice
//       CHECK:   tensor.extract_slice
