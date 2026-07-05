// RUN: mlir-opt -split-input-file -test-tensor-transform-patterns=test-fold-extract-from-collapse-shape %s | FileCheck %s

// CHECK-LABEL: @extract_from_collapse_shape
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1x1x8xi8>)
func.func @extract_from_collapse_shape(%arg0: tensor<1x1x8xi8>) -> (i8, i8) {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %collapsed = tensor.collapse_shape %arg0 [[0, 1, 2]] : tensor<1x1x8xi8> into tensor<8xi8>
  %extracted = tensor.extract %collapsed[%c0] : tensor<8xi8>
  %extracted_0 = tensor.extract %collapsed[%c1] : tensor<8xi8>
  func.return %extracted, %extracted_0 : i8, i8
}

// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[RESULT0:.*]] = tensor.extract %[[ARG0]][%[[C0]], %[[C0]], %[[C0]]] : tensor<1x1x8xi8>
// CHECK-DAG: %[[RESULT1:.*]] = tensor.extract %[[ARG0]][%[[C0]], %[[C0]], %[[C1]]] : tensor<1x1x8xi8>
// CHECK-NEXT: return %[[RESULT0]], %[[RESULT1]] : i8, i8

// -----

// CHECK-LABEL: @extract_from_static_shape
// CHECK-SAME: (%[[ARG0:.*]]: tensor<2x6x32xf32>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index)
func.func @extract_from_static_shape(%arg0 : tensor<2x6x32xf32>, %arg1 : index, %arg2 : index) -> f32 {
  %0 = tensor.collapse_shape %arg0 [[0, 1], [2]] : tensor<2x6x32xf32> into tensor<12x32xf32>
  %1 = tensor.extract %0[%arg1, %arg2] : tensor<12x32xf32>
  return %1 : f32
}
// CHECK-NEXT: %[[MODIFIED_INDEXES:.*]]:2 = affine.delinearize_index %[[ARG1]] into (2, 6)
// CHECK-NEXT: %[[RESULT:.*]] = tensor.extract %[[ARG0]][%[[MODIFIED_INDEXES]]#0, %[[MODIFIED_INDEXES]]#1, %[[ARG2]]] : tensor<2x6x32xf32>
// CHECK-NEXT: return %[[RESULT]] : f32
