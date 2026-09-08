// RUN: mlir-opt %s -linalg-merge-consecutive-reduce-ops -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @merge_consecutive_reduce(
// CHECK-SAME:    %[[INPUT:[a-zA-Z0-9_]+]]: tensor<2x3x4x5xf32>
// CHECK-SAME:    %[[INIT:[a-zA-Z0-9_]+]]: tensor<f32>
// CHECK:         %[[REDUCED:.+]] = linalg.reduce { arith.addf } ins(%[[INPUT]] : tensor<2x3x4x5xf32>) outs(%[[INIT]] : tensor<f32>) dimensions = [0, 1, 2, 3]
// CHECK-NEXT:    return %[[REDUCED]] : tensor<f32>
func.func @merge_consecutive_reduce(
    %input: tensor<2x3x4x5xf32>, %init: tensor<f32>) -> tensor<f32> {
  %cst = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<3x5xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<3x5xf32>) -> tensor<3x5xf32>
  %first_reduce = linalg.reduce { arith.addf }
      ins(%input : tensor<2x3x4x5xf32>)
      outs(%fill : tensor<3x5xf32>)
      dimensions = [0, 2]
  %second_reduce = linalg.reduce { arith.addf }
      ins(%first_reduce : tensor<3x5xf32>)
      outs(%init : tensor<f32>)
      dimensions = [0, 1]
  return %second_reduce : tensor<f32>
}

// -----

// CHECK-LABEL: func.func @merge_consecutive_reduce_with_projected_dims(
// CHECK-SAME:    %[[INPUT:[a-zA-Z0-9_]+]]: tensor<2x3x4x5x6xf32>
// CHECK-SAME:    %[[INIT:[a-zA-Z0-9_]+]]: tensor<5xf32>
// CHECK:         %[[REDUCED:.+]] = linalg.reduce { arith.addf } ins(%[[INPUT]] : tensor<2x3x4x5x6xf32>) outs(%[[INIT]] : tensor<5xf32>) dimensions = [0, 1, 2, 4]
// CHECK-NEXT:    return %[[REDUCED]] : tensor<5xf32>
func.func @merge_consecutive_reduce_with_projected_dims(
    %input: tensor<2x3x4x5x6xf32>, %init: tensor<5xf32>) -> tensor<5xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %empty = tensor.empty() : tensor<3x4x5xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<3x4x5xf32>) -> tensor<3x4x5xf32>
  %first_reduce = linalg.reduce { arith.addf }
      ins(%input : tensor<2x3x4x5x6xf32>)
      outs(%fill : tensor<3x4x5xf32>)
      dimensions = [0, 4]
  %second_reduce = linalg.reduce { arith.addf }
      ins(%first_reduce : tensor<3x4x5xf32>)
      outs(%init : tensor<5xf32>)
      dimensions = [0, 1]
  return %second_reduce : tensor<5xf32>
}
