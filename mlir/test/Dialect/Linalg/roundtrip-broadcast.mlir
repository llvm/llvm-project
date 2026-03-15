// RUN: mlir-opt %s -linalg-generalize-named-ops | mlir-opt --linalg-specialize-generic-ops | FileCheck %s

// CHECK-LABEL: broadcast_first_dimension
// CHECK-SAME:   %[[A:.+]]: tensor<?x?xf32>, %[[Out:.+]]: tensor<?x?x?xf32>)
// CHECK-NOT:     linalg.generic
// CHECK:         %broadcasted = linalg.broadcast ins(%[[A]] : tensor<?x?xf32>) outs(%[[Out]] : tensor<?x?x?xf32>) dimensions = [0]
//
func.func @broadcast_first_dimension(%A: tensor<?x?xf32>, %Out: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
   %res = linalg.broadcast ins(%A: tensor<?x?xf32>) outs(%Out: tensor<?x?x?xf32>) dimensions = [0]
  return %res : tensor<?x?x?xf32>
}

// CHECK-LABEL: broadcast_mid_dimension
// CHECK-SAME:   %[[A:.+]]: tensor<3x5xf32>, %[[Out:.+]]: tensor<3x4x5xf32>)
// CHECK-NOT:     linalg.generic
// CHECK:         %broadcasted = linalg.broadcast ins(%[[A]] : tensor<3x5xf32>) outs(%[[Out]] : tensor<3x4x5xf32>) dimensions = [1]
//
func.func @broadcast_mid_dimension(%A: tensor<3x5xf32>, %Out: tensor<3x4x5xf32>) -> tensor<3x4x5xf32> {
   %res = linalg.broadcast ins(%A: tensor<3x5xf32>) outs(%Out: tensor<3x4x5xf32>) dimensions = [1]
  return %res : tensor<3x4x5xf32>
}


// CHECK-LABEL: broadcast_multiple_dimensions
// CHECK-SAME:   %[[A:.+]]: tensor<4x5x7xf32>, %[[Out:.+]]: tensor<3x4x5x6x7x8x9xf32>)
// CHECK-NOT:     linalg.generic
// CHECK:         %broadcasted = linalg.broadcast ins(%[[A]] : tensor<4x5x7xf32>) outs(%[[Out]] : tensor<3x4x5x6x7x8x9xf32>) dimensions = [0, 3, 5, 6]
//
func.func @broadcast_multiple_dimensions(%A: tensor<4x5x7xf32>, %Out: tensor<3x4x5x6x7x8x9xf32>) -> tensor<3x4x5x6x7x8x9xf32> {
   %res = linalg.broadcast ins(%A: tensor<4x5x7xf32>) outs(%Out: tensor<3x4x5x6x7x8x9xf32>) dimensions = [0,3,5,6]
  return %res : tensor<3x4x5x6x7x8x9xf32>
}
