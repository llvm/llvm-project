// RUN: mlir-opt %s -linalg-generalize-named-ops | mlir-opt --linalg-specialize-generic-ops | FileCheck %s

// CHECK-LABEL: transpose2D
// CHECK-SAME:  %[[A:.+]]: tensor<16x64xf32>, %[[Out:.+]]: tensor<64x16xf32>
// CHECK-NOT:   linalg.generic
// CHECK:  %transposed = linalg.transpose ins(%[[A]] : tensor<16x64xf32>) outs(%[[Out]] : tensor<64x16xf32>) permutation = [1, 0]
//
func.func @transpose2D(%A: tensor<16x64xf32>, %Out: tensor<64x16xf32>) -> tensor<64x16xf32> {
  %res = linalg.transpose ins(%A: tensor<16x64xf32>) outs(%Out: tensor<64x16xf32>) permutation = [1,0]
  return %res : tensor<64x16xf32>
}


// CHECK-LABEL: transpose3D
// CHECK-SAME:  %[[A:.+]]: tensor<7x8x9xf32>, %[[Out:.+]]: tensor<9x7x8xf32>
// CHECK-NOT:   linalg.generic
// CHECK:  %transposed = linalg.transpose ins(%[[A]] : tensor<7x8x9xf32>) outs(%[[Out]] : tensor<9x7x8xf32>) permutation = [2, 0, 1]
//
func.func @transpose3D(%arg0: tensor<7x8x9xf32>, %arg1: tensor<9x7x8xf32>) -> tensor<9x7x8xf32> {
  %transposed = linalg.transpose ins(%arg0 : tensor<7x8x9xf32>) outs(%arg1 : tensor<9x7x8xf32>) permutation = [2, 0, 1]
  return %transposed : tensor<9x7x8xf32>
}
