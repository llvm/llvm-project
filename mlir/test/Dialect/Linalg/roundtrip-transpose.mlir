// RUN: mlir-opt %s -linalg-generalize-named-ops | mlir-opt --linalg-specialize-generic-ops | FileCheck %s

// CHECK-LABEL: linalg_transpose
// CHECK-SAME:  %[[A:.+]]: tensor<16x64xf32>, %[[Out:.+]]: tensor<64x16xf32>
// CHECK-NOT:   linalg.generic
// CHECK:  %transposed = linalg.transpose ins(%[[A]] : tensor<16x64xf32>) outs(%[[Out]] : tensor<64x16xf32>) permutation = [1, 0]
//
func.func @linalg_transpose(%A: tensor<16x64xf32>, %Out: tensor<64x16xf32>) -> tensor<64x16xf32> {
  %res = linalg.transpose ins(%A: tensor<16x64xf32>) outs(%Out: tensor<64x16xf32>) permutation = [1,0]
  return %res : tensor<64x16xf32>
}
