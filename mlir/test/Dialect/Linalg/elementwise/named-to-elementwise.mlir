// RUN: mlir-opt %s -linalg-morph-ops=named-to-category -split-input-file | FileCheck %s

// CHECK: @ternary_select(%[[A:.+]]: tensor<4x8x16xi1>, %[[B:.+]]: tensor<4x8x16xf32>, %[[C:.+]]: tensor<4x8x16xf32>)
// CHECK:   %[[E:.+]] =  tensor.empty() : tensor<4x8x16xf32>
// CHECK: {{.*}} = linalg.elementwise
// CHECK-SAME:       kind=#linalg.elementwise_kind<select>
// CHECK-SAME:       ins(%[[A]], %[[B]], %[[C]] : tensor<4x8x16xi1>, tensor<4x8x16xf32>, tensor<4x8x16xf32>)
// CHECK-SAME:       outs(%[[E]] : tensor<4x8x16xf32>) -> tensor<4x8x16xf32>
//
func.func @ternary_select(%A: tensor<4x8x16xi1>, %B: tensor<4x8x16xf32>, %C: tensor<4x8x16xf32>)
             -> tensor<4x8x16xf32> {
  %empty = tensor.empty() : tensor<4x8x16xf32>
  %select = linalg.select
              ins(%A, %B, %C : tensor<4x8x16xi1>, tensor<4x8x16xf32>, tensor<4x8x16xf32>)
              outs(%empty: tensor<4x8x16xf32>) -> tensor<4x8x16xf32>
  return %select : tensor<4x8x16xf32>
}
