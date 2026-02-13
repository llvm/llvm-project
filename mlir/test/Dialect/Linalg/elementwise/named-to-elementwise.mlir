// RUN: mlir-opt %s -linalg-morph-ops=named-to-category -split-input-file | FileCheck %s

// CHECK: @exp(%[[A:.+]]: tensor<16x8xf32>, %[[B:.+]]: tensor<16x8xf32>) ->  tensor<16x8xf32> {
// CHECK: {{.*}} = linalg.elementwise
// CHECK-SAME:       kind=#linalg.elementwise_kind<exp>
// CHECK-SAME:       ins(%[[A]] : tensor<16x8xf32>)
// CHECK-SAME:       outs(%[[B]] : tensor<16x8xf32>) -> tensor<16x8xf32>
//
func.func @exp(%A : tensor<16x8xf32>, %B : tensor<16x8xf32>) ->  tensor<16x8xf32> {
  %exp = linalg.exp ins(%A : tensor<16x8xf32>) outs(%B :  tensor<16x8xf32>) -> tensor<16x8xf32>
  return %exp :  tensor<16x8xf32>
}

// ----

// CHECK: @add(%[[A:.+]]: tensor<16x8xf32>, %[[B:.+]]: tensor<16x8xf32>, %[[C:.+]]: tensor<16x8xf32>) ->  tensor<16x8xf32> {
// CHECK: {{.*}} = linalg.elementwise
// CHECK-SAME:       kind=#linalg.elementwise_kind<add>
// CHECK-SAME:       ins(%[[A]], %[[B]] : tensor<16x8xf32>, tensor<16x8xf32>)
// CHECK-SAME:       outs(%[[C]] : tensor<16x8xf32>) -> tensor<16x8xf32>
//
func.func @add(%A : tensor<16x8xf32>, %B: tensor<16x8xf32>, %C : tensor<16x8xf32>) ->  tensor<16x8xf32> {
  %add = linalg.add ins(%A, %B : tensor<16x8xf32>, tensor<16x8xf32>) outs(%C :  tensor<16x8xf32>) -> tensor<16x8xf32>
  return %add :  tensor<16x8xf32>
}

// ----

// CHECK: @sub(%[[A:.+]]: tensor<16x8xf32>, %[[B:.+]]: tensor<16x8xf32>, %[[C:.+]]: tensor<16x8xf32>) -> tensor<16x8xf32> {
// CHECK: {{.*}} = linalg.elementwise
// CHECK-SAME:       kind=#linalg.elementwise_kind<sub>
// CHECK-SAME:       ins(%[[A]], %[[B]] : tensor<16x8xf32>, tensor<16x8xf32>)
// CHECK-SAME:       outs(%[[C]] : tensor<16x8xf32>)
//
func.func @sub(%A : tensor<16x8xf32>, %B: tensor<16x8xf32>, %C : tensor<16x8xf32>) -> tensor<16x8xf32> {
  %sub = linalg.sub ins(%A, %B : tensor<16x8xf32>, tensor<16x8xf32>) outs(%C :  tensor<16x8xf32>) -> tensor<16x8xf32>
  return %sub : tensor<16x8xf32>
}

// ----

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
