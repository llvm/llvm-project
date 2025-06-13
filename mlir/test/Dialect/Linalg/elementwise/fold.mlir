// RUN: mlir-opt %s -linalg-fold-into-elementwise -split-input-file | FileCheck %s

// CHECK-DAG: #[[IDENTITY:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-DAG: #[[TRANSPOSED:.+]] = affine_map<(d0, d1, d2) -> (d1, d0, d2)>
//
// CHECK:  func.func @unary_transpose(%[[A:.+]]: tensor<16x8x32xf32>, %[[B:.+]]: tensor<8x16x32xf32>) -> tensor<8x16x32xf32> {
// CHECK-NEXT:  %[[RES:.+]] = linalg.elementwise kind=#linalg.elementwise_kind<exp>
// CHECK-SAME:       indexing_maps = [#[[TRANSPOSED]], #[[IDENTITY]]]
// CHECK-SAME:       ins(%[[A]] : tensor<16x8x32xf32>) outs(%[[B]] : tensor<8x16x32xf32>) -> tensor<8x16x32xf32>
// CHECK-NEXT:    return %[[RES]] : tensor<8x16x32xf32>
//
func.func @unary_transpose(%A : tensor<16x8x32xf32>, %B: tensor<8x16x32xf32>) ->  tensor<8x16x32xf32> {
  %empty = tensor.empty() : tensor<8x16x32xf32>
  %transposed_A = linalg.transpose ins(%A : tensor<16x8x32xf32>) outs(%empty :  tensor<8x16x32xf32>) permutation = [1, 0, 2]
  %result = linalg.elementwise kind=#linalg.elementwise_kind<exp>
                          ins(%transposed_A : tensor<8x16x32xf32>) outs(%B: tensor<8x16x32xf32>) -> tensor<8x16x32xf32>
  return %result : tensor<8x16x32xf32>
}

// -----

// CHECK-DAG: #[[IDENTITY:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: #[[TRANSPOSED:.+]] = affine_map<(d0, d1) -> (d1, d0)>
//
// CHECK:  func.func @binary_transposed(%[[A:.+]]: tensor<?x?xf32>, %[[B:.+]]: tensor<?x?xf32>, %[[C:.+]]: tensor<?x?xf32>) -> tensor<?x?xf32> {
// CHECK-NEXT:  %[[RES:.+]] = linalg.elementwise kind=#linalg.elementwise_kind<add>
// CHECK-SAME:              indexing_maps = [#[[IDENTITY]], #[[TRANSPOSED]], #[[IDENTITY]]]
// CHECK-SAME:              ins(%[[A]], %[[B]] : tensor<?x?xf32>, tensor<?x?xf32>) outs(%[[C]] : tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK-NEXT:  return %[[RES]] : tensor<?x?xf32>
//
func.func @binary_transposed(%A : tensor<?x?xf32>, %B: tensor<?x?xf32>, %C: tensor<?x?xf32>) ->  tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim0 = tensor.dim %A, %c0 : tensor<?x?xf32>
  %dim1 = tensor.dim %A, %c1 : tensor<?x?xf32>

  %empty = tensor.empty(%dim1, %dim0) : tensor<?x?xf32>
  %transposed_B = linalg.transpose ins(%B : tensor<?x?xf32>) outs(%empty :  tensor<?x?xf32>) permutation = [1, 0]
  %result = linalg.elementwise kind=#linalg.elementwise_kind<add>
                          ins(%A, %transposed_B : tensor<?x?xf32>,  tensor<?x?xf32>)
                          outs(%C: tensor<?x?xf32>) -> tensor<?x?xf32>
  return %result : tensor<?x?xf32>
}
