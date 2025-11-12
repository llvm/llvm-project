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
func.func @unary_transpose(%A: tensor<16x8x32xf32>, %B: tensor<8x16x32xf32>) -> tensor<8x16x32xf32> {
  %empty = tensor.empty() : tensor<8x16x32xf32>
  %transposed_A = linalg.transpose ins(%A : tensor<16x8x32xf32>) outs(%empty : tensor<8x16x32xf32>) permutation = [1, 0, 2]
  %result = linalg.elementwise kind=#linalg.elementwise_kind<exp>
                          ins(%transposed_A : tensor<8x16x32xf32>) outs(%B : tensor<8x16x32xf32>) -> tensor<8x16x32xf32>
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
func.func @binary_transposed(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %C: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim0 = tensor.dim %A, %c0 : tensor<?x?xf32>
  %dim1 = tensor.dim %A, %c1 : tensor<?x?xf32>

  %empty = tensor.empty(%dim1, %dim0) : tensor<?x?xf32>
  %transposed_B = linalg.transpose ins(%B : tensor<?x?xf32>) outs(%empty : tensor<?x?xf32>) permutation = [1, 0]
  %result = linalg.elementwise kind=#linalg.elementwise_kind<add>
                          ins(%A, %transposed_B : tensor<?x?xf32>, tensor<?x?xf32>)
                          outs(%C : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %result : tensor<?x?xf32>
}

// -----

// CHECK-DAG: #[[IDENTITY:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-DAG: #[[BROADCASTED:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
//
// CHECK:  func.func @unary_broadcasted(%[[A:.+]]: tensor<8x32xf32>, %[[B:.+]]: tensor<8x16x32xf32>) -> tensor<8x16x32xf32> {
// CHECK-NEXT:  %[[RES:.+]] = linalg.elementwise kind=#linalg.elementwise_kind<exp>
// CHECK-SAME:       indexing_maps = [#[[BROADCASTED]], #[[IDENTITY]]]
// CHECK-SAME:       ins(%[[A]] : tensor<8x32xf32>) outs(%[[B]] : tensor<8x16x32xf32>) -> tensor<8x16x32xf32>
// CHECK-NEXT:    return %[[RES]] : tensor<8x16x32xf32>
//
func.func @unary_broadcasted(%A: tensor<8x32xf32>, %B: tensor<8x16x32xf32>) -> tensor<8x16x32xf32> {
  %empty = tensor.empty() : tensor<8x16x32xf32>
  %broadcasted_A = linalg.broadcast ins(%A : tensor<8x32xf32>) outs(%empty : tensor<8x16x32xf32>) dimensions = [1]
  %result = linalg.elementwise kind=#linalg.elementwise_kind<exp>
                          ins(%broadcasted_A : tensor<8x16x32xf32>) outs(%B : tensor<8x16x32xf32>) -> tensor<8x16x32xf32>
  return %result : tensor<8x16x32xf32>
}

// -----

// CHECK-DAG: #[[IDENTITY:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: #[[BROADCASTED:.+]] = affine_map<(d0, d1) -> (d0)>
//
// CHECK:  func.func @binary_broadcasted(%[[A:.+]]: tensor<?x?xf32>, %[[B:.+]]: tensor<?xf32>, %[[C:.+]]: tensor<?x?xf32>) -> tensor<?x?xf32> {
// CHECK-NEXT:  %[[RES:.+]] = linalg.elementwise kind=#linalg.elementwise_kind<add>
// CHECK-SAME:              indexing_maps = [#[[IDENTITY]], #[[BROADCASTED]], #[[IDENTITY]]]
// CHECK-SAME:              ins(%[[A]], %[[B]] : tensor<?x?xf32>, tensor<?xf32>) outs(%[[C]] : tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK-NEXT:  return %[[RES]] : tensor<?x?xf32>
//
func.func @binary_broadcasted(%A: tensor<?x?xf32>, %B: tensor<?xf32>, %C: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim0 = tensor.dim %A, %c0 : tensor<?x?xf32>
  %dim1 = tensor.dim %A, %c1 : tensor<?x?xf32>

  %empty = tensor.empty(%dim1, %dim0) : tensor<?x?xf32>
  %broadcasted_B = linalg.broadcast ins(%B : tensor<?xf32>) outs(%empty : tensor<?x?xf32>) dimensions = [1]
  %result = linalg.elementwise kind=#linalg.elementwise_kind<add>
                          ins(%A, %broadcasted_B : tensor<?x?xf32>, tensor<?x?xf32>)
                          outs(%C : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %result : tensor<?x?xf32>
}

// -----

// CHECK-DAG: #[[IDENTITY:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: #[[COMPOSED_MAP:.+]] = affine_map<(d0, d1) -> (d0)>
//
// CHECK:  func.func @fold_broadcast_after_transpose_fold(%[[A:.+]]: tensor<16xf32>, %[[B:.+]]: tensor<16x32xf32>) -> tensor<16x32xf32> {
// CHECK-NEXT:  %[[RES:.+]] = linalg.elementwise kind=#linalg.elementwise_kind<exp>
// CHECK-SAME:              indexing_maps = [#[[COMPOSED_MAP]], #[[IDENTITY]]]
// CHECK-SAME:              ins(%[[A]] : tensor<16xf32>) outs(%[[B]] : tensor<16x32xf32>) -> tensor<16x32xf32>
// CHECK-NEXT:  return %[[RES]] : tensor<16x32xf32>
//
#identity = affine_map<(d0, d1) -> (d0, d1)>
#transpose = affine_map<(d0, d1) -> (d1, d0)>

func.func @fold_broadcast_after_transpose_fold(%A: tensor<16xf32>, %B: tensor<16x32xf32>) -> tensor<16x32xf32> {
  %empty_b = tensor.empty() : tensor<32x16xf32>

  %broadcasted_A = linalg.broadcast ins(%A : tensor<16xf32>) outs(%empty_b : tensor<32x16xf32>) dimensions = [0]

  %result = linalg.elementwise kind=#linalg.elementwise_kind<exp>
                          indexing_maps = [#transpose, #identity]
                          ins(%broadcasted_A : tensor<32x16xf32>) outs(%B : tensor<16x32xf32>) -> tensor<16x32xf32>
  return %result : tensor<16x32xf32>
}

// -----

// CHECK-DAG: #[[IDENTITY:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-DAG: #[[COMPOSED_MAP:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
//
// CHECK:  func.func @fold_transpose_after_broadcast_fold(%[[A:.+]]: tensor<32x16xf32>, %[[B:.+]]: tensor<8x16x32xf32>) -> tensor<8x16x32xf32> {
// CHECK-NEXT:  %[[RES:.+]] = linalg.elementwise kind=#linalg.elementwise_kind<exp>
// CHECK-SAME:              indexing_maps = [#[[COMPOSED_MAP]], #[[IDENTITY]]]
// CHECK-SAME:              ins(%[[A]] : tensor<32x16xf32>) outs(%[[B]] : tensor<8x16x32xf32>) -> tensor<8x16x32xf32>
// CHECK-NEXT:  return %[[RES]] : tensor<8x16x32xf32>
//
#identity = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#broadcast = affine_map<(d0, d1, d2) -> (d1, d2)>

func.func @fold_transpose_after_broadcast_fold(%A: tensor<32x16xf32>, %B: tensor<8x16x32xf32>) -> tensor<8x16x32xf32> {
  %empty_t = tensor.empty() : tensor<16x32xf32>
  %transposed_A = linalg.transpose ins(%A : tensor<32x16xf32>) outs(%empty_t : tensor<16x32xf32>) permutation = [1, 0]

  %result = linalg.elementwise kind=#linalg.elementwise_kind<exp>
                          indexing_maps = [#broadcast, #identity]
                          ins(%transposed_A : tensor<16x32xf32>) outs(%B : tensor<8x16x32xf32>) -> tensor<8x16x32xf32>
  return %result : tensor<8x16x32xf32>
}

// -----

// CHECK-DAG: #[[IDENTITY:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: #[[COMPOSED_MAP:.+]] = affine_map<(d0, d1) -> (d0)>
//
// CHECK:  func.func @fold_broadcast_after_transpose_fold_binary(%[[A:.+]]: tensor<?xf32>, %[[B:.+]]: tensor<?x?xf32>, %[[C:.+]]: tensor<?x?xf32>) -> tensor<?x?xf32> {
// CHECK-NEXT:  %[[RES:.+]] = linalg.elementwise kind=#linalg.elementwise_kind<add>
// CHECK-SAME:              indexing_maps = [#[[COMPOSED_MAP]], #[[IDENTITY]], #[[IDENTITY]]]
// CHECK-SAME:              ins(%[[A]], %[[B]] : tensor<?xf32>, tensor<?x?xf32>) outs(%[[C]] : tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK-NEXT:  return %[[RES]] : tensor<?x?xf32>
//
#identity = affine_map<(d0, d1) -> (d0, d1)>
#transpose = affine_map<(d0, d1) -> (d1, d0)>

func.func @fold_broadcast_after_transpose_fold_binary(%A: tensor<?xf32>, %B: tensor<?x?xf32>, %C: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim0 = tensor.dim %B, %c0 : tensor<?x?xf32>
  %dim1 = tensor.dim %B, %c1 : tensor<?x?xf32>

  %empty_b = tensor.empty(%dim1, %dim0) : tensor<?x?xf32>
  %broadcasted_A = linalg.broadcast ins(%A : tensor<?xf32>) outs(%empty_b : tensor<?x?xf32>) dimensions = [0]

  %result = linalg.elementwise kind=#linalg.elementwise_kind<add>
                          indexing_maps = [#transpose, #identity, #identity]
                          ins(%broadcasted_A, %B : tensor<?x?xf32>, tensor<?x?xf32>) outs(%C : tensor<?x?xf32>) -> tensor<?x?xf32>

  return %result : tensor<?x?xf32>
}

// -----

// CHECK-DAG: #[[IDENTITY:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-DAG: #[[COMPOSED_MAP:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
//
// CHECK:  func.func @fold_transpose_after_broadcast_fold_binary(%[[A:.+]]: tensor<?x?xf32>, %[[B:.+]]: tensor<?x?x?xf32>, %[[C:.+]]: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
// CHECK-NEXT:  %[[RES:.+]] = linalg.elementwise kind=#linalg.elementwise_kind<add>
// CHECK-SAME:              indexing_maps = [#[[COMPOSED_MAP]], #[[IDENTITY]], #[[IDENTITY]]]
// CHECK-SAME:              ins(%[[A]], %[[B]] : tensor<?x?xf32>, tensor<?x?x?xf32>) outs(%[[C]] : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
// CHECK-NEXT:  return %[[RES]] : tensor<?x?x?xf32>
//
#identity = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#broadcast = affine_map<(d0, d1, d2) -> (d1, d2)>

func.func @fold_transpose_after_broadcast_fold_binary(%A: tensor<?x?xf32>, %B: tensor<?x?x?xf32>, %C: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %dim0 = tensor.dim %B, %c0 : tensor<?x?x?xf32>
  %dim1 = tensor.dim %B, %c1 : tensor<?x?x?xf32>
  %dim2 = tensor.dim %B, %c2 : tensor<?x?x?xf32>

  %empty_t = tensor.empty(%dim1, %dim2) : tensor<?x?xf32>
  %transposed_A = linalg.transpose ins(%A : tensor<?x?xf32>) outs(%empty_t : tensor<?x?xf32>) permutation = [1, 0]

  %result = linalg.elementwise kind=#linalg.elementwise_kind<add>
                          indexing_maps = [#broadcast, #identity, #identity]
                          ins(%transposed_A, %B : tensor<?x?xf32>, tensor<?x?x?xf32>) outs(%C : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  return %result : tensor<?x?x?xf32>
}

// -----

// CHECK-DAG: #[[IDENTITY:.+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG: #[[DIAGONAL:.+]] = affine_map<(d0) -> (d0, d0)>
//
// CHECK:  func.func @fold_failed_diagonal_map(%[[A:.+]]: tensor<16xf32>, %[[B:.+]]: tensor<16xf32>, %[[C:.+]]: tensor<16xf32>) -> tensor<16xf32> {
// CHECK-NEXT:  %[[EMPTY:.+]] = tensor.empty() : tensor<16x16xf32>
// CHECK-NEXT:  %[[BROADCASTED_B:.+]] = linalg.broadcast ins(%[[B]] : tensor<16xf32>) outs(%[[EMPTY]] : tensor<16x16xf32>) dimensions = [0]
// CHECK-NEXT:  %[[RES:.+]] = linalg.elementwise kind=#linalg.elementwise_kind<add>
// CHECK-SAME:              indexing_maps = [#[[IDENTITY]], #[[DIAGONAL]], #[[IDENTITY]]]
// CHECK-SAME:              ins(%[[A]], %[[BROADCASTED_B]] : tensor<16xf32>, tensor<16x16xf32>) outs(%[[C]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK-NEXT:  return %[[RES]] : tensor<16xf32>
//
#identity = affine_map<(d0) -> (d0)>
#diagonal = affine_map<(d0) -> (d0, d0)>

func.func @fold_failed_diagonal_map(%A: tensor<16xf32>, %B: tensor<16xf32>, %C: tensor<16xf32>) -> tensor<16xf32> {
  %empty = tensor.empty() : tensor<16x16xf32>
  %broadcasted_B = linalg.broadcast ins(%B : tensor<16xf32>) outs(%empty : tensor<16x16xf32>) dimensions = [0]
  %result = linalg.elementwise kind=#linalg.elementwise_kind<add>
                          indexing_maps = [#identity, #diagonal, #identity]
                          ins(%A, %broadcasted_B : tensor<16xf32>, tensor<16x16xf32>) outs(%C : tensor<16xf32>) -> tensor<16xf32>
  return %result : tensor<16xf32>
}

// -----

// CHECK-DAG: #[[IDENTITY:.+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG: #[[CONSTANT:.+]] = affine_map<(d0) -> (0, d0)>
//
// CHECK:  func.func @fold_failed_constant_map(%[[A:.+]]: tensor<16xf32>, %[[B:.+]]: tensor<16x32xf32>, %[[C:.+]]: tensor<16xf32>) -> tensor<16xf32> {
// CHECK-NEXT:  %[[EMPTY:.+]] = tensor.empty() : tensor<32x16xf32>
// CHECK-NEXT:  %[[TRANSPOSED_B:.+]] = linalg.transpose ins(%[[B]] : tensor<16x32xf32>) outs(%[[EMPTY]] : tensor<32x16xf32>) permutation = [1, 0]
// CHECK-NEXT:  %[[RES:.+]] = linalg.elementwise kind=#linalg.elementwise_kind<add>
// CHECK-SAME:              indexing_maps = [#[[IDENTITY]], #[[CONSTANT]], #[[IDENTITY]]]
// CHECK-SAME:              ins(%[[A]], %[[TRANSPOSED_B]] : tensor<16xf32>, tensor<32x16xf32>) outs(%[[C]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK-NEXT:  return %[[RES]] : tensor<16xf32>
//
#identity = affine_map<(d0) -> (d0)>
#constant = affine_map<(d0) -> (0, d0)>

func.func @fold_failed_constant_map(%A: tensor<16xf32>, %B: tensor<16x32xf32>, %C: tensor<16xf32>) -> tensor<16xf32> {
  %empty = tensor.empty() : tensor<32x16xf32>
  %transposed_B = linalg.transpose ins(%B : tensor<16x32xf32>) outs(%empty : tensor<32x16xf32>) permutation = [1, 0]
  %result = linalg.elementwise kind=#linalg.elementwise_kind<add>
                          indexing_maps = [#identity, #constant, #identity]
                          ins(%A, %transposed_B : tensor<16xf32>, tensor<32x16xf32>) outs(%C : tensor<16xf32>) -> tensor<16xf32>
  return %result : tensor<16xf32>
}
