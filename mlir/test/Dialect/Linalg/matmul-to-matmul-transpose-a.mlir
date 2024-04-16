// RUN: mlir-opt -linalg-matmul-to-matmul-transpose-a -cse -canonicalize -split-input-file %s | FileCheck %s

// CHECK-LABEL:   func.func @static(
// CHECK-SAME:                      %[[A:.*]]: tensor<16x8xf32>,
// CHECK-SAME:                      %[[B:.*]]: tensor<8x16xf32>) -> tensor<16x16xf32> {
// CHECK:           %[[C0_F32:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[C_INIT:.*]] = tensor.empty() : tensor<16x16xf32>
// CHECK:           %[[C_ZERO:.*]] = linalg.fill ins(%[[C0_F32]] : f32) outs(%[[C_INIT]] : tensor<16x16xf32>) -> tensor<16x16xf32>
// CHECK:           %[[A_TRANSP_INIT:.*]] = tensor.empty() : tensor<8x16xf32>
// CHECK:           %[[A_TRANSP:.*]] = linalg.transpose ins(%[[A]] : tensor<16x8xf32>) outs(%[[A_TRANSP_INIT]] : tensor<8x16xf32>) permutation = [1, 0]
// CHECK:           %[[C:.*]] = linalg.matmul_transpose_a ins(%[[A_TRANSP]], %[[B]] : tensor<8x16xf32>, tensor<8x16xf32>) outs(%[[C_ZERO]] : tensor<16x16xf32>) -> tensor<16x16xf32>
// CHECK:           return %[[C]] : tensor<16x16xf32>
// CHECK:         }
func.func @static(%A: tensor<16x8xf32>, %B: tensor<8x16xf32>) -> (tensor<16x16xf32>) {
  %cst = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<16x16xf32>
  %C = linalg.fill ins(%cst : f32) outs(%init : tensor<16x16xf32>) -> tensor<16x16xf32>
  %0 = linalg.matmul ins(%A, %B : tensor<16x8xf32>, tensor<8x16xf32>) outs(%C : tensor<16x16xf32>) -> tensor<16x16xf32>
  return %0 : tensor<16x16xf32>
}

//-----

// CHECK-LABEL:   func.func @dynamic(
// CHECK-SAME:                       %[[A:.*]]: tensor<?x?xf32>,
// CHECK-SAME:                       %[[B:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32> {
// CHECK:           %[[C0_F32:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[C1:.*]] = arith.constant 1 : index
// CHECK:           %[[A_DIM0:.*]] = tensor.dim %[[A]], %[[C0]] : tensor<?x?xf32>
// CHECK:           %[[B_DIM1:.*]] = tensor.dim %[[B]], %[[C1]] : tensor<?x?xf32>
// CHECK:           %[[C_INIT:.*]] = tensor.empty(%[[A_DIM0]], %[[B_DIM1]]) : tensor<?x?xf32>
// CHECK:           %[[C_ZERO:.*]] = linalg.fill ins(%[[C0_F32]] : f32) outs(%[[C_INIT]] : tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           %[[A_DIM1:.*]] = tensor.dim %[[A]], %[[C1]] : tensor<?x?xf32>
// CHECK:           %[[A_TRANSP_INIT:.*]] = tensor.empty(%[[A_DIM1]], %[[A_DIM0]]) : tensor<?x?xf32>
// CHECK:           %[[A_TRANSP:.*]] = linalg.transpose ins(%[[A]] : tensor<?x?xf32>) outs(%[[A_TRANSP_INIT]] : tensor<?x?xf32>) permutation = [1, 0]
// CHECK:           %[[C:.*]] = linalg.matmul_transpose_a ins(%[[A_TRANSP]], %[[B]] : tensor<?x?xf32>, tensor<?x?xf32>) outs(%[[C_ZERO]] : tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:           return %[[C]] : tensor<?x?xf32>
// CHECK:         }
func.func @dynamic(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>) -> (tensor<?x?xf32>) {
  %cst = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %A, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %B, %c1 : tensor<?x?xf32>
  %init = tensor.empty(%d0, %d1) : tensor<?x?xf32>
  %C = linalg.fill ins(%cst : f32) outs(%init : tensor<?x?xf32>) -> tensor<?x?xf32>
  %0 = linalg.matmul ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>) outs(%C : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

//-----

// CHECK-LABEL:   func.func @mixed(
// CHECK-SAME:                     %[[A:.*]]: tensor<?x8xf32>,
// CHECK-SAME:                     %[[B:.*]]: tensor<8x16xf32>) -> tensor<?x16xf32> {
// CHECK:           %[[C0_F32:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[A_DIM0:.*]] = tensor.dim %[[A]], %[[C0]] : tensor<?x8xf32>
// CHECK:           %[[C_INIT:.*]] = tensor.empty(%[[A_DIM0]]) : tensor<?x16xf32>
// CHECK:           %[[C_ZERO:.*]] = linalg.fill ins(%[[C0_F32]] : f32) outs(%[[C_INIT]] : tensor<?x16xf32>) -> tensor<?x16xf32>
// CHECK:           %[[A_TRANSP_INIT:.*]] = tensor.empty(%[[A_DIM0]]) : tensor<8x?xf32>
// CHECK:           %[[A_TRANSP:.*]] = linalg.transpose ins(%[[A]] : tensor<?x8xf32>) outs(%[[A_TRANSP_INIT]] : tensor<8x?xf32>) permutation = [1, 0]
// CHECK:           %[[B0:.*]] = linalg.matmul_transpose_a ins(%[[A_TRANSP]], %[[B]] : tensor<8x?xf32>, tensor<8x16xf32>) outs(%[[C_ZERO]] : tensor<?x16xf32>) -> tensor<?x16xf32>
// CHECK:           return %[[B0]] : tensor<?x16xf32>
// CHECK:         }
func.func @mixed(%A: tensor<?x8xf32>, %B: tensor<8x16xf32>) -> (tensor<?x16xf32>) {
  %cst = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %A, %c0 : tensor<?x8xf32>
  %init = tensor.empty(%d0) : tensor<?x16xf32>
  %C = linalg.fill ins(%cst : f32) outs(%init : tensor<?x16xf32>) -> tensor<?x16xf32>
  %0 = linalg.matmul ins(%A, %B : tensor<?x8xf32>, tensor<8x16xf32>) outs(%C : tensor<?x16xf32>) -> tensor<?x16xf32>
  return %0 : tensor<?x16xf32>
}
