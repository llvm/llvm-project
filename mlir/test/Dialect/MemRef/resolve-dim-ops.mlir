// RUN: mlir-opt --resolve-ranked-shaped-type-result-dims --split-input-file %s | FileCheck %s

// CHECK-LABEL: func @dim_out_of_bounds(
//  CHECK-NEXT:   arith.constant
//  CHECK-NEXT:   memref.dim
//  CHECK-NEXT:   return
func.func @dim_out_of_bounds(%m : memref<7x8xf32>) -> index {
  %idx = arith.constant 7 : index
  %0 = memref.dim %m, %idx : memref<7x8xf32>
  return %0 : index
}

// -----

// CHECK-LABEL: func @dim_out_of_bounds_2(
//  CHECK-NEXT:   arith.constant
//  CHECK-NEXT:   arith.constant
//  CHECK-NEXT:   bufferization.alloc_tensor
//  CHECK-NEXT:   tensor.dim
//  CHECK-NEXT:   return
func.func @dim_out_of_bounds_2(%idx1 : index, %idx2 : index) -> index {
  %idx = arith.constant 7 : index
  %sz = arith.constant 5 : index
  %alloc = bufferization.alloc_tensor(%sz, %sz) : tensor<?x?xf32>
  %0 = tensor.dim %alloc, %idx : tensor<?x?xf32>
  return %0 : index
}

// -----

// CHECK-LABEL:   func.func @dynamic_dim_of_transpose_op(
//  CHECK-SAME:                                   %[[arg:.*]]: tensor<1x2x?x8xi8>) -> index {
//  CHECK-NEXT:           %[[c2:.*]] = arith.constant 2
//  CHECK-NEXT:           tensor.dim %[[arg]], %[[c2]]
//  CHECK-NEXT:           return
func.func @dynamic_dim_of_transpose_op(%arg0: tensor<1x2x?x8xi8>) -> index {
  %0 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
  %1 = tosa.transpose %arg0, %0 : (tensor<1x2x?x8xi8>, tensor<4xi32>) -> tensor<1x8x2x?xi8>
  %c3 = arith.constant 3 : index
  %dim = tensor.dim %1, %c3 : tensor<1x8x2x?xi8>
  return %dim : index
}

// -----

// CHECK-LABEL:   func.func @static_dim_of_transpose_op(
//  CHECK:           arith.constant 100 : index
//  CHECK:           return
func.func @static_dim_of_transpose_op(%arg0: tensor<1x100x?x8xi8>) -> index {
  %0 = "tosa.const"() <{value = dense<[0, 3, 1, 2]> : tensor<4xi32>}> : () -> tensor<4xi32>
  %1 = tosa.transpose %arg0, %0 : (tensor<1x100x?x8xi8>, tensor<4xi32>) -> tensor<1x8x100x?xi8>
  %c2 = arith.constant 2 : index
  %dim = tensor.dim %1, %c2 : tensor<1x8x100x?xi8>
  return %dim : index
}
