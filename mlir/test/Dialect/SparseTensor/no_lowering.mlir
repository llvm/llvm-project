// RUN: mlir-opt %s --lower-sparse-ops-to-foreach --split-input-file | FileCheck %s

// Ensure that we exit gracefully rather than crashing.

// CHECK-LABEL: func.func @test_tensor_dim_unranked
//       CHECK: tensor.dim
func.func @test_tensor_dim_unranked(%arg0: tensor<*xf32>) -> index {
  %c = arith.constant 0 : index
  %0 = tensor.dim %arg0, %c : tensor<*xf32>
  return %0 : index
}

// -----

#SparseVector = #sparse_tensor.encoding<{
  map = (d0) -> (d0 : compressed)
}>

// CHECK-LABEL: func.func @test_no_constant_dim
//       CHECK: tensor.dim
func.func @test_no_constant_dim(%arg0: tensor<?xf64, #SparseVector>, %arg1: index) -> index {
  %0 = tensor.dim %arg0, %arg1 : tensor<?xf64, #SparseVector>
  return %0 : index
}

// -----

// CHECK-LABEL: func.func @test_tensor_dim_no_encoding
//       CHECK: tensor.dim
func.func @test_tensor_dim_no_encoding(%arg0: tensor<?xf32>) -> index {
  %c = arith.constant 0 : index
  %0 = tensor.dim %arg0, %c : tensor<?xf32>
  return %0 : index
}

// -----

// CHECK-LABEL: func.func @test_tensor_reshape_unranked
//       CHECK: tensor.reshape
func.func @test_tensor_reshape_unranked(%src: tensor<*xf32>, %shape: tensor<1xi32>) -> tensor<?xf32> {
  %dst = tensor.reshape %src(%shape)
         : (tensor<*xf32>, tensor<1xi32>) -> tensor<?xf32>
  return %dst : tensor<?xf32>
}

// -----

// CHECK-LABEL: func.func @test_tensor_reshape_no_encoding
//       CHECK: tensor.reshape
func.func @test_tensor_reshape_no_encoding(%src: tensor<?x?xf32>, %shape: tensor<1xi32>) -> tensor<?xf32> {
  %dst = tensor.reshape %src(%shape)
         : (tensor<?x?xf32>, tensor<1xi32>) -> tensor<?xf32>
  return %dst : tensor<?xf32>
}
