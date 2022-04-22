// RUN: mlir-opt %s -linalg-comprehensive-module-bufferize="test-analysis-only allow-return-allocs init-tensor-elimination" -split-input-file | FileCheck %s

// -----

//===----------------------------------------------------------------------===//
// InitTensorOp elimination
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @buffer_forwarding_conflict
func.func @buffer_forwarding_conflict(%arg0: tensor<?xf32> {linalg.inplaceable = true}, %arg1: index) -> (tensor<?xf32>, tensor<?xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  //      CHECK: tensor.extract_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["false", "none"]
  // Instead of allocating, share buffer with some inplace bufferization?
  %0 = linalg.init_tensor [%arg1] : tensor<?xf32>

  //      CHECK: linalg.fill
  // CHECK-SAME: {__inplace_operands_attr__ = ["none", "true"]
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<?xf32>) -> tensor<?xf32>

  //      CHECK: tensor.insert_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "false", "none"]
  %2 = tensor.insert_slice %1 into %arg0[0] [%arg1] [1] : tensor<?xf32> into tensor<?xf32>

  //      CHECK: tensor.insert_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "true", "none"]
  %3 = tensor.insert_slice %1 into %arg0[42] [%arg1] [1] : tensor<?xf32> into tensor<?xf32>

  //      CHECK: return
  // CHECK-SAME: __equivalent_func_args__ = [-1, 0]
  return %2, %3 : tensor<?xf32>, tensor<?xf32>
}

// -----

// CHECK-LABEL: func @buffer_forwarding_no_conflict
func.func @buffer_forwarding_no_conflict(%arg0: tensor<?xf32> {linalg.inplaceable = true}, %arg1: index) -> (tensor<?xf32>, tensor<?xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  //      CHECK: tensor.extract_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "none"]
  // Instead of allocating, share buffer with some inplace bufferization?
  %0 = linalg.init_tensor [%arg1] : tensor<?xf32>

  //      CHECK: linalg.fill
  // CHECK-SAME: {__inplace_operands_attr__ = ["none", "true"]
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<?xf32>) -> tensor<?xf32>

  //      CHECK: tensor.insert_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "true", "none"]
  %2 = tensor.insert_slice %1 into %arg0[42] [%arg1] [1] : tensor<?xf32> into tensor<?xf32>

  //      CHECK: return
  // CHECK-SAME: __equivalent_func_args__ = [0, 0]
  return %2, %2 : tensor<?xf32>, tensor<?xf32>
}
