// RUN: mlir-opt %s -test-tensorlike-bufferlike -split-input-file | FileCheck %s

// CHECK: func.func @builtin_unranked
// CHECK-SAME: {found = {operand_0 = "is_tensor_like", result_0 = "is_buffer_like"}}
func.func @builtin_unranked(%t: tensor<*xf32>) -> (memref<*xf32>)
{
  %0 = bufferization.to_memref %t : tensor<*xf32> to memref<*xf32>
  return %0 : memref<*xf32>
}

// -----

// CHECK: func.func @builtin_ranked
// CHECK-SAME: {found = {operand_0 = "is_tensor_like", result_0 = "is_buffer_like"}}
func.func @builtin_ranked(%t: tensor<42xf32>) -> (memref<42xf32>)
{
  %0 = bufferization.to_memref %t : tensor<42xf32> to memref<42xf32>
  return %0 : memref<42xf32>
}

// -----

// CHECK: func.func @custom_tensor
// CHECK-SAME: {found = {operand_0 = "is_tensor_like"}}
func.func @custom_tensor(%t: !test.test_tensor<[42], f32>) -> ()
{
  return
}

// -----

// CHECK: func.func @custom_memref
// CHECK-SAME: {found = {operand_0 = "is_buffer_like"}}
func.func @custom_memref(%t: !test.test_memref<[42], f32>) -> ()
{
  return
}
