// RUN: mlir-opt %s -split-input-file -verify-diagnostics | FileCheck %s

//===----------------------------------------------------------------------===//
// Test mixed normal and variadic operands
//===----------------------------------------------------------------------===//

func @correct_variadic_operand(%arg0: tensor<f32>, %arg1: f32) {
  // CHECK: mixed_normal_variadic_operand
  "test.mixed_normal_variadic_operand"(%arg0, %arg0, %arg0, %arg0, %arg0) : (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>) -> ()
  return
}

// -----

func @error_in_first_variadic_operand(%arg0: tensor<f32>, %arg1: f32) {
  // expected-error @+1 {{operand #1 must be tensor of any type}}
  "test.mixed_normal_variadic_operand"(%arg0, %arg1, %arg0, %arg0, %arg0) : (tensor<f32>, f32, tensor<f32>, tensor<f32>, tensor<f32>) -> ()
  return
}

// -----

func @error_in_normal_operand(%arg0: tensor<f32>, %arg1: f32) {
  // expected-error @+1 {{operand #2 must be tensor of any type}}
  "test.mixed_normal_variadic_operand"(%arg0, %arg0, %arg1, %arg0, %arg0) : (tensor<f32>, tensor<f32>, f32, tensor<f32>, tensor<f32>) -> ()
  return
}

// -----

func @error_in_second_variadic_operand(%arg0: tensor<f32>, %arg1: f32) {
  // expected-error @+1 {{operand #3 must be tensor of any type}}
  "test.mixed_normal_variadic_operand"(%arg0, %arg0, %arg0, %arg1, %arg0) : (tensor<f32>, tensor<f32>, tensor<f32>, f32, tensor<f32>) -> ()
  return
}

// -----

func @testfunc(%arg0: i32) {
  return
}
func @invalid_call_operandtype() {
  %0 = constant 0.0 : f32
  // expected-error @+1 {{operand type mismatch: expected operand type 'i32', but provided 'f32' for operand number 0}}
  call @testfunc(%0) : (f32) -> ()
  return
}
