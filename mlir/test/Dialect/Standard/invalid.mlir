// RUN: mlir-opt -split-input-file %s -verify-diagnostics

func @test_index_cast_shape_error(%arg0 : tensor<index>) -> tensor<2xi64> {
  // expected-error @+1 {{all non-scalar operands/results must have the same shape and base type}}
  %0 = index_cast %arg0 : tensor<index> to tensor<2xi64>
  return %0 : tensor<2xi64>
}

// -----

func @test_index_cast_tensor_error(%arg0 : tensor<index>) -> i64 {
  // expected-error @+1 {{if an operand is non-scalar, then there must be at least one non-scalar result}}
  %0 = index_cast %arg0 : tensor<index> to i64
  return %0 : i64
}

// -----

func @non_signless_constant() {
  // expected-error @+1 {{requires integer result types to be signless}}
  %0 = constant 0 : ui32
  return
}

// -----

func @non_signless_constant() {
  // expected-error @+1 {{requires integer result types to be signless}}
  %0 = constant 0 : si32
  return
}

// -----

func @unsupported_attribute() {
  // expected-error @+1 {{unsupported 'value' attribute: "" : index}}
  %0 = constant "" : index
  return
}

// -----

func @complex_constant_wrong_array_attribute_length() {
  // expected-error @+1 {{requires 'value' to be a complex constant, represented as array of two values}}
  %0 = constant [1.0 : f32] : complex<f32>
  return
}

// -----

func @complex_constant_wrong_attribute_type() {
  // expected-error @+1 {{requires attribute's type ('f32') to match op's return type ('complex<f32>')}}
  %0 = "std.constant" () {value = 1.0 : f32} : () -> complex<f32>
  return
}

// -----

func @complex_constant_wrong_element_types() {
  // expected-error @+1 {{requires attribute's element types ('f32', 'f32') to match the element type of the op's return type ('f64')}}
  %0 = constant [1.0 : f32, -1.0 : f32] : complex<f64>
  return
}

// -----

func @complex_constant_two_different_element_types() {
  // expected-error @+1 {{requires attribute's element types ('f32', 'f64') to match the element type of the op's return type ('f64')}}
  %0 = constant [1.0 : f32, -1.0 : f64] : complex<f64>
  return
}

// -----

func @return_i32_f32() -> (i32, f32) {
  %0 = constant 1 : i32
  %1 = constant 1. : f32
  return %0, %1 : i32, f32
}

func @call() {
  // expected-error @+3 {{op result type mismatch at index 0}}
  // expected-note @+2 {{op result types: 'f32', 'i32'}}
  // expected-note @+1 {{function result types: 'i32', 'f32'}}
  %0:2 = call @return_i32_f32() : () -> (f32, i32)
  return
}
