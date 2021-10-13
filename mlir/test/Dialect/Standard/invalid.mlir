// RUN: mlir-opt -split-input-file %s -verify-diagnostics

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
  %0 = arith.constant 1 : i32
  %1 = arith.constant 1. : f32
  return %0, %1 : i32, f32
}

func @call() {
  // expected-error @+3 {{op result type mismatch at index 0}}
  // expected-note @+2 {{op result types: 'f32', 'i32'}}
  // expected-note @+1 {{function result types: 'i32', 'f32'}}
  %0:2 = call @return_i32_f32() : () -> (f32, i32)
  return
}
