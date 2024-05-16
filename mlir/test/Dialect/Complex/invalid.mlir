// RUN: mlir-opt -split-input-file %s -verify-diagnostics

func.func @complex_constant_wrong_array_attribute_length() {
  // expected-error @+1 {{requires 'value' to be a complex constant, represented as array of two values}}
  %0 = complex.constant [1.0 : f32] : complex<f32>
  return
}

// -----

func.func @complex_constant_wrong_element_types() {
  // expected-error @+1 {{requires attribute's element types ('f32', 'f32') to match the element type of the op's return type ('f64')}}
  %0 = complex.constant [1.0 : f32, -1.0 : f32] : complex<f64>
  return
}

// -----

func.func @complex_constant_two_different_element_types() {
  // expected-error @+1 {{requires attribute's element types ('f32', 'f64') to match the element type of the op's return type ('f64')}}
  %0 = complex.constant [1.0 : f32, -1.0 : f64] : complex<f64>
  return
}

// -----

func.func @complex_bitcast_i64(%arg0 : i64) {
  // expected-error @+1 {{op requires that either input or output has a complex type}}
  %0 = complex.bitcast %arg0: i64 to f64
  return
}
