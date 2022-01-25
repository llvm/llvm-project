// RUN: mlir-opt -split-input-file %s -verify-diagnostics

func @unsupported_attribute() {
  // expected-error @+1 {{unsupported 'value' attribute: "" : index}}
  %0 = constant "" : index
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
