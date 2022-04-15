// RUN: mlir-opt -split-input-file %s -verify-diagnostics

func @unsupported_attribute() {
  // expected-error @+1 {{invalid kind of attribute specified}}
  %0 = constant "" : index
  return
}

// -----

func private @return_i32_f32() -> (i32, f32)

func @call() {
  // expected-error @+3 {{op result type mismatch at index 0}}
  // expected-note @+2 {{op result types: 'f32', 'i32'}}
  // expected-note @+1 {{function result types: 'i32', 'f32'}}
  %0:2 = call @return_i32_f32() : () -> (f32, i32)
  return
}
