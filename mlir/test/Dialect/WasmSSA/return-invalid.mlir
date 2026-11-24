// RUN: mlir-opt %s -split-input-file -verify-diagnostics

wasmssa.func @return_type_mismatch() -> i32 {
  %0 = wasmssa.const 0.5 : f32
  // expected-error@+1 {{type of return operand #0 ('f32') doesn't match function result type ('i32')}}
  wasmssa.return %0 : f32
}

// -----

wasmssa.func @return_too_few_operands() -> i32 {
  // expected-error@+1 {{has 0 operands, but enclosing function returns 1}}
  wasmssa.return
}

// -----

wasmssa.func @return_too_many_operands() {
  %0 = wasmssa.const 1 : i32
  // expected-error@+1 {{has 1 operands, but enclosing function returns 0}}
  wasmssa.return %0 : i32
}

// -----

wasmssa.func @return_in_nested_region(%arg0 : !wasmssa<local ref to i32>) -> i64 {
  %cond = wasmssa.local_get %arg0 : ref to i32
  wasmssa.if %cond : {
    %c0 = wasmssa.const 1 : i32
    // expected-error@+1 {{type of return operand #0 ('i32') doesn't match function result type ('i64')}}
    wasmssa.return %c0 : i32
  } >^bb1
 ^bb1:
  %c1 = wasmssa.const 2 : i64
  wasmssa.return %c1 : i64
}
