// RUN: mlir-opt %s -split-input-file -verify-diagnostics


wasmssa.func @extend_low_64() -> i32 {
  %0 = wasmssa.const 10 : i32
  // expected-error@+1 {{extend op can only take 8, 16 or 32 bits. Got 64}}
  %1 = wasmssa.extend 64 low bits from %0: i32
  wasmssa.return %1 : i32
}

// -----

wasmssa.func @extend_too_much() -> i32 {
  %0 = wasmssa.const 10 : i32
  // expected-error@+1 {{trying to extend the 32 low bits from a 'i32' value is illegal}}
  %1 = wasmssa.extend 32 low bits from %0: i32
  wasmssa.return %1 : i32
}
