// RUN: mlir-opt %s -split-input-file -verify-diagnostics

wasmssa.func @f32_reinterpret_f32() -> f32 {
  %0 = wasmssa.const -1.000000e+00 : f32
  // expected-error@+1 {{reinterpret input and output type should be distinct}}
  %1 = wasmssa.reinterpret %0 : f32 as f32
  wasmssa.return %1 : f32
}

// -----

wasmssa.func @f64_reinterpret_f32() -> f32 {
  %0 = wasmssa.const -1.000000e+00 : f64
  // expected-error@+1 {{input type ('f64') and output type ('f32') have incompatible bit widths}}
  %1 = wasmssa.reinterpret %0 : f64 as f32
  wasmssa.return %1 : f32
}
