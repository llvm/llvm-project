// RUN: mlir-opt %s -split-input-file -verify-diagnostics

wasmssa.func @local_set_err(%arg0: !wasmssa<local ref to i32>) -> i64 {
  %0 = wasmssa.const 3 : i64
  // expected-error@+1 {{input type and result type of local.set do not match}}
  wasmssa.local_set %arg0 : ref to i32 to %0 : i64
  wasmssa.return %0 : i64
}

// -----

wasmssa.func @local_tee_err(%arg0: !wasmssa<local ref to i32>) -> i32 {
  %0 = wasmssa.const 3 : i64
  // expected-error@+1 {{input type and output type of local.tee do not match}}
  %1 = wasmssa.local_tee %arg0 :  ref to i32 to %0 : i64
  wasmssa.return %1 : i32
}
