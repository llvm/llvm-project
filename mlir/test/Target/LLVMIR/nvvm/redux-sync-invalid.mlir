// RUN: mlir-translate -verify-diagnostics -split-input-file -mlir-to-llvmir %s

// -----

llvm.func @redux_sync_i32_with_abs(%value: i32, %offset: i32) {
  // expected-error@+1 {{'nvvm.redux.sync' op abs attribute is supported only for f32 type}}
  %res = nvvm.redux.sync add %value, %offset {abs = true}: i32 -> i32
  llvm.return
}

// -----

llvm.func @redux_sync_i32_with_nan(%value: i32, %offset: i32) {
  // expected-error@+1 {{'nvvm.redux.sync' op nan attribute is supported only for f32 type}}
  %res = nvvm.redux.sync add %value, %offset {nan = true}: i32 -> i32
  llvm.return
}

// -----

llvm.func @redux_sync_f32_with_invalid_kind(%value: f32, %offset: i32) {
  // expected-error@+1 {{'nvvm.redux.sync' op only fmin and fmax redux kinds are supported for f32 type}}
  %res = nvvm.redux.sync add %value, %offset: f32 -> f32
  llvm.return
}

// -----

llvm.func @redux_sync_i32_with_invalid_kind(%value: i32, %offset: i32) {
  // expected-error@+1 {{'nvvm.redux.sync' op fmin and fmax redux kind must be used with f32 type}}
  %res = nvvm.redux.sync fmin %value, %offset: i32 -> i32
  llvm.return
}
