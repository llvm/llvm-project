// RUN: mlir-translate -verify-diagnostics -split-input-file -mlir-to-llvmir %s

// -----

llvm.func @redux_sync_i32_with_abs(%value: i32, %offset: i32) {
  // expected-error@+1 {{abs attribute is supported only for f32 type}}
  %res = nvvm.redux.sync add %value, %offset {abs = true}: i32 -> i32
  llvm.return
}

// -----

llvm.func @redux_sync_i32_with_nan(%value: i32, %offset: i32) {
  // expected-error@+1 {{nan attribute is supported only for f32 type}}
  %res = nvvm.redux.sync add %value, %offset {nan = true}: i32 -> i32
  llvm.return
}

// -----

llvm.func @redux_sync_f32_with_invalid_kind_add(%value: f32, %offset: i32) {
  // expected-error@+1 {{'add' redux kind unsupported with 'f32' type. Only supported type is 'i32'.}}
  %res = nvvm.redux.sync add %value, %offset: f32 -> f32
  llvm.return
}

// -----

llvm.func @redux_sync_f32_with_invalid_kind_and(%value: f32, %offset: i32) {
  // expected-error@+1 {{'and' redux kind unsupported with 'f32' type. Only supported type is 'i32'.}}
  %res = nvvm.redux.sync and %value, %offset: f32 -> f32
  llvm.return
}

// -----

llvm.func @redux_sync_i32_with_invalid_kind_fmin(%value: i32, %offset: i32) {
  // expected-error@+1 {{'fmin' redux kind unsupported with 'i32' type. Only supported type is 'f32'.}}
  %res = nvvm.redux.sync fmin %value, %offset: i32 -> i32
  llvm.return
}

// -----

llvm.func @redux_sync_non_matching_types(%value: i32, %offset: i32) {
  // expected-error@+1 {{failed to verify that all of {res, val} have same type}}
  %res = nvvm.redux.sync add %value, %offset: i32 -> f32
  llvm.return
}
