// RUN: mlir-opt -split-input-file -verify-diagnostics %s

/// Test `to_ptr` verifiers.
func.func @invalid_to_ptr(%v: memref<f32, 0>) {
  // expected-error@+1 {{expected the input and output to have the same memory space}}
  %r = ptr.to_ptr %v : memref<f32, 0> -> !ptr.ptr<#ptr.generic_space>
  return
}

// -----

func.func @invalid_to_ptr(%v: !ptr.ptr<#ptr.generic_space>) {
  // expected-error@+1 {{the input value cannot be of type `!ptr.ptr`}}
  %r = ptr.to_ptr %v : !ptr.ptr<#ptr.generic_space> -> !ptr.ptr<#ptr.generic_space>
  return
}
