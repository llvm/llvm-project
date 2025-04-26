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

// -----

/// Test `from_ptr` verifiers.
func.func @invalid_from_ptr(%v: !ptr.ptr<#ptr.generic_space>) {
  // expected-error@+1 {{expected either a metadata argument or the `trivial_metadata` flag to be set}}
  %r = ptr.from_ptr %v : !ptr.ptr<#ptr.generic_space> -> memref<f32, #ptr.generic_space>
  return
}

// -----

func.func @invalid_from_ptr(%v: !ptr.ptr<#ptr.generic_space>, %m: !ptr.ptr_metadata<memref<f32, #ptr.generic_space>>) {
  // expected-error@+1 {{expected either a metadata argument or the `trivial_metadata` flag, not both}}
  %r = ptr.from_ptr %v metadata %m trivial_metadata : !ptr.ptr<#ptr.generic_space> -> memref<f32, #ptr.generic_space> 
  return
}
