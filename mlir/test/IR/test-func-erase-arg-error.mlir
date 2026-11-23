// RUN: mlir-opt %s -test-func-erase-arg -split-input-file -verify-diagnostics

// Erasing arguments that still have uses must fail (see #203218).

// expected-error @below {{cannot erase argument 0 which still has uses}}
// expected-error @below {{cannot erase argument 1 which still has uses}}
func.func @f(%arg0: f32 {test.erase_this_arg},
             %arg1: f32 {test.erase_this_arg}) -> (f32, f32) {
  return %arg0, %arg1 : f32, f32
}

// -----

// An external function has no body, so erasure must succeed.
func.func private @ext(f32 {test.erase_this_arg})
