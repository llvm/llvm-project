// RUN: mlir-opt %s -test-func-set-type -split-input-file | FileCheck %s

// It's currently not possible to have an attribute with a function type due to
// parser ambiguity. So instead we reference a function declaration to take the
// type from.

// -----

// Test case: The setType call needs to erase some arg attrs.

// CHECK: func private @erase_arg(f32 {test.A})
// CHECK-NOT: attributes{{.*arg[0-9]}}
func private @t(f32)
func private @erase_arg(%arg0: f32 {test.A}, %arg1: f32 {test.B})
attributes {test.set_type_from = @t}

// -----

// Test case: The setType call needs to erase some result attrs.

// CHECK: func private @erase_result() -> (f32 {test.A})
// CHECK-NOT: attributes{{.*result[0-9]}}
func private @t() -> (f32)
func private @erase_result() -> (f32 {test.A}, f32 {test.B})
attributes {test.set_type_from = @t}
