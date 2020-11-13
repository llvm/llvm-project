// RUN: mlir-opt -allow-unregistered-dialect %s -inline -split-input-file | FileCheck %s

// This file tests the callgraph dead code elimination performed by the inliner.

// Function is already dead.
// CHECK-NOT: func private @dead_function
func private @dead_function() {
  return
}

// Function becomes dead after inlining.
// CHECK-NOT: func private @dead_function_b
func @dead_function_b() {
  return
}

// CHECK: func @live_function()
func @live_function() {
  call @dead_function_b() : () -> ()
  return
}

// Same as above, but a transitive example.

// CHECK: func @live_function_b
func @live_function_b() {
  return
}
// CHECK-NOT: func private @dead_function_c
func private @dead_function_c() {
  call @live_function_b() : () -> ()
  return
}
// CHECK-NOT: func private @dead_function_d
func private @dead_function_d() {
  call @dead_function_c() : () -> ()
  call @dead_function_c() : () -> ()
  return
}
// CHECK: func @live_function_c
func @live_function_c() {
  call @dead_function_c() : () -> ()
  call @dead_function_d() : () -> ()
  return
}

// Function is referenced by non-callable top-level user.
// CHECK: func private @live_function_d
func private @live_function_d() {
  return
}

"live.user"() {use = @live_function_d} : () -> ()

// -----

// This test checks that the inliner can properly handle the deletion of
// functions in different SCCs that are referenced by calls materialized during
// canonicalization.
// CHECK: func @live_function_e
func @live_function_e() {
  call @dead_function_e() : () -> ()
  return
}
// CHECK-NOT: func @dead_function_e
func private @dead_function_e() -> () {
  "test.fold_to_call_op"() {callee=@dead_function_f} : () -> ()
  return
}
// CHECK-NOT: func private @dead_function_f
func private @dead_function_f() {
  return
}
