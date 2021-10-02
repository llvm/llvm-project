// RUN: mlir-opt  %s -test-opaque-loc -mlir-print-debuginfo -mlir-print-local-scope | FileCheck %s
// This test verifies that debug opaque locations can be printed.

#set0 = affine_set<(d0) : (1 == 0)>

// CHECK: MyLocation: 0: 'test.foo' op
// CHECK: nullptr: 'test.foo' op
// CHECK: MyLocation: 0: 'test.foo' op
// CHECK: MyLocation: 1: 'std.constant' op
// CHECK: nullptr: 'std.constant' op
// CHECK: MyLocation: 1: 'std.constant' op

// CHECK-LABEL: func @inline_notation
func @inline_notation() -> i32 {
  // CHECK: -> i32 loc("foo")
  // CHECK: -> i32 loc("foo")
  // CHECK: -> i32 loc(unknown)
  %1 = "test.foo"() : () -> i32 loc("foo")

  // CHECK: constant 4 : index loc(callsite("foo" at "mysource.cc":10:8))
  // CHECK: constant 4 : index loc(callsite("foo" at "mysource.cc":10:8))
  // CHECK: constant 4 : index loc(unknown)
  %2 = constant 4 : index loc(callsite("foo" at "mysource.cc":10:8))

  // CHECK: } loc(unknown)
  affine.for %i0 = 0 to 8 {
  } loc(fused["foo", "mysource.cc":10:8])

  // CHECK: } loc(unknown)
  affine.for %i0 = 0 to 8 {
  } loc(fused["foo", "mysource.cc":10:8, callsite("foo" at "mysource.cc":10:8)])

  // CHECK: return %{{.*}} : i32 loc(unknown)
  return %1 : i32 loc(unknown)
}
