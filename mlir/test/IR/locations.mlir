// RUN: mlir-opt -allow-unregistered-dialect %s -mlir-print-debuginfo -mlir-print-local-scope | FileCheck %s
// RUN: mlir-opt -allow-unregistered-dialect %s -mlir-print-debuginfo | FileCheck %s --check-prefix=CHECK-ALIAS
// This test verifies that debug locations are round-trippable.

#set0 = affine_set<(d0) : (1 == 0)>

// CHECK-LABEL: func @inline_notation
func @inline_notation() -> i32 {
  // CHECK: -> i32 loc("foo")
  %1 = "foo"() : () -> i32 loc("foo")

  // CHECK: constant 4 : index loc(callsite("foo" at "mysource.cc":10:8))
  %2 = constant 4 : index loc(callsite("foo" at "mysource.cc":10:8))

  // CHECK: } loc(fused["foo", "mysource.cc":10:8])
  affine.for %i0 = 0 to 8 {
  } loc(fused["foo", "mysource.cc":10:8])

  // CHECK: } loc(fused<"myPass">["foo", "foo2"])
  affine.if #set0(%2) {
  } loc(fused<"myPass">["foo", "foo2"])

  // CHECK: return %0 : i32 loc(unknown)
  return %1 : i32 loc(unknown)
}

// CHECK-LABEL: func private @loc_attr(i1 {foo.loc_attr = loc(callsite("foo" at "mysource.cc":10:8))})
func private @loc_attr(i1 {foo.loc_attr = loc(callsite("foo" at "mysource.cc":10:8))})

// CHECK-ALIAS: "foo.op"() : () -> () loc(#[[LOC:.*]])
"foo.op"() : () -> () loc(#loc)

// CHECK-ALIAS: #[[LOC]] = loc("out_of_line_location")
#loc = loc("out_of_line_location")
