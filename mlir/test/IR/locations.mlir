// RUN: mlir-opt -allow-unregistered-dialect %s -mlir-print-debuginfo -mlir-print-local-scope | FileCheck %s
// RUN: mlir-opt -allow-unregistered-dialect %s -mlir-print-debuginfo | FileCheck %s --check-prefix=CHECK-ALIAS
// RUN: mlir-opt -allow-unregistered-dialect %s -mlir-print-debuginfo | mlir-opt -allow-unregistered-dialect -mlir-print-debuginfo | FileCheck %s --check-prefix=CHECK-ALIAS
// This test verifies that debug locations are round-trippable.

#set0 = affine_set<(d0) : (1 == 0)>

// CHECK-LABEL: func @inline_notation
func.func @inline_notation() -> i32 {
  // CHECK: -> i32 loc("foo")
  %1 = "foo"() : () -> i32 loc("foo")

  // CHECK: arith.constant 4 : index loc(callsite("foo" at "mysource.cc":10:8))
  %2 = arith.constant 4 : index loc(callsite("foo" at "mysource.cc":10:8))

  // CHECK: affine.for %arg0 loc("IVlocation") = 0 to 8 {
  // CHECK: } loc(fused["foo", "mysource.cc":10:8])
  affine.for %i0 loc("IVlocation") = 0 to 8 {
  } loc(fused["foo", "mysource.cc":10:8])

  // CHECK: } loc(fused<"myPass">["foo", "foo2"])
  affine.if #set0(%2) {
  } loc(fused<"myPass">["foo", "foo2"])

  // CHECK: } loc(fused<"myPass">["foo"])
  affine.if #set0(%2) {
  } loc(fused<"myPass">["foo"])

  // CHECK: return %0 : i32 loc(unknown)
  return %1 : i32 loc(unknown)
}

// CHECK-LABEL: func private @loc_attr(i1 {foo.loc_attr = loc(callsite("foo" at "mysource.cc":10:8))})
func.func private @loc_attr(i1 {foo.loc_attr = loc(callsite("foo" at "mysource.cc":10:8))})

  // Check that locations get properly escaped.
// CHECK-LABEL: func @escape_strings()
func.func @escape_strings() {
  // CHECK: loc("escaped\0A")
  "foo"() : () -> () loc("escaped\n")

  // CHECK: loc("escaped\0A")
  "foo"() : () -> () loc("escaped\0A")

  // CHECK: loc("escaped\0A":0:0)
  "foo"() : () -> () loc("escaped\n":0:0)
  return
}

// CHECK-ALIAS: "foo.op"() : () -> () loc(#[[LOC:.*]])
"foo.op"() : () -> () loc(#loc)

// CHECK-LABEL: func @argLocs(
// CHECK-SAME:  %arg0: i32 loc({{.*}}locations.mlir":[[# @LINE+1]]:20),
func.func @argLocs(%x: i32,
// CHECK-SAME:  %arg1: i64 loc("hotdog")
              %y: i64 loc("hotdog")) {
  return
}

// CHECK-LABEL: "foo.unknown_op_with_bbargs"()
// CHECK-ALIAS: "foo.unknown_op_with_bbargs"()
"foo.unknown_op_with_bbargs"() ({
// CHECK-NEXT: ^bb0(%arg0: i32 loc({{.*}}locations.mlir":[[# @LINE+2]]:7),
// CHECK-ALIAS-NEXT: ^bb0(%arg0: i32 loc({{.*}}locations.mlir":[[# @LINE+1]]:7),
 ^bb0(%x: i32,
// CHECK-SAME: %arg1: i32 loc("cheetos"),
// CHECK-ALIAS-SAME: %arg1: i32 loc("cheetos"),
      %y: i32 loc("cheetos"),
// CHECK-SAME: %arg2: i32 loc("out_of_line_location2")):
// CHECK-ALIAS-SAME: %arg2: i32 loc("out_of_line_location2")):
      %z: i32 loc("out_of_line_location2")):
    %1 = arith.addi %x, %y : i32
    "foo.yield"(%1) : (i32) -> ()
  }) : () -> ()

// CHECK-LABEL: func @location_name_child_is_name
func.func @location_name_child_is_name() {
  // CHECK: "foo"("foo")
  return loc("foo"("foo"))
}

// CHECK-ALIAS: #[[LOC]] = loc("out_of_line_location")
#loc = loc("out_of_line_location")
