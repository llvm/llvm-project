// RUN: mlir-opt -allow-unregistered-dialect %s | FileCheck %s
// RUN: mlir-opt -allow-unregistered-dialect -mlir-print-op-generic -mlir-print-debuginfo -mlir-print-local-scope %s | FileCheck %s --check-prefix=CHECK-GENERIC

// CHECK-LABEL: func @wrapping_op
// CHECK-GENERIC: "func.func"
// CHECK-GENERIC-SAME: sym_name = "wrapping_op"
func.func @wrapping_op(%arg0 : i32, %arg1 : f32) -> (i3, i2, i1) {
// CHECK: %some_NameLoc, %some_NameLoc_0, %some_NameLoc_1 = test.wrapping_region wraps "some.op"(%arg1, %arg0) {test.attr = "attr"} : (f32, i32) -> (i1, i2, i3)
// CHECK-GENERIC: "test.wrapping_region"() ({
// CHECK-GENERIC:   %some_NameLoc_2, %some_NameLoc_3, %some_NameLoc_4 = "some.op"(%arg1, %arg0) {test.attr = "attr"} : (f32, i32) -> (i1, i2, i3) loc("some_NameLoc")
// CHECK-GENERIC:   "test.return"(%some_NameLoc_2, %some_NameLoc_3, %some_NameLoc_4) : (i1, i2, i3) -> () loc("some_NameLoc")
// CHECK-GENERIC: }) : () -> (i1, i2, i3) loc("some_NameLoc")
  %res:3 = test.wrapping_region wraps "some.op"(%arg1, %arg0) { test.attr = "attr" } : (f32, i32) -> (i1, i2, i3) loc("some_NameLoc")
  return %res#2, %res#1, %res#0 : i3, i2, i1
}
