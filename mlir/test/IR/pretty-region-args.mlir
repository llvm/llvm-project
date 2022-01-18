// RUN: mlir-opt -allow-unregistered-dialect %s | FileCheck %s

// CHECK-LABEL: func @custom_region_names
func @custom_region_names() -> () {
  "test.polyfor"() ({
  ^bb0(%arg0: index, %arg1: index, %arg2: index):
    "foo"() : () -> ()
  }) { arg_names = ["i", "j", "k"] } : () -> ()
  // CHECK: test.polyfor
  // CHECK-NEXT: ^bb{{.*}}(%i: index, %j: index, %k: index):
  return
}

// CHECK-LABEL: func @weird_names
// Make sure the asmprinter handles weird names correctly.
func @weird_names() -> () {
  "test.polyfor"() ({
  ^bb0(%arg0: i32, %arg1: i32, %arg2: index):
    "foo"() : () -> i32
  }) { arg_names = ["a .^x", "0"] } : () -> ()
  // CHECK: test.polyfor
  // CHECK-NEXT: ^bb{{.*}}(%a_.5Ex: i32, %_0: i32, %arg0: index):
  // CHECK-NEXT: %0 = "foo"()
  return
}

