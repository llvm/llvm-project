// RUN: mlir-opt %s -test-pdll-pass | FileCheck %s

// CHECK-LABEL: func @simpleTest
func.func @simpleTest() {
  // CHECK: test.success
  "test.simple"() : () -> ()
  return
}

// CHECK-LABEL: func @testImportedInterface
func.func @testImportedInterface() -> i1 {
  // CHECK: test.non_cast
  // CHECK: test.success
  "test.non_cast"() : () -> ()
  %value = "builtin.unrealized_conversion_cast"() : () -> (i1)
  return %value : i1
}

// CHECK-LABEL: func @testWithConstraint
func.func @testWithConstraint(%a: i32) {
    // CHECK: test.success
    %b = "test.op_a"(%a) { attr = 0 : i32} : (i32) -> (i32)
    return
}
