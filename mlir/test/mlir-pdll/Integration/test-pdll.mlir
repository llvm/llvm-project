// RUN: mlir-opt %s -test-pdll-pass | FileCheck %s

// CHECK-LABEL: func @simpleTest
func @simpleTest() {
  // CHECK: test.success
  "test.simple"() : () -> ()
  return
}

// CHECK-LABEL: func @testImportedInterface
func @testImportedInterface() {
  // CHECK: test.non_cast
  // CHECK: test.success
  "test.non_cast"() : () -> ()
  "builtin.unrealized_conversion_cast"() : () -> (i1)
  return
}
