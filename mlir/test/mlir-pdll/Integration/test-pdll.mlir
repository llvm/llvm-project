// RUN: mlir-opt %s -test-pdll-pass | FileCheck %s

// CHECK-LABEL: func @simpleTest
func @simpleTest() {
  // CHECK: test.success
  "test.simple"() : () -> ()
  return
}
