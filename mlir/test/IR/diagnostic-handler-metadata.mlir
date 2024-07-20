// RUN: mlir-opt %s -pass-pipeline="builtin.module(func.func(test-diagnostic-metadata))" -o - 2>&1 | FileCheck %s
// This test verifies that diagnostic handler can filter the diagnostic whether to emit the errors.

// CHECK-LABEL: Test 'test'
// CHECK-NEXT: 8:3: error: test diagnostic metadata
// CHECK-NOT: 13:3: error: test diagnostic metadata
func.func @test() {
  "test.test_attr0"() {
    // CHECK: attr = "emit_error"
    attr = "emit_error"
  } : () -> ()

  "test.test_attr1"() {
    // CHECK: attr = "do_not_emit_error"
    attr = "do_not_emit_error"
  } : () -> ()

  return
}
