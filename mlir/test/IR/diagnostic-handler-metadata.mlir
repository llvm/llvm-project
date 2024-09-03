// RUN: mlir-opt %s -pass-pipeline="builtin.module(func.func(test-diagnostic-metadata))" -verify-diagnostics -o - 2>&1 | FileCheck %s
// COM: This test verifies that diagnostic handler can filter the diagnostic based on its metadata
// COM: whether to emit the errors.

// CHECK-LABEL: Test 'test'
func.func @test() {
  // expected-error @+1 {{test diagnostic metadata}}
  "test.emit_error"() {
    // CHECK: attr = "emit_error"
    attr = "emit_error"
  } : () -> ()

  "test.do_not_emit_error"() {
    // CHECK: attr = "do_not_emit_error"
    attr = "do_not_emit_error"
  } : () -> ()

  return
}
