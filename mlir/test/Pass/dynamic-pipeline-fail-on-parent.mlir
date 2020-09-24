// RUN: mlir-opt %s -pass-pipeline='module(test-dynamic-pipeline{op-name=inner_mod1 run-on-parent=1 dynamic-pipeline=test-patterns})'  -split-input-file -verify-diagnostics

// Verify that we fail to schedule a dynamic pipeline on the parent operation.

// expected-error @+1 {{'module' op Trying to schedule a dynamic pipeline on an operation that isn't nested under the current operation}}
module {
module @inner_mod1 {
  "test.symbol"() {sym_name = "foo"} : () -> ()
  func @bar()
}
}
