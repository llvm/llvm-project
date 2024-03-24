// This test just needs to parse. Note that the diagnostic message below will
// be produced in *another* multi-file test, do *not* -verify-diagnostics here.
// RUN: mlir-opt %s

// RUN: mlir-transform-opt %s --transform-library=%p/external-def.mlir | FileCheck %s

module attributes {transform.with_named_sequence} {
  // The definition should not be printed here.
  // CHECK: @external_def
  // CHECK-NOT: transform.print
  transform.named_sequence private @external_def(%root: !transform.any_op {transform.readonly})

  transform.named_sequence private @__transform_main(%root: !transform.any_op) {
    // expected-error @below {{unresolved external named sequence}}
    transform.include @external_def failures(propagate) (%root) : (!transform.any_op) -> ()
    transform.yield
  }
}
