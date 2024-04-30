// RUN: mlir-opt %s --transform-interpreter --allow-unregistered-dialect --verify-diagnostics | FileCheck %s

// RUN: mlir-opt %s --transform-interpreter --allow-unregistered-dialect --verify-diagnostics \
// RUN:   --mlir-print-debuginfo | FileCheck %s --check-prefix=CHECK-LOC

func.func @nested_ops() {
  "test.qux"() ({
    // expected-error @below{{fail_to_verify is set}}
    "test.baz"() ({
      "test.bar"() : () -> ()
    }) : () -> ()
  }) : () -> ()
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    // CHECK-LABEL{LITERAL}: [[[ IR printer: START top-level ]]]
    // CHECK-NEXT:  module {
    // CHECK-LOC-LABEL{LITERAL}: [[[ IR printer: START top-level ]]]
    // CHECK-LOC-NEXT:  #{{.+}} = loc(
    // CHECK-LOC-NEXT:  module {
    transform.print {name = "START"}

    // CHECK{LITERAL}: [[[ IR printer: Local scope top-level ]]]
    // CHECK-NEXT:      module {
    // CHECK-LOC{LITERAL}: [[[ IR printer: Local scope top-level ]]]
    // CHECK-LOC-NEXT:      module {
    transform.print {name = "Local scope", use_local_scope}

    %baz = transform.structured.match ops{["test.baz"]} in %arg0 : (!transform.any_op) -> !transform.any_op

    // CHECK{LITERAL}: [[[ IR printer: ]]]
    // CHECK-NEXT:      "test.baz"() ({
    // CHECK-NEXT:        "test.bar"() : () -> ()
    // CHECK-NEXT:       }) : () -> ()
    transform.print %baz : !transform.any_op

    // CHECK{LITERAL}: [[[ IR printer: Baz ]]]
    // CHECK-NEXT:      "test.baz"() ({
    transform.print %baz {name = "Baz"} : !transform.any_op

    // CHECK{LITERAL}: [[[ IR printer: No region ]]]
    // CHECK-NEXT:      "test.baz"() ({...}) : () -> ()
    transform.print %baz {name = "No region", skip_regions} : !transform.any_op

    // CHECK{LITERAL}: [[[ IR printer: No verify ]]]
    // CHECK-NEXT:      "test.baz"() ({
    // CHECK-NEXT:        transform.test_dummy_payload_op  {fail_to_verify} : () -> ()
    transform.test_produce_invalid_ir %baz : !transform.any_op
    transform.print %baz {name = "No verify", assume_verified} : !transform.any_op

    // CHECK-LABEL{LITERAL}: [[[ IR printer: END top-level ]]]
    transform.print {name = "END"}
    transform.yield
  }
}
