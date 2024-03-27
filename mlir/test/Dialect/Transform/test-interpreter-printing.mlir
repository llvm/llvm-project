// RUN: mlir-opt %s --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics | FileCheck %s

func.func @nested_ops() {
  "test.foo"() ({
    "test.foo"() ({
      "test.qux"() ({
        "test.baz"() ({
          "test.bar"() : () -> ()
        }) : () -> ()
      }) : () -> ()
    }) : () -> ()
  }) : () -> ()
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %baz = transform.structured.match ops{["test.baz"]} in %arg0 : (!transform.any_op) -> !transform.any_op

    // CHECK-LABEL{LITERAL}: [[[ IR printer: START top-level ]]]
    // CHECK-NEXT:  module {
    transform.print {name = "START"}
    // CHECK{LITERAL}: [[[ IR printer: ]]]
    // CHECK-NEXT:      "test.baz"() ({
    // CHECK-NEXT:        "test.bar"() : () -> ()
    // CHECK-NEXT:       }) : () -> ()
    transform.print %baz : !transform.any_op

    // CHECK{LITERAL}: [[[ IR printer: Baz ]]]
    // CHECK-NEXT:      "test.baz"() ({
    transform.print %baz {name = "Baz"} : !transform.any_op

    // CHECK{LITERAL}: [[[ IR printer: No verify ]]]
    // CHECK-NEXT:      "test.baz"() ({
    transform.print %baz {name = "No verify", assume_verified} : !transform.any_op

    // CHECK{LITERAL}: [[[ IR printer: Local scope ]]]
    // CHECK-NEXT:      "test.baz"() ({
    // CHECK-NEXT:        "test.bar"() : () -> ()
    transform.print %baz {name = "Local scope", use_local_scope} : !transform.any_op

    // CHECK{LITERAL}: [[[ IR printer: No region ]]]
    // CHECK-NEXT:      "test.baz"() ({...}) : () -> ()
    transform.print %baz {name = "No region", skip_regions} : !transform.any_op

    // CHECK-LABEL{LITERAL}: [[[ IR printer: END top-level ]]]
    transform.print {name = "END"}
    transform.yield
  }
}
