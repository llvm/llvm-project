// RUN: mlir-opt %s --pass-pipeline="builtin.module(test-transform-dialect-interpreter{transform-library-paths=%p%{fs-sep}include%{fs-sep}test-interpreter-external-symbol-def-invalid.mlir}, test-transform-dialect-interpreter)" \
// RUN:             --verify-diagnostics --split-input-file

// The definition of the @print_message named sequence is provided in another file. It
// will be included because of the pass option.

module attributes {transform.with_named_sequence} {
  // expected-error @below {{external definition has a mismatching signature}}
  transform.named_sequence private @print_message(!transform.op<"builtin.module"> {transform.readonly})

  // expected-error @below {{failed to merge library symbols into transform root}}
  transform.sequence failures(propagate) {
  ^bb0(%arg0: !transform.op<"builtin.module">):
    include @print_message failures(propagate) (%arg0) : (!transform.op<"builtin.module">) -> ()
  }
}

// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence private @undefined_sequence()

  transform.sequence failures(suppress) {
  ^bb0(%arg0: !transform.any_op):
    // expected-error @below {{unresolved external named sequence}}
    include @undefined_sequence failures(suppress) () : () -> ()
  }
}

// -----

module attributes {transform.with_named_sequence} {
  // expected-error @below {{external definition has mismatching consumption annotations for argument #0}}
  transform.named_sequence private @consuming(%arg0: !transform.any_op {transform.readonly})

  // expected-error @below {{failed to merge library symbols into transform root}}
  transform.sequence failures(suppress) {
  ^bb0(%arg0: !transform.any_op):
    include @consuming failures(suppress) (%arg0) : (!transform.any_op) -> ()
  }
}

// -----

module attributes {transform.with_named_sequence} {
  // expected-error @below {{doubly defined symbol @print_message}}
  transform.named_sequence @print_message(%arg0: !transform.any_op {transform.readonly}) {
    transform.test_print_remark_at_operand %arg0, "message" : !transform.any_op
    transform.yield
  }

  // expected-error @below {{failed to merge library symbols into transform root}}
  transform.sequence failures(suppress) {
  ^bb0(%arg0: !transform.any_op):
    include @print_message failures(propagate) (%arg0) : (!transform.any_op) -> ()
  }
}
