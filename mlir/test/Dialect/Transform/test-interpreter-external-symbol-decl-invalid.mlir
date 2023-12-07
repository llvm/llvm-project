// RUN: mlir-opt %s --pass-pipeline="builtin.module(test-transform-dialect-interpreter{transform-library-file-name=%p/test-interpreter-external-symbol-def.mlir}, test-transform-dialect-interpreter)" \
// RUN:             --verify-diagnostics --split-input-file

// The definition of the @foo named sequence is provided in another file. It
// will be included because of the pass option.

module attributes {transform.with_named_sequence} {
  // expected-error @below {{external definition has a mismatching signature}}
  transform.named_sequence private @foo(!transform.op<"builtin.module"> {transform.readonly})

  transform.sequence failures(propagate) {
  ^bb0(%arg0: !transform.op<"builtin.module">):
    include @foo failures(propagate) (%arg0) : (!transform.op<"builtin.module">) -> ()
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

  transform.sequence failures(suppress) {
  ^bb0(%arg0: !transform.any_op):
    include @consuming failures(suppress) (%arg0) : (!transform.any_op) -> ()
  }
}
