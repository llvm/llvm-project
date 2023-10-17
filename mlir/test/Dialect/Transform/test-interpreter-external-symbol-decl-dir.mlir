// RUN: mlir-opt %s --pass-pipeline="builtin.module(test-transform-dialect-interpreter{transform-library-paths=%p%{fs-sep}test-interpreter-library})" \
// RUN:             --verify-diagnostics --split-input-file | FileCheck %s

// RUN: mlir-opt %s --pass-pipeline="builtin.module(test-transform-dialect-interpreter{transform-library-paths=%p%{fs-sep}test-interpreter-library/definitions-self-contained.mlir,%p%{fs-sep}test-interpreter-library/definitions-with-unresolved.mlir})" \
// RUN:             --verify-diagnostics --split-input-file | FileCheck %s

// RUN: mlir-opt %s --pass-pipeline="builtin.module(test-transform-dialect-interpreter{transform-library-paths=%p%{fs-sep}test-interpreter-library}, test-transform-dialect-interpreter)" \
// RUN:             --verify-diagnostics --split-input-file | FileCheck %s

// The definition of the @foo named sequence is provided in another file. It
// will be included because of the pass option. Repeated application of the
// same pass, with or without the library option, should not be a problem.
// Note that the same diagnostic produced twice at the same location only
// needs to be matched once.

// expected-remark @below {{message}}
module attributes {transform.with_named_sequence} {
  // CHECK: transform.named_sequence @print_message
  transform.named_sequence @print_message(%arg0: !transform.any_op {transform.readonly})

  transform.named_sequence @reference_other_module(!transform.any_op {transform.readonly})

  transform.sequence failures(propagate) {
  ^bb0(%arg0: !transform.any_op):
    include @print_message failures(propagate) (%arg0) : (!transform.any_op) -> ()
    include @reference_other_module failures(propagate) (%arg0) : (!transform.any_op) -> ()
  }
}
