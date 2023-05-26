// RUN: mlir-opt %s --pass-pipeline="builtin.module(test-transform-dialect-interpreter{transform-library-file-name=%p/test-interpreter-external-symbol-def.mlir})" \
// RUN:             --verify-diagnostics --split-input-file | FileCheck %s

// RUN: mlir-opt %s --pass-pipeline="builtin.module(test-transform-dialect-interpreter{transform-library-file-name=%p/test-interpreter-external-symbol-def.mlir}, test-transform-dialect-interpreter)" \
// RUN:             --verify-diagnostics --split-input-file | FileCheck %s

// RUN: mlir-opt %s --pass-pipeline="builtin.module(test-transform-dialect-interpreter{transform-library-file-name=%p/test-interpreter-external-symbol-def.mlir}, test-transform-dialect-interpreter{transform-library-file-name=%p/test-interpreter-external-symbol-def.mlir})" \
// RUN:             --verify-diagnostics --split-input-file | FileCheck %s

// The definition of the @foo named sequence is provided in another file. It
// will be included because of the pass option. Repeated application of the
// same pass, with or without the library option, should not be a problem.
// Note that the same diagnostic produced twice at the same location only
// needs to be matched once.

// expected-remark @below {{message}}
// expected-remark @below {{unannotated}}
module attributes {transform.with_named_sequence} {
  // CHECK: transform.named_sequence @foo
  // CHECK: test_print_remark_at_operand %{{.*}}, "message"
  transform.named_sequence private @foo(!transform.any_op {transform.readonly})

  // CHECK: transform.named_sequence @unannotated
  // CHECK: test_print_remark_at_operand %{{.*}}, "unannotated"
  transform.named_sequence private @unannotated(!transform.any_op {transform.readonly})

  transform.sequence failures(propagate) {
  ^bb0(%arg0: !transform.any_op):
    include @foo failures(propagate) (%arg0) : (!transform.any_op) -> ()
    include @unannotated failures(propagate) (%arg0) : (!transform.any_op) -> ()
  }
}
