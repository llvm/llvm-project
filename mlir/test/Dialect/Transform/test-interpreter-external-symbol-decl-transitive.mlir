// RUN: mlir-opt %s --pass-pipeline="builtin.module(test-transform-dialect-interpreter{transform-library-file-name=%p/test-interpreter-external-symbol-def.mlir})" \
// RUN:             --verify-diagnostics --split-input-file | FileCheck %s

// RUN: mlir-opt %s --pass-pipeline="builtin.module(test-transform-dialect-interpreter{transform-library-file-name=%p/test-interpreter-external-symbol-def.mlir}, test-transform-dialect-interpreter)" \
// RUN:             --verify-diagnostics --split-input-file | FileCheck %s

// RUN: mlir-opt %s --pass-pipeline="builtin.module(test-transform-dialect-interpreter{transform-library-file-name=%p/test-interpreter-external-symbol-def.mlir}, test-transform-dialect-interpreter{transform-library-file-name=%p/test-interpreter-external-symbol-def.mlir})" \
// RUN:             --verify-diagnostics --split-input-file | FileCheck %s

// The definition of the @bar named sequence is provided in another file. It
// will be included because of the pass option. That sequence uses another named
// sequence @foo, which should be made available here. Repeated application of
// the same pass, with or without the library option, should not be a problem.
// Note that the same diagnostic produced twice at the same location only
// needs to be matched once.

// expected-remark @below {{message}}
module attributes {transform.with_named_sequence} {
  // CHECK-DAG: transform.named_sequence @foo
  // CHECK-DAG: transform.named_sequence @bar
  transform.named_sequence private @bar(!transform.any_op {transform.readonly})

  transform.sequence failures(propagate) {
  ^bb0(%arg0: !transform.any_op):
    include @bar failures(propagate) (%arg0) : (!transform.any_op) -> ()
  }
}
