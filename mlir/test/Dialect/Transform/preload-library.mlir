// RUN: mlir-opt %s \
// RUN:   -transform-preload-library=transform-library-paths=%p%{fs-sep}include%{fs-sep}test-interpreter-library \
// RUN:   -transform-interpreter=entry-point=private_helper \
// RUN:   -split-input-file -verify-diagnostics

// RUN: mlir-opt %s \
// RUN:   -transform-preload-library=transform-library-paths=%p%{fs-sep}include%{fs-sep}test-interpreter-library/definitions-self-contained.mlir \
// RUN:   -transform-preload-library=transform-library-paths=%p%{fs-sep}include%{fs-sep}test-interpreter-library/definitions-with-unresolved.mlir \
// RUN:   -transform-interpreter=entry-point=private_helper \
// RUN:   -split-input-file -verify-diagnostics

// RUN: mlir-opt %s \
// RUN:   -transform-preload-library=transform-library-paths=%p%{fs-sep}include%{fs-sep}test-interpreter-library/definitions-with-unresolved.mlir \
// RUN:   -transform-preload-library=transform-library-paths=%p%{fs-sep}include%{fs-sep}test-interpreter-library/definitions-self-contained.mlir \
// RUN:   -transform-interpreter=entry-point=private_helper \
// RUN:   -split-input-file -verify-diagnostics

// expected-remark @below {{message}}
module {}

// -----

// Note: no remark here since local entry point takes precedence.
module attributes { transform.with_named_sequence } {
  transform.named_sequence @private_helper(!transform.any_op {transform.readonly}) {
  ^bb0(%arg0: !transform.any_op):
    // expected-remark @below {{applying transformation}}
    transform.test_transform_op
    transform.yield
  }
}
