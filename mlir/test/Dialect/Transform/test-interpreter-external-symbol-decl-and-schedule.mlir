// RUN: mlir-opt %s --pass-pipeline="builtin.module(test-transform-dialect-interpreter{transform-file-name=%p/test-interpreter-external-symbol-decl.mlir transform-library-paths=%p/test-interpreter-library/definitions-self-contained.mlir})" \
// RUN:             --verify-diagnostics

// RUN: mlir-opt %s --pass-pipeline="builtin.module(test-transform-dialect-interpreter{transform-file-name=%p/test-interpreter-external-symbol-decl.mlir transform-library-paths=%p/test-interpreter-library/definitions-self-contained.mlir}, test-transform-dialect-interpreter{transform-file-name=%p/test-interpreter-external-symbol-decl.mlir transform-library-paths=%p/test-interpreter-library/definitions-self-contained.mlir})" \
// RUN:             --verify-diagnostics

// The external transform script has a declaration to the named sequence @foo,
// the definition of which is provided in another file. Repeated application
// of the same pass should not be a problem. Note that the same diagnostic
// produced twice at the same location only needs to be matched once.

// expected-remark @below {{message}}
// expected-remark @below {{unannotated}}
// expected-remark @below {{internal colliding (without suffix)}}
// expected-remark @below {{internal colliding_0}}
// expected-remark @below {{internal colliding_1}}
// expected-remark @below {{internal colliding_3}}
// expected-remark @below {{internal colliding_4}}
// expected-remark @below {{internal colliding_5}}
module {}
