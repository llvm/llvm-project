// RUN: cd %p && mlir-opt %s --pass-pipeline="builtin.module(test-transform-dialect-interpreter{transform-file-name=test-interpreter-external-source.mlir})" \
// RUN:               --verify-diagnostics

// The schedule in the separate file emits remarks at the payload root.

// expected-remark @below {{outer}}
// expected-remark @below {{inner}}
module {}
