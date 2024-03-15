// RUN: mlir-opt %s --pass-pipeline="builtin.module(\
// RUN:                 transform-preload-library{transform-library-paths=%p%{fs-sep}include%{fs-sep}test-interpreter-external-source.mlir},\
// RUN:                 transform-interpreter)" \
// RUN:             --verify-diagnostics

// The schedule in the separate file emits remarks at the payload root.

// expected-remark @below {{outer}}
// expected-remark @below {{inner}}
module {}
