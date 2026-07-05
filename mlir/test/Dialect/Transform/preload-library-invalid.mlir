// RUN: mlir-opt %s \
// RUN:   -transform-preload-library=transform-library-paths=%p%{fs-sep}include%{fs-sep}test-interpreter-library-invalid \
// RUN:   -transform-interpreter=entry-point=private_helper \
// RUN:   -verify-diagnostics

// This test checks if the preload mechanism fails gracefully when passed an
// invalid transform file.
