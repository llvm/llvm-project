// RUN: mlir-opt %s -test-print-liveness -split-input-file -verify-diagnostics

// Unnamed modules do not implement SymbolOpInterface.
// expected-error-re @+1 {{trying to schedule pass '{{.*}}TestLivenessPass' on an unsupported operation}}
module {}

// -----

// Named modules implement SymbolOpInterface.
module @named_module {}
