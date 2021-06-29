// RUN: mlir-opt %s -linalg-comprehensive-module-bufferize -split-input-file -verify-diagnostics

// -----

// expected-error @-3 {{expected callgraph to be free of circular dependencies}}

func @foo() {
  call @bar() : () -> ()
  return
}

func @bar() {
  call @foo() : () -> ()
  return
}
