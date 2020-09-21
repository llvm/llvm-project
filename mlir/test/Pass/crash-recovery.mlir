// RUN: mlir-opt %s -pass-pipeline='func(test-function-pass, test-pass-crash)' -pass-pipeline-crash-reproducer=%t -verify-diagnostics
// RUN: cat %t | FileCheck -check-prefix=REPRO %s
// RUN: mlir-opt %s -pass-pipeline='func(test-function-pass, test-pass-crash)' -pass-pipeline-crash-reproducer=%t -verify-diagnostics -pass-pipeline-local-reproducer
// RUN: cat %t | FileCheck -check-prefix=REPRO_LOCAL %s

// Check that we correctly handle verifiers passes with local reproducer, this use to crash.
// RUN: mlir-opt %s -test-function-pass -test-function-pass  -test-module-pass -pass-pipeline-crash-reproducer=%t -pass-pipeline-local-reproducer

// expected-error@+1 {{A failure has been detected while processing the MLIR module}}
module {
  func @foo() {
    return
  }
}

// REPRO: configuration: -pass-pipeline='func(test-function-pass, test-pass-crash)'

// REPRO: module
// REPRO: func @foo() {
// REPRO-NEXT: return

// REPRO_LOCAL: configuration: -pass-pipeline='func(test-pass-crash)'

// REPRO_LOCAL: module
// REPRO_LOCAL: func @foo() {
// REPRO_LOCAL-NEXT: return
