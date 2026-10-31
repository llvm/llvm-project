// Test that an unmatched `expected-*` directive using the `@unknown` location
// specifier emits a diagnostic instead of crashing, and only once even though
// mlir-opt verifies twice (explicitly and in the handler's destructor).
// See https://github.com/llvm/llvm-project/issues/163343

// RUN: not mlir-opt --verify-diagnostics %s 2>&1 \
// RUN:   | FileCheck %s --implicit-check-not=error:

// CHECK: error: expected warning "some warning that is never produced" was not produced

// expected-warning @unknown {{some warning that is never produced}}
