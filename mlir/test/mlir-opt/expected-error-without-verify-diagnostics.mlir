// Check that expected diagnostic directives are ignored without -verify-diagnostics.

// RUN: mlir-opt %s 2>&1 | FileCheck %s

// CHECK-NOT: expected error

// expected-error @unknown {{an error that is never produced}}
module {}
