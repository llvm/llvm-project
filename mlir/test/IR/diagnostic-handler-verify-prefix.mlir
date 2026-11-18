// RUN: not mlir-opt %s -verify-diagnostics 2>&1 | FileCheck %s

// A directive prefix in the middle of a word is not recognized, so the
// 'unexpected-error' below does not match and the error is reported.
// CHECK: error: unexpected error: operation being parsed with an unregistered dialect
"foo.bar"() : () -> () // unexpected-error {{unregistered dialect}}
