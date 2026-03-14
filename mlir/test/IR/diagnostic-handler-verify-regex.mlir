// RUN: not mlir-opt -allow-unregistered-dialect %s -split-input-file -verify-diagnostics 2>&1 | FileCheck %s

// CHECK: found start of regex with no end '}}'
// expected-error-re {{{{}}

// -----

// CHECK: invalid regex: parentheses not balanced
// expected-error-re {{ {{(}} }}

// -----

func.func @foo() -> i32 {
  // expected-error-re@+1 {{'func.return' op has 0 operands, but enclosing function (@{{.*}}) returns 1}}
  return
}
