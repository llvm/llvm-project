// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// -----

func @module_op() {
  // expected-error@+1 {{'builtin.module' op expects region #0 to have 0 or 1 blocks}}
  builtin.module {
  ^bb1:
    "test.dummy"() : () -> ()
  ^bb2:
    "test.dummy"() : () -> ()
  }
  return
}

// -----

func @module_op() {
  // expected-error@+1 {{region should have no arguments}}
  builtin.module {
  ^bb1(%arg: i32):
  }
  return
}

// -----

// expected-error@+1 {{can only contain attributes with dialect-prefixed names}}
module attributes {attr} {
}

