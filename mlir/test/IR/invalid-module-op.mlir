// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// -----

func @module_op() {
  // expected-error@+1 {{Operations with a 'SymbolTable' must have exactly one block}}
  module {
  ^bb1:
    "module_terminator"() : () -> ()
  ^bb2:
    "module_terminator"() : () -> ()
  }
  return
}

// -----

func @module_op() {
  // expected-error@+1 {{region should have no arguments}}
  module {
  ^bb1(%arg: i32):
    "module_terminator"() : () -> ()
  }
  return
}

// -----

func @module_op() {
  // expected-error@below {{expects regions to end with 'module_terminator'}}
  // expected-note@below {{the absence of terminator implies 'module_terminator'}}
  module {
    return
  }
  return
}

// -----

func @module_op() {
  // expected-error@+1 {{expects parent op 'module'}}
  "module_terminator"() : () -> ()
}

// -----

// expected-error@+1 {{can only contain attributes with dialect-prefixed names}}
module attributes {attr} {
}

