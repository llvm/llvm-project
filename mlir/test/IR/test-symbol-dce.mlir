// RUN: mlir-opt %s -symbol-dce -split-input-file -verify-diagnostics | FileCheck %s
// RUN: mlir-opt %s -pass-pipeline="module(symbol-dce)" -split-input-file | FileCheck %s --check-prefix=NESTED

// Check that trivially dead and trivially live non-nested cases are handled.

// CHECK-LABEL: module attributes {test.simple}
module attributes {test.simple} {
  // CHECK-NOT: func @dead_private_function
  func @dead_private_function() attributes { sym_visibility = "nested" }

  // CHECK-NOT: func @dead_nested_function
  func @dead_nested_function() attributes { sym_visibility = "nested" }

  // CHECK: func @live_private_function
  func @live_private_function() attributes { sym_visibility = "nested" }

  // CHECK: func @live_nested_function
  func @live_nested_function() attributes { sym_visibility = "nested" }

  // CHECK: func @public_function
  func @public_function() {
    "foo.return"() {uses = [@live_private_function, @live_nested_function]} : () -> ()
  }

  // CHECK: func @public_function_explicit
  func @public_function_explicit() attributes { sym_visibility = "public" }
}

// -----

// Check that we don't DCE nested symbols if they are used.
// CHECK-LABEL: module attributes {test.nested}
module attributes {test.nested} {
  // CHECK: module @public_module
  module @public_module {
    // CHECK-NOT: func @dead_nested_function
    func @dead_nested_function() attributes { sym_visibility = "nested" }

    // CHECK: func @private_function
    func @private_function() attributes { sym_visibility = "private" }

    // CHECK: func @nested_function
    func @nested_function() attributes { sym_visibility = "nested" } {
      "foo.return"() {uses = [@private_function]} : () -> ()
    }
  }

  "live.user"() {uses = [@public_module::@nested_function]} : () -> ()
}

// -----

// Check that we don't DCE symbols if we can't prove that the top-level symbol
// table that we are running on is hidden from above.
// NESTED-LABEL: module attributes {test.no_dce_non_hidden_parent}
module attributes {test.no_dce_non_hidden_parent} {
  // NESTED: module @public_module
  module @public_module {
    // NESTED: func @nested_function
    func @nested_function() attributes { sym_visibility = "nested" }
  }
  // NESTED: module @nested_module
  module @nested_module attributes { sym_visibility = "nested" } {
    // NESTED: func @nested_function
    func @nested_function() attributes { sym_visibility = "nested" }
  }

  // Only private modules can be assumed to be hidden.
  // NESTED: module @private_module
  module @private_module attributes { sym_visibility = "private" } {
    // NESTED-NOT: func @nested_function
    func @nested_function() attributes { sym_visibility = "nested" }
  }

  "live.user"() {uses = [@nested_module, @private_module]} : () -> ()
}

// -----

module {
  func @private_symbol() attributes { sym_visibility = "private" }

  // expected-error@+1 {{contains potentially unknown symbol table}}
  "foo.possibly_unknown_symbol_table"() ({
  }) : () -> ()
}

// -----

module {
  // expected-error@+1 {{unable to resolve reference to symbol}}
  "live.user"() {uses = [@unknown_symbol]} : () -> ()
}
