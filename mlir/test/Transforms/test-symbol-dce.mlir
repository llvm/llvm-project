// RUN: mlir-opt -allow-unregistered-dialect %s -symbol-dce -split-input-file -verify-diagnostics | FileCheck %s
// RUN: mlir-opt -allow-unregistered-dialect %s -pass-pipeline="builtin.module(symbol-dce)" -split-input-file | FileCheck %s --check-prefix=NESTED

// Check that trivially dead and trivially live non-nested cases are handled.

// CHECK-LABEL: module attributes {test.simple}
module attributes {test.simple} {
  // CHECK-NOT: func private @dead_private_function
  func.func private @dead_private_function()

  // CHECK-NOT: func nested @dead_nested_function
  func.func nested @dead_nested_function()

  // CHECK: func private @live_private_function
  func.func private @live_private_function()

  // CHECK: func nested @live_nested_function
  func.func nested @live_nested_function()

  // CHECK: func @public_function
  func.func @public_function() {
    "foo.return"() {uses = [@live_private_function, @live_nested_function]} : () -> ()
  }
}

// -----

// Check that we don't DCE nested symbols if they are used.
// CHECK-LABEL: module attributes {test.nested}
module attributes {test.nested} {
  // CHECK: module @public_module
  module @public_module {
    // CHECK-NOT: func nested @dead_nested_function
    func.func nested @dead_nested_function()

    // CHECK: func private @private_function
    func.func private @private_function()

    // CHECK: func nested @nested_function
    func.func nested @nested_function() {
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
    // NESTED: func nested @nested_function
    func.func nested @nested_function()
  }
  // NESTED: module @nested_module
  module @nested_module attributes { sym_visibility = "nested" } {
    // NESTED: func nested @nested_function
    func.func nested @nested_function()
  }

  // Only private modules can be assumed to be hidden.
  // NESTED: module @private_module
  module @private_module attributes { sym_visibility = "private" } {
    // NESTED-NOT: func nested @nested_function
    func.func nested @nested_function()
  }

  "live.user"() {uses = [@nested_module, @private_module]} : () -> ()
}

// -----

module {
  func.func private @private_symbol()

  // expected-error@+1 {{contains potentially unknown symbol table}}
  "foo.possibly_unknown_symbol_table"() ({
  }) : () -> ()
}

// -----

// Check that unknown symbol references are OK.
module {
  // CHECK-NOT: func private @dead_private_function
  func.func private @dead_private_function()

  // CHECK: func private @live_private_function
  func.func private @live_private_function()

  // CHECK: "live.user"() {uses = [@live_private_function]} : () -> ()
  "live.user"() {uses = [@live_private_function]} : () -> ()

  // CHECK: "live.user"() {uses = [@unknown_symbol]} : () -> ()
  "live.user"() {uses = [@unknown_symbol]} : () -> ()
}
