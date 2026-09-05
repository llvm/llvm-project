// RUN: mlir-opt %s -verify-diagnostics -split-input-file

module {
  func.func private @existing_symbol()

  "test.type_producer"() : () -> !test.symbol_ref<@existing_symbol>
}

// -----

module {
  // expected-error@+1 {{'@non_existent_symbol' does not reference a valid symbol}}
  "test.type_producer"() : () -> !test.symbol_ref<@non_existent_symbol>
}

// -----

module {
  func.func private @existing_symbol()

  %0 = "test.type_producer"() : () -> !test.symbol_ref<@existing_symbol>
  "test.type_consumer"(%0) : (!test.symbol_ref<@existing_symbol>) -> ()
}

// -----

module {
  func.func private @existing_symbol()

  "test.type_producer"() : () -> tuple<!test.symbol_ref<@existing_symbol>>
}

// -----

module {
  // expected-error@+1 {{'@non_existent_symbol' does not reference a valid symbol}}
  "test.type_producer"() : () -> tuple<!test.symbol_ref<@non_existent_symbol>>
}

// -----

module {
  func.func private @existing_symbol()

  func.func private @uses_symbol_type(%arg0: !test.symbol_ref<@existing_symbol>)
}

// -----

module {
  // expected-error@+1 {{'@non_existent_symbol' does not reference a valid symbol}}
  func.func private @uses_symbol_type(%arg0: !test.symbol_ref<@non_existent_symbol>)
}

// -----

module {
  func.func private @existing_symbol()

  "test.one_region_op"() ({
  ^bb0(%arg0: !test.symbol_ref<@existing_symbol>):
    "test.valid"() : () -> ()
  }) : () -> ()
}

// -----

module {
  // expected-error@+1 {{'@non_existent_symbol' does not reference a valid symbol}}
  "test.one_region_op"() ({
  ^bb0(%arg0: !test.symbol_ref<@non_existent_symbol>):
    "test.valid"() : () -> ()
  }) : () -> ()
}

// -----

module {
  func.func private @existing_symbol()

  "test.typed_attr"() <{
    type = !test.symbol_ref<@existing_symbol>,
    attr = 0 : i32
  }> : () -> ()
}

// -----

module {
  // expected-error@+1 {{'@non_existent_symbol' does not reference a valid symbol}}
  "test.typed_attr"() <{
    type = !test.symbol_ref<@non_existent_symbol>,
    attr = 0 : i32
  }> : () -> ()
}
