// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// Test basic symbol verification using discardable attribute.
module {
  func.func @existing_symbol() { return }
  
  func.func @test() attributes {symbol_ref = #test.symbol_ref_attr<@existing_symbol>} { return }
}

// -----

// Test invalid symbol reference, symbol does not exist.
module {
  // expected-error@+1 {{TestSymbolRefAttr::verifySymbolUses: '@non_existent_symbol' does not reference a valid symbol}}
  func.func @test() attributes {symbol_ref = #test.symbol_ref_attr<@non_existent_symbol>} { return }
}
