// RUN: mlir-opt -allow-unregistered-dialect %s -split-input-file -verify-diagnostics

//===----------------------------------------------------------------------===//
// Parser error tests
//===----------------------------------------------------------------------===//

// Test missing '<' after 'quantile' keyword.
// expected-error @+1 {{expected '<' in quantile type}}
func.func private @missing_lt() -> quantile ui4:f16, {1.0}>

// -----

// Test missing ':' between storage type and quantile type.
// expected-error @+1 {{expected ':' in quantile type}}
func.func private @missing_colon() -> quantile<ui4 f16, {1.0}>

// -----

// Test missing ',' between quantile type and quantile value list.
// expected-error @+1 {{expected ',' in quantile type}}
func.func private @missing_comma() -> quantile<ui4:f16 {1.0}>

// -----

// Test missing '{' before quantile value list.
// expected-error @+1 {{expected '{' in quantile type}}
func.func private @missing_lbrace() -> quantile<ui4:f16, 1.0}>

// -----

// Test missing '}' after quantile value list.
// expected-error @+1 {{expected '}' in quantile type}}
func.func private @missing_rbrace() -> quantile<ui4:f16, {1.0>

// -----

// Test missing '>' closing the quantile type.
// expected-error @+1 {{expected '>' in quantile type}}
func.func private @missing_gt() -> quantile<ui4:f16, {1.0}
