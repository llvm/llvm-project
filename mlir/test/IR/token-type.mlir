// RUN: mlir-opt %s -verify-diagnostics -split-input-file | FileCheck %s

// Tests for the builtin `token` type and the `Token`, `AnyType` ODS predicates.
// The default `AnyType` predicate excludes tokens.

// CHECK-LABEL: @token_produce_consume
func.func @token_produce_consume() {
  // CHECK: %[[T:.*]] = test.token.produce
  %t = test.token.produce
  // CHECK: test.token.consume %[[T]]
  test.token.consume %t
  return
}

// -----

// `AnyType` accepts arbitrary non-token types.
// CHECK-LABEL: @any_type_with_non_token
func.func @any_type_with_non_token(%arg0: i32) {
  // CHECK: test.token.any_type %{{.*}} : i32
  test.token.any_type %arg0 : i32
  return
}

// -----

// `AnyType` rejects tokens by default.
func.func @any_type_rejects_token() {
  %t = test.token.produce
  // expected-error @below {{operand #0 must be any non-token type}}
  test.token.any_type %t : token
  return
}

// -----

// `Token` rejects non-token types. The op's operand type is fixed to the
// builtin `token` (it's a `BuildableType`), so passing a non-token SSA value
// fails at parse time with an SSA type mismatch.
// expected-note @below {{prior use here}}
func.func @token_rejects_non_token(%arg0: i32) {
  // expected-error @below {{use of value '%arg0' expects different type than prior uses: 'token' vs 'i32'}}
  test.token.consume %arg0
  return
}
