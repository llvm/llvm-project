// RUN: mlir-opt %s -verify-diagnostics -split-input-file | FileCheck %s

// Tests for the builtin `token` type, the token producer/consumer operation
// traits, and the `Token`, `AnyType` ODS predicates. The default `AnyType`
// predicate excludes tokens.

// CHECK-LABEL: @token_produce_consume
func.func @token_produce_consume() {
  // CHECK: %[[T:.*]] = test.token.produce
  %t = test.token.produce
  // CHECK: test.token.consume %[[T]]
  test.token.consume %t
  return
}

// -----

// Region entry block arguments may produce tokens when the parent op opts in.
// CHECK-LABEL: @token_region_entry_block_arg
func.func @token_region_entry_block_arg() {
  // CHECK: "test.token.region"
  "test.token.region"() ({
  ^bb0(%arg0: token):
    // CHECK: test.token.consume
    test.token.consume %arg0
    "test.finish"() : () -> ()
  }) : () -> ()
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

// Token-producing ops must have the TokenProducerTrait.
func.func @token_result_requires_producer_trait() {
  // expected-error @below {{'test.token.produce_without_trait' op produces token result #0 but does not have the TokenProducerTrait}}
  %t = test.token.produce_without_trait : token
  return
}

// -----

// Token-consuming ops must have the TokenConsumerTrait.
func.func @token_operand_requires_consumer_trait() {
  %t = test.token.produce
  // expected-error @below {{'test.token.consume_without_trait' op consumes token operand #0 but does not have the TokenConsumerTrait}}
  test.token.consume_without_trait %t : token
  return
}

// -----

// Token entry block arguments require the parent op to have the
// TokenProducerTrait.
func.func @token_entry_block_arg_requires_parent_producer_trait() {
  "test.token.region_without_trait"() ({
  // expected-error @below {{token entry block argument #0 requires the parent operation to have the TokenProducerTrait}}
  ^bb0(%arg0: token):
    test.token.consume %arg0
    "test.finish"() : () -> ()
  }) : () -> ()
  return
}

// -----

// A region with a parent op still cannot have token entry block arguments unless
// the parent op has the TokenProducerTrait.
func.func @token_entry_block_arg_requires_parent_producer_trait_without_uses() {
  "test.token.region_without_trait"() ({
  // expected-error @below {{token entry block argument #0 requires the parent operation to have the TokenProducerTrait}}
  ^bb0(%arg0: token):
    "test.finish"() : () -> ()
  }) : () -> ()
  return
}

// -----

// Token entry block arguments still require consumers to have the
// TokenConsumerTrait.
func.func @token_entry_block_arg_use_requires_consumer_trait() {
  "test.token.region"() ({
  ^bb0(%arg0: token):
    // expected-error @below {{'test.token.consume_without_trait' op consumes token operand #0 but does not have the TokenConsumerTrait}}
    test.token.consume_without_trait %arg0 : token
    "test.finish"() : () -> ()
  }) : () -> ()
  return
}

// -----

// Tokens cannot be non-entry block arguments.
func.func @token_non_entry_block_arg_is_rejected() {
  "test.token.region"() ({
    "test.finish"() : () -> ()
  // expected-error @below {{token block argument #0 is only allowed in a region entry block}}
  ^bb1(%arg0: token):
    "test.finish"() : () -> ()
  }) : () -> ()
  return
}

// -----

// Function entry blocks do not opt in to producing builtin tokens.
// expected-error @below {{token entry block argument #0 requires the parent operation to have the TokenProducerTrait}}
func.func @token_region_arg(%arg0: token) {
  test.token.consume %arg0
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
