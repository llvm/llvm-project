// RUN: mlir-opt %s -test-elements-attr-interface -verify-diagnostics

// This test contains various `ElementsAttr` attributes, and tests the support
// for iterating the values of these attributes using various native C++ types.
// This tests that the abstract iteration of ElementsAttr works properly, and
// is properly failable when necessary.

// expected-error@below {{Test iterating `uint64_t`: 10, 11, 12, 13, 14}}
// expected-error@below {{Test iterating `APInt`: 10, 11, 12, 13, 14}}
// expected-error@below {{Test iterating `IntegerAttr`: 10 : i64, 11 : i64, 12 : i64, 13 : i64, 14 : i64}}
arith.constant #test.i64_elements<[10, 11, 12, 13, 14]> : tensor<5xi64>

// expected-error@below {{Test iterating `uint64_t`: 10, 11, 12, 13, 14}}
// expected-error@below {{Test iterating `APInt`: 10, 11, 12, 13, 14}}
// expected-error@below {{Test iterating `IntegerAttr`: 10 : i64, 11 : i64, 12 : i64, 13 : i64, 14 : i64}}
arith.constant dense<[10, 11, 12, 13, 14]> : tensor<5xi64>

// expected-error@below {{Test iterating `uint64_t`: unable to iterate type}}
// expected-error@below {{Test iterating `APInt`: unable to iterate type}}
// expected-error@below {{Test iterating `IntegerAttr`: unable to iterate type}}
arith.constant opaque<"_", "0xDEADBEEF"> : tensor<5xi64>

// Check that we don't crash on empty element attributes.
// expected-error@below {{Test iterating `uint64_t`: }}
// expected-error@below {{Test iterating `APInt`: }}
// expected-error@below {{Test iterating `IntegerAttr`: }}
arith.constant dense<> : tensor<0xi64>
