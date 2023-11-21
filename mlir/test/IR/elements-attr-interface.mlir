// RUN: mlir-opt %s -test-elements-attr-interface -verify-diagnostics

// Parsing external resources does not work on big-endian platforms currently
// XFAIL: target=s390x-{{.*}}

// This test contains various `ElementsAttr` attributes, and tests the support
// for iterating the values of these attributes using various native C++ types.
// This tests that the abstract iteration of ElementsAttr works properly, and
// is properly failable when necessary.

// expected-error@below {{Test iterating `int64_t`: unable to iterate type}}
// expected-error@below {{Test iterating `uint64_t`: 10, 11, 12, 13, 14}}
// expected-error@below {{Test iterating `APInt`: 10, 11, 12, 13, 14}}
// expected-error@below {{Test iterating `IntegerAttr`: 10 : i64, 11 : i64, 12 : i64, 13 : i64, 14 : i64}}
arith.constant #test.i64_elements<[10, 11, 12, 13, 14]> : tensor<5xi64>

// expected-error@below {{Test iterating `int64_t`: 10, 11, 12, 13, 14}}
// expected-error@below {{Test iterating `uint64_t`: 10, 11, 12, 13, 14}}
// expected-error@below {{Test iterating `APInt`: 10, 11, 12, 13, 14}}
// expected-error@below {{Test iterating `IntegerAttr`: 10 : i64, 11 : i64, 12 : i64, 13 : i64, 14 : i64}}
arith.constant dense<[10, 11, 12, 13, 14]> : tensor<5xi64>

// This test is expected to only be called on integer elements.
// expected-error@below {{Test iterating `int64_t`: expected element type to be an integer type}}
// expected-error@below {{Test iterating `uint64_t`: expected element type to be an integer type}}
// expected-error@below {{Test iterating `APInt`: expected element type to be an integer type}}
// expected-error@below {{Test iterating `IntegerAttr`: expected element type to be an integer type}}
arith.constant dense<[1.1, 1.2, 1.3]> : tensor<3xf32>

// Check that we don't crash on empty element attributes.
// expected-error@below {{Test iterating `int64_t`: }}
// expected-error@below {{Test iterating `uint64_t`: }}
// expected-error@below {{Test iterating `APInt`: }}
// expected-error@below {{Test iterating `IntegerAttr`: }}
arith.constant dense<> : tensor<0xi64>

// Check that we handle an external constant parsed from the config.
// expected-error@below {{Test iterating `int64_t`: unable to iterate type}}
// expected-error@below {{Test iterating `uint64_t`: 1, 2, 3}}
// expected-error@below {{Test iterating `APInt`: unable to iterate type}}
// expected-error@below {{Test iterating `IntegerAttr`: unable to iterate type}}
arith.constant #test.e1di64_elements<blob1> : tensor<3xi64>

{-#
  dialect_resources: {
    test: {
      blob1: "0x08000000010000000000000002000000000000000300000000000000"
    }
  }
#-}

// expected-error@below {{Test iterating `int64_t`: 0, 0, 1}}
// expected-error@below {{Test iterating `uint64_t`: 0, 0, 1}}
// expected-error@below {{Test iterating `APInt`: 0, 0, 1}}
// expected-error@below {{Test iterating `IntegerAttr`: 0 : i64, 0 : i64, 1 : i64}}
arith.constant sparse<[[0, 0, 2]], 1> : vector <1x1x3xi64>
