// RUN: mlir-opt %s -test-elements-attr-interface -verify-diagnostics

// Parsing external resources does not work on big-endian platforms currently
// XFAIL: s390x-

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

// expected-error@below {{Test iterating `int64_t`: unable to iterate type}}
// expected-error@below {{Test iterating `uint64_t`: unable to iterate type}}
// expected-error@below {{Test iterating `APInt`: unable to iterate type}}
// expected-error@below {{Test iterating `IntegerAttr`: unable to iterate type}}
arith.constant opaque<"_", "0xDEADBEEF"> : tensor<5xi64>

// Check that we don't crash on empty element attributes.
// expected-error@below {{Test iterating `int64_t`: }}
// expected-error@below {{Test iterating `uint64_t`: }}
// expected-error@below {{Test iterating `APInt`: }}
// expected-error@below {{Test iterating `IntegerAttr`: }}
arith.constant dense<> : tensor<0xi64>

// expected-error@below {{Test iterating `int8_t`: 10, 11, -12, 13, 14}}
arith.constant [:i8 10, 11, -12, 13, 14]
// expected-error@below {{Test iterating `int16_t`: 10, 11, -12, 13, 14}}
arith.constant [:i16 10, 11, -12, 13, 14]
// expected-error@below {{Test iterating `int32_t`: 10, 11, -12, 13, 14}}
arith.constant [:i32 10, 11, -12, 13, 14]
// expected-error@below {{Test iterating `int64_t`: 10, 11, -12, 13, 14}}
arith.constant [:i64 10, 11, -12, 13, 14]
// expected-error@below {{Test iterating `float`: 10.00, 11.00, -12.00, 13.00, 14.00}}
arith.constant [:f32 10., 11., -12., 13., 14.]
// expected-error@below {{Test iterating `double`: 10.00, 11.00, -12.00, 13.00, 14.00}}
arith.constant [:f64 10., 11., -12., 13., 14.]

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
