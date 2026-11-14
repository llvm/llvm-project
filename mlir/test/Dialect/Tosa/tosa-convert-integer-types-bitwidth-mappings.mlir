// REQUIRES: host-byteorder-little-endian
// RUN: mlir-opt -split-input-file -tosa-convert-integer-types-bitwidth='bitwidth-mappings=4:8,48:64' -mlir-print-op-generic %s | FileCheck %s

// Convert nested types even when resource data is unavailable and its element
// type is unchanged. Reversing the mapping list must have no effect.
// CHECK-LABEL: "builtin.module"() <{sym_name = "mapping_encodings"}>
// CHECK: test.type = tensor<2xi8, {test.widths = [i8, i64, i8]}>
// CHECK-SAME: test.unresolved_resource = dense_resource<missing> : tensor<2xi7, {test.widths = [i8, i64, i8]}>
module @mapping_encodings attributes {
  test.type = tensor<2xi4, {test.widths = [i4, i48, i8]}>,
  test.unresolved_resource = dense_resource<missing> : tensor<2xi7, {test.widths = [i4, i48, i8]}>
} {}

// -----

// Widen resource elements and their encodings together. Discard padding bits
// before interpreting the i4 sign bit.
// CHECK-LABEL: "builtin.module"() <{sym_name = "widen_resources"}>
// CHECK: test.values = dense_resource<[[VALUES:[a-zA-Z0-9_]+]]> : tensor<4xi8, {test.widths = [i8, i64, i8]}>
module @widen_resources attributes {
  test.values = dense_resource<values> : tensor<4xi4, {test.widths = [i4, i48, i8]}>
} {}

{-# dialect_resources: { builtin: {
  // [-8, -1, 0, 7], with one byte per i4 value and nonzero padding bits.
  values: "0x01000000F8FFB0A7"
  // CHECK: [[VALUES]]: "0x01000000F8FF0007"
} } #-}
