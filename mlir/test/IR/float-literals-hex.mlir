// RUN: mlir-opt %s --mlir-print-float-special-literals-as-hex | FileCheck %s --check-prefix=HEX
// RUN: mlir-opt %s --mlir-print-float-special-literals-as-hex | mlir-opt | FileCheck %s --check-prefix=ROUNDTRIP

// The --mlir-print-float-special-literals-as-hex flag prints infinities and NaNs as a
// hexadecimal bit pattern (the legacy form). The parser still accepts both the
// hex and the human-readable spellings, so the hex output round-trips back to
// the human-readable form.

// HEX-LABEL: @infinities
// ROUNDTRIP-LABEL: @infinities
func.func @infinities() {
  // HEX: arith.constant 0x7F800000 : f32
  // ROUNDTRIP: arith.constant +inf : f32
  %0 = arith.constant +inf : f32
  // HEX: arith.constant 0xFF800000 : f32
  // ROUNDTRIP: arith.constant -inf : f32
  %1 = arith.constant -inf : f32
  // HEX: arith.constant 0x7FF0000000000000 : f64
  // ROUNDTRIP: arith.constant +inf : f64
  %2 = arith.constant +inf : f64
  return
}

// HEX-LABEL: @nans
// ROUNDTRIP-LABEL: @nans
func.func @nans() {
  // HEX: arith.constant 0x7FC00000 : f32
  // ROUNDTRIP: arith.constant +qnan : f32
  %0 = arith.constant +qnan : f32
  // HEX: arith.constant 0xFFF8000000000000 : f64
  // ROUNDTRIP: arith.constant -qnan : f64
  %1 = arith.constant -qnan : f64
  // HEX: arith.constant 0x7FC00001 : f32
  // ROUNDTRIP: arith.constant +nan(0x1) : f32
  %2 = arith.constant +nan(0x1) : f32
  // HEX: arith.constant 0x7FF0000000000001 : f64
  // ROUNDTRIP: arith.constant +snan(0x1) : f64
  %3 = arith.constant +snan(0x1) : f64
  // NanOnly types also print their bit pattern in the legacy form and round-trip
  // to the human-readable `nan`.
  // HEX: arith.constant 0x7F : f8E4M3FN
  // ROUNDTRIP: arith.constant +nan : f8E4M3FN
  %4 = arith.constant +nan : f8E4M3FN
  // HEX: arith.constant 0xFF : f8E4M3FN
  // ROUNDTRIP: arith.constant -nan : f8E4M3FN
  %5 = arith.constant -nan : f8E4M3FN
  // HEX: arith.constant 0x80 : f8E4M3FNUZ
  // ROUNDTRIP: arith.constant -nan : f8E4M3FNUZ
  %6 = arith.constant -nan : f8E4M3FNUZ
  return
}

// Special values also print as hex inside dense elements attributes.
// HEX-LABEL: @special_values_in_elements
// ROUNDTRIP-LABEL: @special_values_in_elements
func.func @special_values_in_elements() {
  // HEX: dense<[0x7F800000, 0xFF800000, 0x7FC00000]> : tensor<3xf32>
  // ROUNDTRIP: dense<[+inf, -inf, +qnan]> : tensor<3xf32>
  %0 = arith.constant dense<[+inf, -inf, +qnan]> : tensor<3xf32>
  return
}
