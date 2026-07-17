// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// Human-readable floating point special values and C-style hexadecimal floats,
// as accepted by the parser and produced by the printer.

// CHECK-LABEL: @infinities
func.func @infinities() {
  // CHECK: arith.constant +inf : f32
  %0 = arith.constant +inf : f32
  // CHECK: arith.constant -inf : f32
  %1 = arith.constant -inf : f32
  // CHECK: arith.constant +inf : f64
  %2 = arith.constant +inf : f64
  // CHECK: arith.constant -inf : f16
  %3 = arith.constant -inf : f16
  // CHECK: arith.constant +inf : bf16
  %4 = arith.constant +inf : bf16
  return
}

// CHECK-LABEL: @quiet_nans
func.func @quiet_nans() {
  // CHECK: arith.constant +qnan : f32
  %0 = arith.constant +qnan : f32
  // CHECK: arith.constant -qnan : f64
  %1 = arith.constant -qnan : f64
  // CHECK: arith.constant +qnan : f16
  %2 = arith.constant +qnan : f16
  // CHECK: arith.constant -qnan : bf16
  %3 = arith.constant -qnan : bf16
  return
}

// CHECK-LABEL: @nan_payloads
func.func @nan_payloads() {
  // CHECK: arith.constant +nan(0x1) : f32
  %0 = arith.constant +nan(0x1) : f32
  // CHECK: arith.constant -nan(0x3FFFFF) : f32
  %1 = arith.constant -nan(0x3FFFFF) : f32
  // CHECK: arith.constant +snan(0x1) : f64
  %2 = arith.constant +snan(0x1) : f64
  // CHECK: arith.constant -snan(0x1000000) : f64
  %3 = arith.constant -snan(0x1000000) : f64
  return
}

// Bare `nan`/`snan` (no payload) are accepted on IEEE types: `nan` normalizes to
// the canonical quiet NaN, `snan` to a signaling NaN with the default payload.
// CHECK-LABEL: @bare_nan_snan
func.func @bare_nan_snan() {
  // CHECK: arith.constant +qnan : f32
  %0 = arith.constant +nan : f32
  // CHECK: arith.constant -qnan : f32
  %1 = arith.constant -nan : f32
  // CHECK: arith.constant +snan(0x200000) : f32
  %2 = arith.constant +snan : f32
  // CHECK: arith.constant -snan(0x4000000000000) : f64
  %3 = arith.constant -snan : f64
  return
}

// CHECK-LABEL: @hex_floats
func.func @hex_floats() {
  // 0x1.8p3 == 1.5 * 2^3 == 12.0; the exponent sign and 'p'/'P' case may vary.
  // CHECK: arith.constant 1.200000e+01 : f64
  %0 = arith.constant 0x1.8p3 : f64
  // CHECK: arith.constant 1.200000e+01 : f64
  %1 = arith.constant 0x1.8p+3 : f64
  // CHECK: arith.constant 1.200000e+01 : f64
  %2 = arith.constant 0x1.8P3 : f64
  // CHECK: arith.constant -1.200000e+01 : f64
  %3 = arith.constant -0x1.8p3 : f64
  // 0x1.8p-3 == 1.5 * 2^-3 == 0.1875
  // CHECK: arith.constant 1.875000e-01 : f64
  %4 = arith.constant 0x1.8p-3 : f64
  // No fractional part.
  // CHECK: arith.constant 1.600000e+01 : f32
  %5 = arith.constant 0x1p4 : f32
  // CHECK: arith.constant 0.000000e+00 : f32
  %6 = arith.constant 0x0p0 : f32
  // Built directly in the (wider than double) target semantics.
  // CHECK: arith.constant 1.200000e+01 : f80
  %7 = arith.constant 0x1.8p3 : f80
  // CHECK: arith.constant 1.200000e+01 : f128
  %8 = arith.constant 0x1.8p3 : f128
  // A value that needs full precision to round-trip.
  // CHECK: arith.constant 3.1415926535897931 : f64
  %9 = arith.constant 0x1.921fb54442d18p1 : f64
  return
}

// CHECK-LABEL: @special_values_in_elements
func.func @special_values_in_elements() {
  // CHECK: dense<[+inf, -inf, +qnan]> : tensor<3xf32>
  %0 = arith.constant dense<[+inf, -inf, +qnan]> : tensor<3xf32>
  // A splat built from a single special value.
  // CHECK: dense<-inf> : tensor<4xf64>
  %1 = arith.constant dense<-inf> : tensor<4xf64>
  return
}

// Low-precision and other builtin float types on the special-value path.
// CHECK-LABEL: @low_precision_and_other_types
func.func @low_precision_and_other_types() {
  // f8E8M0FNU is a NanOnly (AllOnes) unsigned type: a single NaN, spelled `nan`;
  // the `qnan` alias normalizes to it.
  // CHECK: arith.constant +nan : f8E8M0FNU
  %0 = arith.constant +qnan : f8E8M0FNU
  // A direct `+nan` round-trips (not only via the `qnan` alias).
  // CHECK: arith.constant +nan : f8E8M0FNU
  %1 = arith.constant +nan : f8E8M0FNU
  // f8E5M2 is IEEE-encoded, so it keeps a payload (masked to 1 bit here).
  // CHECK: arith.constant +nan(0x1) : f8E5M2
  %2 = arith.constant +nan(0x7) : f8E5M2
  // CHECK: arith.constant +inf : tf32
  %3 = arith.constant +inf : tf32
  return
}

// Payload spelling normalizations and overflow tolerance.
// CHECK-LABEL: @nan_payload_and_overflow
func.func @nan_payload_and_overflow() {
  // A zero payload normalizes to the canonical quiet NaN.
  // CHECK: arith.constant +qnan : f32
  %0 = arith.constant +nan(0x0) : f32
  // An unprefixed payload is decimal; it prints back as hex.
  // CHECK: arith.constant +nan(0xA) : f32
  %1 = arith.constant +nan(10) : f32
  // Decimal overflow saturates to infinity rather than erroring.
  // CHECK: arith.constant +inf : f16
  %2 = arith.constant 1.0e40 : f16
  return
}

// AllOnes NaN encoding (f8E4M3FN): a single NaN per sign, spelled `nan`, with no
// payload and no quiet/signaling bit. The `qnan` alias normalizes to `nan`.
// CHECK-LABEL: @nan_all_ones
func.func @nan_all_ones() {
  // CHECK: arith.constant +nan : f8E4M3FN
  %0 = arith.constant +nan : f8E4M3FN
  // CHECK: arith.constant -nan : f8E4M3FN
  %1 = arith.constant -nan : f8E4M3FN
  // CHECK: arith.constant +nan : f8E4M3FN
  %2 = arith.constant +qnan : f8E4M3FN
  // CHECK: arith.constant -nan : f8E4M3FN
  %3 = arith.constant -qnan : f8E4M3FN
  return
}

// NegativeZero NaN encoding (f8E4M3FNUZ): a single, always-negative NaN, spelled
// `-nan`. The `-qnan` alias normalizes to it.
// CHECK-LABEL: @nan_negative_zero
func.func @nan_negative_zero() {
  // CHECK: arith.constant -nan : f8E4M3FNUZ
  %0 = arith.constant -nan : f8E4M3FNUZ
  // CHECK: arith.constant -nan : f8E4M3FNUZ
  %1 = arith.constant -qnan : f8E4M3FNUZ
  return
}
