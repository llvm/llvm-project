// Value tests for the narrow-float truncf/extf expansion. Expected codes come
// from an independent model of APFloat::convert (formats without Inf or NaN
// saturate); inputs are runtime arguments, so nothing is constant folded.

// RUN: mlir-opt %s --arith-expand="include-f6e2m3fn=true include-f6e3m2fn=true \
// RUN:     include-f8e3m4=true include-f8e4m3=true include-f8e4m3b11fnuz=true \
// RUN:     include-f8e4m3fnuz=true include-f8e5m2fnuz=true" \
// RUN:     --convert-vector-to-scf --convert-scf-to-cf --convert-cf-to-llvm \
// RUN:     --convert-vector-to-llvm --convert-func-to-llvm \
// RUN:     --convert-arith-to-llvm --reconcile-unrealized-casts | \
// RUN:   mlir-runner -e entry --entry-point-result=void \
// RUN:               --shared-libs=%mlir_c_runner_utils | \
// RUN:   FileCheck %s --match-full-lines

// Prints the codes of a vector as unsigned integers.
func.func @print_f6E2M3FN(%v : vector<16xf6E2M3FN>) {
  %bits = arith.bitcast %v : vector<16xf6E2M3FN> to vector<16xi6>
  %u = vector.bitcast %bits : vector<16xi6> to vector<16xui6>
  vector.print %u : vector<16xui6>
  return
}

func.func @print_f6E3M2FN(%v : vector<16xf6E3M2FN>) {
  %bits = arith.bitcast %v : vector<16xf6E3M2FN> to vector<16xi6>
  %u = vector.bitcast %bits : vector<16xi6> to vector<16xui6>
  vector.print %u : vector<16xui6>
  return
}

func.func @print_f8E3M4(%v : vector<16xf8E3M4>) {
  %bits = arith.bitcast %v : vector<16xf8E3M4> to vector<16xi8>
  %u = vector.bitcast %bits : vector<16xi8> to vector<16xui8>
  vector.print %u : vector<16xui8>
  return
}

func.func @print_f8E4M3(%v : vector<16xf8E4M3>) {
  %bits = arith.bitcast %v : vector<16xf8E4M3> to vector<16xi8>
  %u = vector.bitcast %bits : vector<16xi8> to vector<16xui8>
  vector.print %u : vector<16xui8>
  return
}

func.func @print_f8E4M3FNUZ(%v : vector<16xf8E4M3FNUZ>) {
  %bits = arith.bitcast %v : vector<16xf8E4M3FNUZ> to vector<16xi8>
  %u = vector.bitcast %bits : vector<16xi8> to vector<16xui8>
  vector.print %u : vector<16xui8>
  return
}

func.func @print_f8E4M3B11FNUZ(%v : vector<16xf8E4M3B11FNUZ>) {
  %bits = arith.bitcast %v : vector<16xf8E4M3B11FNUZ> to vector<16xi8>
  %u = vector.bitcast %bits : vector<16xi8> to vector<16xui8>
  vector.print %u : vector<16xui8>
  return
}

func.func @print_f8E5M2FNUZ(%v : vector<16xf8E5M2FNUZ>) {
  %bits = arith.bitcast %v : vector<16xf8E5M2FNUZ> to vector<16xi8>
  %u = vector.bitcast %bits : vector<16xi8> to vector<16xui8>
  vector.print %u : vector<16xui8>
  return
}

func.func @print_f32(%v : vector<8xf32>) {
  %bits = arith.bitcast %v : vector<8xf32> to vector<8xi32>
  %u = vector.bitcast %bits : vector<8xi32> to vector<8xui32>
  vector.print %u : vector<8xui32>
  return
}

func.func @print_f16(%v : vector<8xf16>) {
  %bits = arith.bitcast %v : vector<8xf16> to vector<8xi16>
  %u = vector.bitcast %bits : vector<8xi16> to vector<8xui16>
  vector.print %u : vector<8xui16>
  return
}

func.func @print_bf16(%v : vector<8xbf16>) {
  %bits = arith.bitcast %v : vector<8xbf16> to vector<8xi16>
  %u = vector.bitcast %bits : vector<8xi16> to vector<8xui16>
  vector.print %u : vector<8xui16>
  return
}

func.func @print_f64(%v : vector<8xf64>) {
  %bits = arith.bitcast %v : vector<8xf64> to vector<8xi64>
  %u = vector.bitcast %bits : vector<8xi64> to vector<8xui64>
  vector.print %u : vector<8xui64>
  return
}

func.func @truncf_f32_to_f6E2M3FN(%bits : vector<16xi32>) {
  %x = arith.bitcast %bits : vector<16xi32> to vector<16xf32>
  %0 = arith.truncf %x : vector<16xf32> to vector<16xf6E2M3FN>
  func.call @print_f6E2M3FN(%0) : (vector<16xf6E2M3FN>) -> ()
  %1 = arith.truncf %x to_nearest_away : vector<16xf32> to vector<16xf6E2M3FN>
  func.call @print_f6E2M3FN(%1) : (vector<16xf6E2M3FN>) -> ()
  %2 = arith.truncf %x toward_zero : vector<16xf32> to vector<16xf6E2M3FN>
  func.call @print_f6E2M3FN(%2) : (vector<16xf6E2M3FN>) -> ()
  %3 = arith.truncf %x upward : vector<16xf32> to vector<16xf6E2M3FN>
  func.call @print_f6E2M3FN(%3) : (vector<16xf6E2M3FN>) -> ()
  %4 = arith.truncf %x downward : vector<16xf32> to vector<16xf6E2M3FN>
  func.call @print_f6E2M3FN(%4) : (vector<16xf6E2M3FN>) -> ()
  return
}

func.func @truncf_f32_to_f6E3M2FN(%bits : vector<16xi32>) {
  %x = arith.bitcast %bits : vector<16xi32> to vector<16xf32>
  %0 = arith.truncf %x : vector<16xf32> to vector<16xf6E3M2FN>
  func.call @print_f6E3M2FN(%0) : (vector<16xf6E3M2FN>) -> ()
  %1 = arith.truncf %x to_nearest_away : vector<16xf32> to vector<16xf6E3M2FN>
  func.call @print_f6E3M2FN(%1) : (vector<16xf6E3M2FN>) -> ()
  %2 = arith.truncf %x toward_zero : vector<16xf32> to vector<16xf6E3M2FN>
  func.call @print_f6E3M2FN(%2) : (vector<16xf6E3M2FN>) -> ()
  %3 = arith.truncf %x upward : vector<16xf32> to vector<16xf6E3M2FN>
  func.call @print_f6E3M2FN(%3) : (vector<16xf6E3M2FN>) -> ()
  %4 = arith.truncf %x downward : vector<16xf32> to vector<16xf6E3M2FN>
  func.call @print_f6E3M2FN(%4) : (vector<16xf6E3M2FN>) -> ()
  return
}

func.func @truncf_f32_to_f8E3M4(%bits : vector<16xi32>) {
  %x = arith.bitcast %bits : vector<16xi32> to vector<16xf32>
  %0 = arith.truncf %x : vector<16xf32> to vector<16xf8E3M4>
  func.call @print_f8E3M4(%0) : (vector<16xf8E3M4>) -> ()
  %1 = arith.truncf %x to_nearest_away : vector<16xf32> to vector<16xf8E3M4>
  func.call @print_f8E3M4(%1) : (vector<16xf8E3M4>) -> ()
  %2 = arith.truncf %x toward_zero : vector<16xf32> to vector<16xf8E3M4>
  func.call @print_f8E3M4(%2) : (vector<16xf8E3M4>) -> ()
  %3 = arith.truncf %x upward : vector<16xf32> to vector<16xf8E3M4>
  func.call @print_f8E3M4(%3) : (vector<16xf8E3M4>) -> ()
  %4 = arith.truncf %x downward : vector<16xf32> to vector<16xf8E3M4>
  func.call @print_f8E3M4(%4) : (vector<16xf8E3M4>) -> ()
  return
}

func.func @truncf_f32_to_f8E4M3(%bits : vector<16xi32>) {
  %x = arith.bitcast %bits : vector<16xi32> to vector<16xf32>
  %0 = arith.truncf %x : vector<16xf32> to vector<16xf8E4M3>
  func.call @print_f8E4M3(%0) : (vector<16xf8E4M3>) -> ()
  %1 = arith.truncf %x to_nearest_away : vector<16xf32> to vector<16xf8E4M3>
  func.call @print_f8E4M3(%1) : (vector<16xf8E4M3>) -> ()
  %2 = arith.truncf %x toward_zero : vector<16xf32> to vector<16xf8E4M3>
  func.call @print_f8E4M3(%2) : (vector<16xf8E4M3>) -> ()
  %3 = arith.truncf %x upward : vector<16xf32> to vector<16xf8E4M3>
  func.call @print_f8E4M3(%3) : (vector<16xf8E4M3>) -> ()
  %4 = arith.truncf %x downward : vector<16xf32> to vector<16xf8E4M3>
  func.call @print_f8E4M3(%4) : (vector<16xf8E4M3>) -> ()
  return
}

func.func @truncf_f32_to_f8E4M3FNUZ(%bits : vector<16xi32>) {
  %x = arith.bitcast %bits : vector<16xi32> to vector<16xf32>
  %0 = arith.truncf %x : vector<16xf32> to vector<16xf8E4M3FNUZ>
  func.call @print_f8E4M3FNUZ(%0) : (vector<16xf8E4M3FNUZ>) -> ()
  %1 = arith.truncf %x to_nearest_away : vector<16xf32> to vector<16xf8E4M3FNUZ>
  func.call @print_f8E4M3FNUZ(%1) : (vector<16xf8E4M3FNUZ>) -> ()
  %2 = arith.truncf %x toward_zero : vector<16xf32> to vector<16xf8E4M3FNUZ>
  func.call @print_f8E4M3FNUZ(%2) : (vector<16xf8E4M3FNUZ>) -> ()
  %3 = arith.truncf %x upward : vector<16xf32> to vector<16xf8E4M3FNUZ>
  func.call @print_f8E4M3FNUZ(%3) : (vector<16xf8E4M3FNUZ>) -> ()
  %4 = arith.truncf %x downward : vector<16xf32> to vector<16xf8E4M3FNUZ>
  func.call @print_f8E4M3FNUZ(%4) : (vector<16xf8E4M3FNUZ>) -> ()
  return
}

func.func @truncf_f32_to_f8E4M3B11FNUZ(%bits : vector<16xi32>) {
  %x = arith.bitcast %bits : vector<16xi32> to vector<16xf32>
  %0 = arith.truncf %x : vector<16xf32> to vector<16xf8E4M3B11FNUZ>
  func.call @print_f8E4M3B11FNUZ(%0) : (vector<16xf8E4M3B11FNUZ>) -> ()
  %1 = arith.truncf %x to_nearest_away
      : vector<16xf32> to vector<16xf8E4M3B11FNUZ>
  func.call @print_f8E4M3B11FNUZ(%1) : (vector<16xf8E4M3B11FNUZ>) -> ()
  %2 = arith.truncf %x toward_zero : vector<16xf32> to vector<16xf8E4M3B11FNUZ>
  func.call @print_f8E4M3B11FNUZ(%2) : (vector<16xf8E4M3B11FNUZ>) -> ()
  %3 = arith.truncf %x upward : vector<16xf32> to vector<16xf8E4M3B11FNUZ>
  func.call @print_f8E4M3B11FNUZ(%3) : (vector<16xf8E4M3B11FNUZ>) -> ()
  %4 = arith.truncf %x downward : vector<16xf32> to vector<16xf8E4M3B11FNUZ>
  func.call @print_f8E4M3B11FNUZ(%4) : (vector<16xf8E4M3B11FNUZ>) -> ()
  return
}

func.func @truncf_f32_to_f8E5M2FNUZ(%bits : vector<16xi32>) {
  %x = arith.bitcast %bits : vector<16xi32> to vector<16xf32>
  %0 = arith.truncf %x : vector<16xf32> to vector<16xf8E5M2FNUZ>
  func.call @print_f8E5M2FNUZ(%0) : (vector<16xf8E5M2FNUZ>) -> ()
  %1 = arith.truncf %x to_nearest_away : vector<16xf32> to vector<16xf8E5M2FNUZ>
  func.call @print_f8E5M2FNUZ(%1) : (vector<16xf8E5M2FNUZ>) -> ()
  %2 = arith.truncf %x toward_zero : vector<16xf32> to vector<16xf8E5M2FNUZ>
  func.call @print_f8E5M2FNUZ(%2) : (vector<16xf8E5M2FNUZ>) -> ()
  %3 = arith.truncf %x upward : vector<16xf32> to vector<16xf8E5M2FNUZ>
  func.call @print_f8E5M2FNUZ(%3) : (vector<16xf8E5M2FNUZ>) -> ()
  %4 = arith.truncf %x downward : vector<16xf32> to vector<16xf8E5M2FNUZ>
  func.call @print_f8E5M2FNUZ(%4) : (vector<16xf8E5M2FNUZ>) -> ()
  return
}

func.func @truncf_f16_to_f8E5M2FNUZ(%bits : vector<16xi16>) {
  %x = arith.bitcast %bits : vector<16xi16> to vector<16xf16>
  %0 = arith.truncf %x to_nearest_even : vector<16xf16> to vector<16xf8E5M2FNUZ>
  func.call @print_f8E5M2FNUZ(%0) : (vector<16xf8E5M2FNUZ>) -> ()
  %1 = arith.truncf %x upward : vector<16xf16> to vector<16xf8E5M2FNUZ>
  func.call @print_f8E5M2FNUZ(%1) : (vector<16xf8E5M2FNUZ>) -> ()
  %2 = arith.truncf %x downward : vector<16xf16> to vector<16xf8E5M2FNUZ>
  func.call @print_f8E5M2FNUZ(%2) : (vector<16xf8E5M2FNUZ>) -> ()
  return
}

func.func @truncf_f16_to_f8E4M3B11FNUZ(%bits : vector<16xi16>) {
  %x = arith.bitcast %bits : vector<16xi16> to vector<16xf16>
  %0 = arith.truncf %x to_nearest_even
      : vector<16xf16> to vector<16xf8E4M3B11FNUZ>
  func.call @print_f8E4M3B11FNUZ(%0) : (vector<16xf8E4M3B11FNUZ>) -> ()
  %1 = arith.truncf %x to_nearest_away
      : vector<16xf16> to vector<16xf8E4M3B11FNUZ>
  func.call @print_f8E4M3B11FNUZ(%1) : (vector<16xf8E4M3B11FNUZ>) -> ()
  %2 = arith.truncf %x toward_zero : vector<16xf16> to vector<16xf8E4M3B11FNUZ>
  func.call @print_f8E4M3B11FNUZ(%2) : (vector<16xf8E4M3B11FNUZ>) -> ()
  %3 = arith.truncf %x upward : vector<16xf16> to vector<16xf8E4M3B11FNUZ>
  func.call @print_f8E4M3B11FNUZ(%3) : (vector<16xf8E4M3B11FNUZ>) -> ()
  %4 = arith.truncf %x downward : vector<16xf16> to vector<16xf8E4M3B11FNUZ>
  func.call @print_f8E4M3B11FNUZ(%4) : (vector<16xf8E4M3B11FNUZ>) -> ()
  return
}

func.func @truncf_bf16_to_f6E2M3FN(%bits : vector<16xi16>) {
  %x = arith.bitcast %bits : vector<16xi16> to vector<16xbf16>
  %0 = arith.truncf %x to_nearest_even : vector<16xbf16> to vector<16xf6E2M3FN>
  func.call @print_f6E2M3FN(%0) : (vector<16xf6E2M3FN>) -> ()
  %1 = arith.truncf %x upward : vector<16xbf16> to vector<16xf6E2M3FN>
  func.call @print_f6E2M3FN(%1) : (vector<16xf6E2M3FN>) -> ()
  %2 = arith.truncf %x downward : vector<16xbf16> to vector<16xf6E2M3FN>
  func.call @print_f6E2M3FN(%2) : (vector<16xf6E2M3FN>) -> ()
  return
}

func.func @extf_f8E4M3(%codes : vector<8xi8>) {
  %x = arith.bitcast %codes : vector<8xi8> to vector<8xf8E4M3>
  %0 = arith.extf %x : vector<8xf8E4M3> to vector<8xf32>
  func.call @print_f32(%0) : (vector<8xf32>) -> ()
  %1 = arith.extf %x : vector<8xf8E4M3> to vector<8xf16>
  func.call @print_f16(%1) : (vector<8xf16>) -> ()
  %2 = arith.extf %x : vector<8xf8E4M3> to vector<8xbf16>
  func.call @print_bf16(%2) : (vector<8xbf16>) -> ()
  %3 = arith.extf %x : vector<8xf8E4M3> to vector<8xf64>
  func.call @print_f64(%3) : (vector<8xf64>) -> ()
  return
}

func.func @extf_f8E3M4(%codes : vector<8xi8>) {
  %x = arith.bitcast %codes : vector<8xi8> to vector<8xf8E3M4>
  %0 = arith.extf %x : vector<8xf8E3M4> to vector<8xf32>
  func.call @print_f32(%0) : (vector<8xf32>) -> ()
  %1 = arith.extf %x : vector<8xf8E3M4> to vector<8xf16>
  func.call @print_f16(%1) : (vector<8xf16>) -> ()
  %2 = arith.extf %x : vector<8xf8E3M4> to vector<8xbf16>
  func.call @print_bf16(%2) : (vector<8xbf16>) -> ()
  %3 = arith.extf %x : vector<8xf8E3M4> to vector<8xf64>
  func.call @print_f64(%3) : (vector<8xf64>) -> ()
  return
}

func.func @extf_f8E4M3FNUZ(%codes : vector<8xi8>) {
  %x = arith.bitcast %codes : vector<8xi8> to vector<8xf8E4M3FNUZ>
  %0 = arith.extf %x : vector<8xf8E4M3FNUZ> to vector<8xf32>
  func.call @print_f32(%0) : (vector<8xf32>) -> ()
  %1 = arith.extf %x : vector<8xf8E4M3FNUZ> to vector<8xf16>
  func.call @print_f16(%1) : (vector<8xf16>) -> ()
  %2 = arith.extf %x : vector<8xf8E4M3FNUZ> to vector<8xbf16>
  func.call @print_bf16(%2) : (vector<8xbf16>) -> ()
  %3 = arith.extf %x : vector<8xf8E4M3FNUZ> to vector<8xf64>
  func.call @print_f64(%3) : (vector<8xf64>) -> ()
  return
}

func.func @extf_f8E5M2FNUZ(%codes : vector<8xi8>) {
  %x = arith.bitcast %codes : vector<8xi8> to vector<8xf8E5M2FNUZ>
  %0 = arith.extf %x : vector<8xf8E5M2FNUZ> to vector<8xf32>
  func.call @print_f32(%0) : (vector<8xf32>) -> ()
  %1 = arith.extf %x : vector<8xf8E5M2FNUZ> to vector<8xf16>
  func.call @print_f16(%1) : (vector<8xf16>) -> ()
  %2 = arith.extf %x : vector<8xf8E5M2FNUZ> to vector<8xbf16>
  func.call @print_bf16(%2) : (vector<8xbf16>) -> ()
  %3 = arith.extf %x : vector<8xf8E5M2FNUZ> to vector<8xf64>
  func.call @print_f64(%3) : (vector<8xf64>) -> ()
  return
}

func.func @extf_f8E4M3B11FNUZ(%codes : vector<8xi8>) {
  %x = arith.bitcast %codes : vector<8xi8> to vector<8xf8E4M3B11FNUZ>
  %0 = arith.extf %x : vector<8xf8E4M3B11FNUZ> to vector<8xf32>
  func.call @print_f32(%0) : (vector<8xf32>) -> ()
  %1 = arith.extf %x : vector<8xf8E4M3B11FNUZ> to vector<8xf16>
  func.call @print_f16(%1) : (vector<8xf16>) -> ()
  %2 = arith.extf %x : vector<8xf8E4M3B11FNUZ> to vector<8xbf16>
  func.call @print_bf16(%2) : (vector<8xbf16>) -> ()
  %3 = arith.extf %x : vector<8xf8E4M3B11FNUZ> to vector<8xf64>
  func.call @print_f64(%3) : (vector<8xf64>) -> ()
  return
}

func.func @extf_f6E2M3FN(%codes : vector<8xi8>) {
  %narrow = arith.trunci %codes : vector<8xi8> to vector<8xi6>
  %x = arith.bitcast %narrow : vector<8xi6> to vector<8xf6E2M3FN>
  %0 = arith.extf %x : vector<8xf6E2M3FN> to vector<8xf32>
  func.call @print_f32(%0) : (vector<8xf32>) -> ()
  %1 = arith.extf %x : vector<8xf6E2M3FN> to vector<8xf16>
  func.call @print_f16(%1) : (vector<8xf16>) -> ()
  %2 = arith.extf %x : vector<8xf6E2M3FN> to vector<8xbf16>
  func.call @print_bf16(%2) : (vector<8xbf16>) -> ()
  %3 = arith.extf %x : vector<8xf6E2M3FN> to vector<8xf64>
  func.call @print_f64(%3) : (vector<8xf64>) -> ()
  return
}

func.func @extf_f6E3M2FN(%codes : vector<8xi8>) {
  %narrow = arith.trunci %codes : vector<8xi8> to vector<8xi6>
  %x = arith.bitcast %narrow : vector<8xi6> to vector<8xf6E3M2FN>
  %0 = arith.extf %x : vector<8xf6E3M2FN> to vector<8xf32>
  func.call @print_f32(%0) : (vector<8xf32>) -> ()
  %1 = arith.extf %x : vector<8xf6E3M2FN> to vector<8xf16>
  func.call @print_f16(%1) : (vector<8xf16>) -> ()
  %2 = arith.extf %x : vector<8xf6E3M2FN> to vector<8xbf16>
  func.call @print_bf16(%2) : (vector<8xbf16>) -> ()
  %3 = arith.extf %x : vector<8xf6E3M2FN> to vector<8xf64>
  func.call @print_f64(%3) : (vector<8xf64>) -> ()
  return
}

func.func @entry() {
  // truncf from f32, one line per mode (nearest even, nearest away, toward
  // zero, upward, downward). With u = 2^-p for p mantissa bits, s the smallest
  // subnormal and M the largest finite value, the elements are
  //   0-7:  1+u/2, -(1+3u/2) (ties to the even neighbor below and above),
  //         1+u/2+2^-20 (just above a tie), -(1+u/4), 1.5s (subnormal tie),
  //         -s/2 (tie with -0), s/4, -0;
  //   8-15: M, M-ulp(M)/2 and M+ulp(M)/2 (ties next to M), -(M+3ulp(M)/4),
  //         +Inf, -Inf, quiet NaN 0x7fe00000, signaling NaN 0xff800001.

  // f6E2M3FN: u = 2^-3, s = 2^-3, M = 7.5 (0x1f); no Inf or NaN.
  %to_f6E2M3FN = arith.constant dense<[
      0x3F880000, 0xBF980000, 0x3F880008, 0xBF840000,
      0x3E400000, 0xBD800000, 0x3D000000, 0x80000000,
      0x40F00000, 0x40E80000, 0x40F80000, 0xC0FC0000,
      0x7F800000, 0xFF800000, 0x7FE00000, 0xFF800001]> : vector<16xi32>
  // CHECK:      ( 8, 42, 9, 40, 2, 32, 0, 32, 31, 30, 31, 63, 31, 63, 31, 31 )
  // CHECK-NEXT: ( 9, 42, 9, 40, 2, 33, 0, 32, 31, 31, 31, 63, 31, 63, 31, 31 )
  // CHECK-NEXT: ( 8, 41, 8, 40, 1, 32, 0, 32, 31, 30, 31, 63, 31, 63, 31, 31 )
  // CHECK-NEXT: ( 9, 41, 9, 40, 2, 32, 1, 32, 31, 31, 31, 63, 31, 63, 31, 31 )
  // CHECK-NEXT: ( 8, 42, 8, 41, 1, 33, 0, 32, 31, 30, 31, 63, 31, 63, 31, 31 )
  func.call @truncf_f32_to_f6E2M3FN(%to_f6E2M3FN) : (vector<16xi32>) -> ()

  // f6E3M2FN: u = 2^-2, s = 2^-4, M = 28 (0x1f); no Inf or NaN.
  %to_f6E3M2FN = arith.constant dense<[
      0x3F900000, 0xBFB00000, 0x3F900008, 0xBF880000,
      0x3DC00000, 0xBD000000, 0x3C800000, 0x80000000,
      0x41E00000, 0x41D00000, 0x41F00000, 0xC1F80000,
      0x7F800000, 0xFF800000, 0x7FE00000, 0xFF800001]> : vector<16xi32>
  // CHECK-NEXT: ( 12, 46, 13, 44, 2, 32, 0, 32, 31, 30, 31, 63, 31, 63, 31, 31 )
  // CHECK-NEXT: ( 13, 46, 13, 44, 2, 33, 0, 32, 31, 31, 31, 63, 31, 63, 31, 31 )
  // CHECK-NEXT: ( 12, 45, 12, 44, 1, 32, 0, 32, 31, 30, 31, 63, 31, 63, 31, 31 )
  // CHECK-NEXT: ( 13, 45, 13, 44, 2, 32, 1, 32, 31, 31, 31, 63, 31, 63, 31, 31 )
  // CHECK-NEXT: ( 12, 46, 12, 45, 1, 33, 0, 32, 31, 30, 31, 63, 31, 63, 31, 31 )
  func.call @truncf_f32_to_f6E3M2FN(%to_f6E3M2FN) : (vector<16xi32>) -> ()

  // f8E3M4: u = 2^-4, s = 2^-6, M = 15.5 (0x6f); Inf 0x70, quiet NaN 0x78.
  %to_f8E3M4 = arith.constant dense<[
      0x3F840000, 0xBF8C0000, 0x3F840008, 0xBF820000,
      0x3CC00000, 0xBC000000, 0x3B800000, 0x80000000,
      0x41780000, 0x41740000, 0x417C0000, 0xC17E0000,
      0x7F800000, 0xFF800000, 0x7FE00000, 0xFF800001]> : vector<16xi32>
  // CHECK-NEXT: ( 48, 178, 49, 176, 2, 128, 0, 128, 111, 110, 112, 240, 112, 240, 124, 248 )
  // CHECK-NEXT: ( 49, 178, 49, 176, 2, 129, 0, 128, 111, 111, 112, 240, 112, 240, 124, 248 )
  // CHECK-NEXT: ( 48, 177, 48, 176, 1, 128, 0, 128, 111, 110, 111, 239, 112, 240, 124, 248 )
  // CHECK-NEXT: ( 49, 177, 49, 176, 2, 128, 1, 128, 111, 111, 112, 239, 112, 240, 124, 248 )
  // CHECK-NEXT: ( 48, 178, 48, 177, 1, 129, 0, 128, 111, 110, 111, 240, 112, 240, 124, 248 )
  func.call @truncf_f32_to_f8E3M4(%to_f8E3M4) : (vector<16xi32>) -> ()

  // f8E4M3: u = 2^-3, s = 2^-9, M = 240 (0x77); Inf 0x78, quiet NaN 0x7c.
  %to_f8E4M3 = arith.constant dense<[
      0x3F880000, 0xBF980000, 0x3F880008, 0xBF840000,
      0x3B400000, 0xBA800000, 0x3A000000, 0x80000000,
      0x43700000, 0x43680000, 0x43780000, 0xC37C0000,
      0x7F800000, 0xFF800000, 0x7FE00000, 0xFF800001]> : vector<16xi32>
  // CHECK-NEXT: ( 56, 186, 57, 184, 2, 128, 0, 128, 119, 118, 120, 248, 120, 248, 126, 252 )
  // CHECK-NEXT: ( 57, 186, 57, 184, 2, 129, 0, 128, 119, 119, 120, 248, 120, 248, 126, 252 )
  // CHECK-NEXT: ( 56, 185, 56, 184, 1, 128, 0, 128, 119, 118, 119, 247, 120, 248, 126, 252 )
  // CHECK-NEXT: ( 57, 185, 57, 184, 2, 128, 1, 128, 119, 119, 120, 247, 120, 248, 126, 252 )
  // CHECK-NEXT: ( 56, 186, 56, 185, 1, 129, 0, 128, 119, 118, 119, 248, 120, 248, 126, 252 )
  func.call @truncf_f32_to_f8E4M3(%to_f8E4M3) : (vector<16xi32>) -> ()

  // f8E4M3FNUZ: u = 2^-3, s = 2^-10, M = 240 (0x7f); no Inf or -0, the only NaN
  // is 0x80.
  %to_f8E4M3FNUZ = arith.constant dense<[
      0x3F880000, 0xBF980000, 0x3F880008, 0xBF840000,
      0x3AC00000, 0xBA000000, 0x39800000, 0x80000000,
      0x43700000, 0x43680000, 0x43780000, 0xC37C0000,
      0x7F800000, 0xFF800000, 0x7FE00000, 0xFF800001]> : vector<16xi32>
  // CHECK-NEXT: ( 64, 194, 65, 192, 2, 0, 0, 0, 127, 126, 128, 128, 128, 128, 128, 128 )
  // CHECK-NEXT: ( 65, 194, 65, 192, 2, 129, 0, 0, 127, 127, 128, 128, 128, 128, 128, 128 )
  // CHECK-NEXT: ( 64, 193, 64, 192, 1, 0, 0, 0, 127, 126, 127, 255, 128, 128, 128, 128 )
  // CHECK-NEXT: ( 65, 193, 65, 192, 2, 0, 1, 0, 127, 127, 128, 255, 128, 128, 128, 128 )
  // CHECK-NEXT: ( 64, 194, 64, 193, 1, 129, 0, 0, 127, 126, 127, 128, 128, 128, 128, 128 )
  func.call @truncf_f32_to_f8E4M3FNUZ(%to_f8E4M3FNUZ) : (vector<16xi32>) -> ()

  // f8E4M3B11FNUZ: u = 2^-3, s = 2^-13, M = 30 (0x7f); no Inf or -0, the only
  // NaN is 0x80.
  %to_f8E4M3B11FNUZ = arith.constant dense<[
      0x3F880000, 0xBF980000, 0x3F880008, 0xBF840000,
      0x39400000, 0xB8800000, 0x38000000, 0x80000000,
      0x41F00000, 0x41E80000, 0x41F80000, 0xC1FC0000,
      0x7F800000, 0xFF800000, 0x7FE00000, 0xFF800001]> : vector<16xi32>
  // CHECK-NEXT: ( 88, 218, 89, 216, 2, 0, 0, 0, 127, 126, 128, 128, 128, 128, 128, 128 )
  // CHECK-NEXT: ( 89, 218, 89, 216, 2, 129, 0, 0, 127, 127, 128, 128, 128, 128, 128, 128 )
  // CHECK-NEXT: ( 88, 217, 88, 216, 1, 0, 0, 0, 127, 126, 127, 255, 128, 128, 128, 128 )
  // CHECK-NEXT: ( 89, 217, 89, 216, 2, 0, 1, 0, 127, 127, 128, 255, 128, 128, 128, 128 )
  // CHECK-NEXT: ( 88, 218, 88, 217, 1, 129, 0, 0, 127, 126, 127, 128, 128, 128, 128, 128 )
  func.call @truncf_f32_to_f8E4M3B11FNUZ(%to_f8E4M3B11FNUZ)
      : (vector<16xi32>) -> ()

  // f8E5M2FNUZ: u = 2^-2, s = 2^-17, M = 57344 (0x7f); no Inf or -0, the only
  // NaN is 0x80.
  %to_f8E5M2FNUZ = arith.constant dense<[
      0x3F900000, 0xBFB00000, 0x3F900008, 0xBF880000,
      0x37400000, 0xB6800000, 0x36000000, 0x80000000,
      0x47600000, 0x47500000, 0x47700000, 0xC7780000,
      0x7F800000, 0xFF800000, 0x7FE00000, 0xFF800001]> : vector<16xi32>
  // CHECK-NEXT: ( 64, 194, 65, 192, 2, 0, 0, 0, 127, 126, 128, 128, 128, 128, 128, 128 )
  // CHECK-NEXT: ( 65, 194, 65, 192, 2, 129, 0, 0, 127, 127, 128, 128, 128, 128, 128, 128 )
  // CHECK-NEXT: ( 64, 193, 64, 192, 1, 0, 0, 0, 127, 126, 127, 255, 128, 128, 128, 128 )
  // CHECK-NEXT: ( 65, 193, 65, 192, 2, 0, 1, 0, 127, 127, 128, 255, 128, 128, 128, 128 )
  // CHECK-NEXT: ( 64, 194, 64, 193, 1, 129, 0, 0, 127, 126, 127, 128, 128, 128, 128, 128 )
  func.call @truncf_f32_to_f8E5M2FNUZ(%to_f8E5M2FNUZ) : (vector<16xi32>) -> ()

  // truncf from f16 and bf16: to_nearest_even, upward and downward, or all
  // five modes (the f8E4M3B11FNUZ helper).

  // f16 -> f8E5M2FNUZ (bias 16 > 15: f16 subnormals in [2^-15, 2^-14) become
  // normals): 1.125, -1.375 (ties), 2^-18 (half the smallest subnormal),
  // 1.25*2^-18, 1.5*2^-17 (tie), 2^-16, 2^-15, -1.5*2^-15, -(2^-14-2^-24),
  // -2^-24, 57344 (max), +-61440 (ties above max), +Inf, -0, -NaN.
  %f16_to_f8E5M2FNUZ = arith.constant dense<[
      0x3C80, 0xBD80, 0x0040, 0x0050,
      0x00C0, 0x0100, 0x0200, 0x8300,
      0x83FF, 0x8001, 0x7B00, 0x7B80,
      0xFB80, 0x7C00, 0x8000, 0xFE00]> : vector<16xi16>
  // CHECK-NEXT: ( 64, 194, 0, 1, 2, 2, 4, 134, 136, 0, 127, 128, 128, 128, 0, 128 )
  // CHECK-NEXT: ( 65, 193, 1, 1, 2, 2, 4, 134, 135, 0, 127, 128, 255, 128, 0, 128 )
  // CHECK-NEXT: ( 64, 194, 0, 0, 1, 2, 4, 134, 136, 129, 127, 127, 128, 128, 0, 128 )
  func.call @truncf_f16_to_f8E5M2FNUZ(%f16_to_f8E5M2FNUZ)
      : (vector<16xi16>) -> ()

  // f16 -> f8E4M3B11FNUZ, in every mode: every f16 subnormal lies below
  // half the smallest destination subnormal 2^-13. +-0, 2^-24, 2^-15,
  // 2^-14-2^-24, -2^-24, -(2^-14-2^-24), 2^-14 (tie with 0), 2^-14+2^-24,
  // 1.5*2^-14, 2^-12, 3*2^-13, 1, 30 (max), +Inf, -NaN.
  %f16_to_f8E4M3B11FNUZ = arith.constant dense<[
      0x0000, 0x8000, 0x0001, 0x0200,
      0x03FF, 0x8001, 0x83FF, 0x0400,
      0x0401, 0x0600, 0x0C00, 0x0E00,
      0x3C00, 0x4F80, 0x7C00, 0xFE00]> : vector<16xi16>
  // CHECK-NEXT: ( 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 3, 88, 127, 128, 128 )
  // CHECK-NEXT: ( 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 3, 88, 127, 128, 128 )
  // CHECK-NEXT: ( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 88, 127, 128, 128 )
  // CHECK-NEXT: ( 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 2, 3, 88, 127, 128, 128 )
  // CHECK-NEXT: ( 0, 0, 0, 0, 0, 129, 129, 0, 0, 0, 2, 3, 88, 127, 128, 128 )
  func.call @truncf_f16_to_f8E4M3B11FNUZ(%f16_to_f8E4M3B11FNUZ)
      : (vector<16xi16>) -> ()

  // bf16 -> f6E2M3FN (no Inf or NaN): -Inf, +Inf, -NaN, +NaN (quiet),
  // -NaN, +NaN (signaling), -7.5 (max), -8, -7.75 (tie above max),
  // -2^100, -0, 1, 7.25, -2.125, 0.0625 (ties), 0.1875 (tie).
  %bf16_to_f6E2M3FN = arith.constant dense<[
      0xFF80, 0x7F80, 0xFFC0, 0x7FC0,
      0xFF81, 0x7F81, 0xC0F0, 0xC100,
      0xC0F8, 0xF180, 0x8000, 0x3F80,
      0x40E8, 0xC008, 0x3D80, 0x3E40]> : vector<16xi16>
  // CHECK-NEXT: ( 63, 31, 31, 31, 31, 31, 63, 63, 63, 63, 32, 8, 30, 48, 0, 2 )
  // CHECK-NEXT: ( 63, 31, 31, 31, 31, 31, 63, 63, 63, 63, 32, 8, 31, 48, 1, 2 )
  // CHECK-NEXT: ( 63, 31, 31, 31, 31, 31, 63, 63, 63, 63, 32, 8, 30, 49, 0, 1 )
  func.call @truncf_bf16_to_f6E2M3FN(%bf16_to_f6E2M3FN) : (vector<16xi16>) -> ()

  // extf from each format; each @extf_* prints the f32, f16, bf16 and f64
  // results.

  // f8E4M3: 2^-9, 2^-6, 240, -240, +Inf, -Inf, sNaN 0x79, -qNaN 0xfc.
  %from_f8E4M3 = arith.constant dense<[
      0x01, 0x08, 0x77, 0xF7, 0x78, 0xF8, 0x79, 0xFC]> : vector<8xi8>
  // CHECK-NEXT: ( 989855744, 1015021568, 1131413504, 3278897152, 2139095040, 4286578688, 2144337920, 4290772992 )
  // CHECK-NEXT: ( 6144, 9216, 23424, 56192, 31744, 64512, 32384, 65024 )
  // CHECK-NEXT: ( 15104, 15488, 17264, 50032, 32640, 65408, 32720, 65472 )
  // CHECK-NEXT: ( 4566650022153682944, 4580160821035794432, 4642648265865560064, 13866020302720335872, 9218868437227405312, 18442240474082181120, 9221683186994511872, 18444492273895866368 )
  func.call @extf_f8E4M3(%from_f8E4M3) : (vector<8xi8>) -> ()

  // f8E3M4: 2^-6, 2^-2, 15.5, -0, -Inf, +Inf, sNaN 0x71, -qNaN 0xf9.
  %from_f8E3M4 = arith.constant dense<[
      0x01, 0x10, 0x6F, 0x80, 0xF0, 0x70, 0x71, 0xF9]> : vector<8xi8>
  // CHECK-NEXT: ( 1015021568, 1048576000, 1098383360, 2147483648, 4286578688, 2139095040, 2143813632, 4291297280 )
  // CHECK-NEXT: ( 9216, 13312, 19392, 32768, 64512, 31744, 32320, 65088 )
  // CHECK-NEXT: ( 15488, 16000, 16760, 32768, 65408, 32640, 32712, 65480 )
  // CHECK-NEXT: ( 4580160821035794432, 4598175219545276416, 4624915342332788736, 9223372036854775808, 18442240474082181120, 9218868437227405312, 9221401712017801216, 18444773748872577024 )
  func.call @extf_f8E3M4(%from_f8E3M4) : (vector<8xi8>) -> ()

  // f8E4M3FNUZ: 2^-10, 2^-7, 1, 240, -240, 0, NaN, -1.5.
  %from_f8E4M3FNUZ = arith.constant dense<[
      0x01, 0x08, 0x40, 0x7F, 0xFF, 0x00, 0x80, 0xC4]> : vector<8xi8>
  // CHECK-NEXT: ( 981467136, 1006632960, 1065353216, 1131413504, 3278897152, 0, 2143289344, 3217031168 )
  // CHECK-NEXT: ( 5120, 8192, 15360, 23424, 56192, 0, 32256, 48640 )
  // CHECK-NEXT: ( 14976, 15360, 16256, 17264, 50032, 0, 32704, 49088 )
  // CHECK-NEXT: ( 4562146422526312448, 4575657221408423936, 4607182418800017408, 4642648265865560064, 13866020302720335872, 0, 9221120237041090560, 13832806255468478464 )
  func.call @extf_f8E4M3FNUZ(%from_f8E4M3FNUZ) : (vector<8xi8>) -> ()

  // f8E5M2FNUZ: 2^-17, 3*2^-17, 2^-15, 57344, -57344, 0, NaN, -0.625.
  %from_f8E5M2FNUZ = arith.constant dense<[
      0x01, 0x03, 0x04, 0x7F, 0xFF, 0x00, 0x80, 0xBD]> : vector<8xi8>
  // CHECK-NEXT: ( 922746880, 935329792, 939524096, 1197473792, 3344957440, 0, 2143289344, 3206545408 )
  // CHECK-NEXT: ( 128, 384, 512, 31488, 64256, 0, 32256, 47360 )
  // CHECK-NEXT: ( 14080, 14272, 14336, 18272, 51040, 0, 32704, 48928 )
  // CHECK-NEXT: ( 4530621225134718976, 4537376624575774720, 4539628424389459968, 4678114112931102720, 13901486149785878528, 0, 9221120237041090560, 13827176755934265344 )
  func.call @extf_f8E5M2FNUZ(%from_f8E5M2FNUZ) : (vector<8xi8>) -> ()

  // f8E4M3B11FNUZ: 2^-13, 2^-10, 1, 30, -30, 0, NaN, -1.125.
  %from_f8E4M3B11FNUZ = arith.constant dense<[
      0x01, 0x08, 0x58, 0x7F, 0xFF, 0x00, 0x80, 0xD9]> : vector<8xi8>
  // CHECK-NEXT: ( 956301312, 981467136, 1065353216, 1106247680, 3253731328, 0, 2143289344, 3213885440 )
  // CHECK-NEXT: ( 2048, 5120, 15360, 20352, 53120, 0, 32256, 48256 )
  // CHECK-NEXT: ( 14592, 14976, 16256, 16880, 49648, 0, 32704, 49040 )
  // CHECK-NEXT: ( 4548635623644200960, 4562146422526312448, 4607182418800017408, 4629137466983448576, 13852509503838224384, 0, 9221120237041090560, 13831117405608214528 )
  func.call @extf_f8E4M3B11FNUZ(%from_f8E4M3B11FNUZ) : (vector<8xi8>) -> ()

  // f6E2M3FN: 0.125, 0.875, 1, 7.5, -7.5, -0, -0.125, 2.75.
  %from_f6E2M3FN = arith.constant dense<[
      0x01, 0x07, 0x08, 0x1F, 0x3F, 0x20, 0x21, 0x13]> : vector<8xi8>
  // CHECK-NEXT: ( 1040187392, 1063256064, 1065353216, 1089470464, 3236954112, 2147483648, 3187671040, 1076887552 )
  // CHECK-NEXT: ( 12288, 15104, 15360, 18304, 51072, 32768, 45056, 16768 )
  // CHECK-NEXT: ( 15872, 16224, 16256, 16624, 49392, 32768, 48640, 16432 )
  // CHECK-NEXT: ( 4593671619917905920, 4606056518893174784, 4607182418800017408, 4620130267728707584, 13843502304583483392, 9223372036854775808, 13817043656772681728, 4613374868287651840 )
  func.call @extf_f6E2M3FN(%from_f6E2M3FN) : (vector<8xi8>) -> ()

  // f6E3M2FN: 0.0625, 0.1875, 0.25, 28, -28, -0, -0.1875, 1.25.
  %from_f6E3M2FN = arith.constant dense<[
      0x01, 0x03, 0x04, 0x1F, 0x3F, 0x20, 0x23, 0x0D]> : vector<8xi8>
  // CHECK-NEXT: ( 1031798784, 1044381696, 1048576000, 1105199104, 3252682752, 2147483648, 3191865344, 1067450368 )
  // CHECK-NEXT: ( 11264, 12800, 13312, 20224, 52992, 32768, 45568, 15616 )
  // CHECK-NEXT: ( 15744, 15936, 16000, 16864, 49632, 32768, 48704, 16288 )
  // CHECK-NEXT: ( 4589168020290535424, 4595923419731591168, 4598175219545276416, 4628574517030027264, 13851946553884803072, 9223372036854775808, 13819295456586366976, 4608308318706860032 )
  func.call @extf_f6E3M2FN(%from_f6E3M2FN) : (vector<8xi8>) -> ()
  return
}
