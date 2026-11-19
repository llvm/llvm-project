// Value tests for the narrow-float truncf/extf expansion. Expected codes come
// from an independent model of APFloat::convert (formats without Inf or NaN
// saturate); inputs are runtime arguments, so nothing is constant folded.

// RUN: mlir-opt %s --arith-expand="include-f4e2m1=true include-f6e2m3fn=true \
// RUN:     include-f6e3m2fn=true include-f8e3m4=true include-f8e4m3=true \
// RUN:     include-f8e4m3b11fnuz=true include-f8e4m3fn=true \
// RUN:     include-f8e4m3fnuz=true include-f8e5m2=true \
// RUN:     include-f8e5m2fnuz=true include-f8e8m0=true" \
// RUN:     --convert-vector-to-scf --convert-scf-to-cf --convert-cf-to-llvm \
// RUN:     --convert-vector-to-llvm --convert-func-to-llvm \
// RUN:     --convert-arith-to-llvm --reconcile-unrealized-casts | \
// RUN:   mlir-runner -e entry --entry-point-result=void \
// RUN:               --shared-libs=%mlir_c_runner_utils | \
// RUN:   FileCheck %s --match-full-lines

// Prints the codes of a vector as unsigned integers.
func.func @print_f4E2M1FN(%v : vector<16xf4E2M1FN>) {
  %bits = arith.bitcast %v : vector<16xf4E2M1FN> to vector<16xi4>
  %u = vector.bitcast %bits : vector<16xi4> to vector<16xui4>
  vector.print %u : vector<16xui4>
  return
}

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

func.func @print_f8E4M3FN(%v : vector<16xf8E4M3FN>) {
  %bits = arith.bitcast %v : vector<16xf8E4M3FN> to vector<16xi8>
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

func.func @print_f8E5M2(%v : vector<16xf8E5M2>) {
  %bits = arith.bitcast %v : vector<16xf8E5M2> to vector<16xi8>
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

func.func @print_f8E8M0FNU(%v : vector<16xf8E8M0FNU>) {
  %bits = arith.bitcast %v : vector<16xf8E8M0FNU> to vector<16xi8>
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

func.func @truncf_f32_to_f4E2M1FN(%bits : vector<16xi32>) {
  %x = arith.bitcast %bits : vector<16xi32> to vector<16xf32>
  %0 = arith.truncf %x : vector<16xf32> to vector<16xf4E2M1FN>
  func.call @print_f4E2M1FN(%0) : (vector<16xf4E2M1FN>) -> ()
  %1 = arith.truncf %x to_nearest_away : vector<16xf32> to vector<16xf4E2M1FN>
  func.call @print_f4E2M1FN(%1) : (vector<16xf4E2M1FN>) -> ()
  %2 = arith.truncf %x toward_zero : vector<16xf32> to vector<16xf4E2M1FN>
  func.call @print_f4E2M1FN(%2) : (vector<16xf4E2M1FN>) -> ()
  %3 = arith.truncf %x upward : vector<16xf32> to vector<16xf4E2M1FN>
  func.call @print_f4E2M1FN(%3) : (vector<16xf4E2M1FN>) -> ()
  %4 = arith.truncf %x downward : vector<16xf32> to vector<16xf4E2M1FN>
  func.call @print_f4E2M1FN(%4) : (vector<16xf4E2M1FN>) -> ()
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

func.func @truncf_f32_to_f8E4M3FN(%bits : vector<16xi32>) {
  %x = arith.bitcast %bits : vector<16xi32> to vector<16xf32>
  %0 = arith.truncf %x : vector<16xf32> to vector<16xf8E4M3FN>
  func.call @print_f8E4M3FN(%0) : (vector<16xf8E4M3FN>) -> ()
  %1 = arith.truncf %x to_nearest_away : vector<16xf32> to vector<16xf8E4M3FN>
  func.call @print_f8E4M3FN(%1) : (vector<16xf8E4M3FN>) -> ()
  %2 = arith.truncf %x toward_zero : vector<16xf32> to vector<16xf8E4M3FN>
  func.call @print_f8E4M3FN(%2) : (vector<16xf8E4M3FN>) -> ()
  %3 = arith.truncf %x upward : vector<16xf32> to vector<16xf8E4M3FN>
  func.call @print_f8E4M3FN(%3) : (vector<16xf8E4M3FN>) -> ()
  %4 = arith.truncf %x downward : vector<16xf32> to vector<16xf8E4M3FN>
  func.call @print_f8E4M3FN(%4) : (vector<16xf8E4M3FN>) -> ()
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

func.func @truncf_f32_to_f8E5M2(%bits : vector<16xi32>) {
  %x = arith.bitcast %bits : vector<16xi32> to vector<16xf32>
  %0 = arith.truncf %x : vector<16xf32> to vector<16xf8E5M2>
  func.call @print_f8E5M2(%0) : (vector<16xf8E5M2>) -> ()
  %1 = arith.truncf %x to_nearest_away : vector<16xf32> to vector<16xf8E5M2>
  func.call @print_f8E5M2(%1) : (vector<16xf8E5M2>) -> ()
  %2 = arith.truncf %x toward_zero : vector<16xf32> to vector<16xf8E5M2>
  func.call @print_f8E5M2(%2) : (vector<16xf8E5M2>) -> ()
  %3 = arith.truncf %x upward : vector<16xf32> to vector<16xf8E5M2>
  func.call @print_f8E5M2(%3) : (vector<16xf8E5M2>) -> ()
  %4 = arith.truncf %x downward : vector<16xf32> to vector<16xf8E5M2>
  func.call @print_f8E5M2(%4) : (vector<16xf8E5M2>) -> ()
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

func.func @truncf_f16_to_f8E5M2(%bits : vector<16xi16>) {
  %x = arith.bitcast %bits : vector<16xi16> to vector<16xf16>
  %0 = arith.truncf %x to_nearest_even : vector<16xf16> to vector<16xf8E5M2>
  func.call @print_f8E5M2(%0) : (vector<16xf8E5M2>) -> ()
  %1 = arith.truncf %x upward : vector<16xf16> to vector<16xf8E5M2>
  func.call @print_f8E5M2(%1) : (vector<16xf8E5M2>) -> ()
  %2 = arith.truncf %x downward : vector<16xf16> to vector<16xf8E5M2>
  func.call @print_f8E5M2(%2) : (vector<16xf8E5M2>) -> ()
  return
}

func.func @truncf_f16_to_f8E5M2_modes(%bits : vector<16xi16>) {
  %x = arith.bitcast %bits : vector<16xi16> to vector<16xf16>
  %0 = arith.truncf %x to_nearest_even : vector<16xf16> to vector<16xf8E5M2>
  func.call @print_f8E5M2(%0) : (vector<16xf8E5M2>) -> ()
  %1 = arith.truncf %x to_nearest_away : vector<16xf16> to vector<16xf8E5M2>
  func.call @print_f8E5M2(%1) : (vector<16xf8E5M2>) -> ()
  %2 = arith.truncf %x toward_zero : vector<16xf16> to vector<16xf8E5M2>
  func.call @print_f8E5M2(%2) : (vector<16xf8E5M2>) -> ()
  %3 = arith.truncf %x upward : vector<16xf16> to vector<16xf8E5M2>
  func.call @print_f8E5M2(%3) : (vector<16xf8E5M2>) -> ()
  %4 = arith.truncf %x downward : vector<16xf16> to vector<16xf8E5M2>
  func.call @print_f8E5M2(%4) : (vector<16xf8E5M2>) -> ()
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

func.func @truncf_f16_to_f4E2M1FN(%bits : vector<16xi16>) {
  %x = arith.bitcast %bits : vector<16xi16> to vector<16xf16>
  %0 = arith.truncf %x to_nearest_even : vector<16xf16> to vector<16xf4E2M1FN>
  func.call @print_f4E2M1FN(%0) : (vector<16xf4E2M1FN>) -> ()
  %1 = arith.truncf %x upward : vector<16xf16> to vector<16xf4E2M1FN>
  func.call @print_f4E2M1FN(%1) : (vector<16xf4E2M1FN>) -> ()
  %2 = arith.truncf %x downward : vector<16xf16> to vector<16xf4E2M1FN>
  func.call @print_f4E2M1FN(%2) : (vector<16xf4E2M1FN>) -> ()
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

func.func @truncf_bf16_to_f8E5M2(%bits : vector<16xi16>) {
  %x = arith.bitcast %bits : vector<16xi16> to vector<16xbf16>
  %0 = arith.truncf %x to_nearest_even : vector<16xbf16> to vector<16xf8E5M2>
  func.call @print_f8E5M2(%0) : (vector<16xf8E5M2>) -> ()
  %1 = arith.truncf %x upward : vector<16xbf16> to vector<16xf8E5M2>
  func.call @print_f8E5M2(%1) : (vector<16xf8E5M2>) -> ()
  %2 = arith.truncf %x downward : vector<16xbf16> to vector<16xf8E5M2>
  func.call @print_f8E5M2(%2) : (vector<16xf8E5M2>) -> ()
  return
}

func.func @truncf_f64_to_f8E4M3FN(%bits : vector<16xi64>) {
  %x = arith.bitcast %bits : vector<16xi64> to vector<16xf64>
  %0 = arith.truncf %x to_nearest_even : vector<16xf64> to vector<16xf8E4M3FN>
  func.call @print_f8E4M3FN(%0) : (vector<16xf8E4M3FN>) -> ()
  %1 = arith.truncf %x upward : vector<16xf64> to vector<16xf8E4M3FN>
  func.call @print_f8E4M3FN(%1) : (vector<16xf8E4M3FN>) -> ()
  %2 = arith.truncf %x downward : vector<16xf64> to vector<16xf8E4M3FN>
  func.call @print_f8E4M3FN(%2) : (vector<16xf8E4M3FN>) -> ()
  return
}

func.func @truncf_f64_to_f4E2M1FN(%bits : vector<16xi64>) {
  %x = arith.bitcast %bits : vector<16xi64> to vector<16xf64>
  %0 = arith.truncf %x to_nearest_even : vector<16xf64> to vector<16xf4E2M1FN>
  func.call @print_f4E2M1FN(%0) : (vector<16xf4E2M1FN>) -> ()
  %1 = arith.truncf %x upward : vector<16xf64> to vector<16xf4E2M1FN>
  func.call @print_f4E2M1FN(%1) : (vector<16xf4E2M1FN>) -> ()
  %2 = arith.truncf %x downward : vector<16xf64> to vector<16xf4E2M1FN>
  func.call @print_f4E2M1FN(%2) : (vector<16xf4E2M1FN>) -> ()
  return
}

func.func @truncf_f128_to_f8E5M2(%bits : vector<16xi128>) {
  %x = arith.bitcast %bits : vector<16xi128> to vector<16xf128>
  %0 = arith.truncf %x to_nearest_even : vector<16xf128> to vector<16xf8E5M2>
  func.call @print_f8E5M2(%0) : (vector<16xf8E5M2>) -> ()
  %1 = arith.truncf %x to_nearest_away : vector<16xf128> to vector<16xf8E5M2>
  func.call @print_f8E5M2(%1) : (vector<16xf8E5M2>) -> ()
  %2 = arith.truncf %x toward_zero : vector<16xf128> to vector<16xf8E5M2>
  func.call @print_f8E5M2(%2) : (vector<16xf8E5M2>) -> ()
  %3 = arith.truncf %x upward : vector<16xf128> to vector<16xf8E5M2>
  func.call @print_f8E5M2(%3) : (vector<16xf8E5M2>) -> ()
  %4 = arith.truncf %x downward : vector<16xf128> to vector<16xf8E5M2>
  func.call @print_f8E5M2(%4) : (vector<16xf8E5M2>) -> ()
  return
}

func.func @truncf_f128_to_f8E4M3FN(%bits : vector<16xi128>) {
  %x = arith.bitcast %bits : vector<16xi128> to vector<16xf128>
  %0 = arith.truncf %x to_nearest_even : vector<16xf128> to vector<16xf8E4M3FN>
  func.call @print_f8E4M3FN(%0) : (vector<16xf8E4M3FN>) -> ()
  %1 = arith.truncf %x to_nearest_away : vector<16xf128> to vector<16xf8E4M3FN>
  func.call @print_f8E4M3FN(%1) : (vector<16xf8E4M3FN>) -> ()
  %2 = arith.truncf %x toward_zero : vector<16xf128> to vector<16xf8E4M3FN>
  func.call @print_f8E4M3FN(%2) : (vector<16xf8E4M3FN>) -> ()
  %3 = arith.truncf %x upward : vector<16xf128> to vector<16xf8E4M3FN>
  func.call @print_f8E4M3FN(%3) : (vector<16xf8E4M3FN>) -> ()
  %4 = arith.truncf %x downward : vector<16xf128> to vector<16xf8E4M3FN>
  func.call @print_f8E4M3FN(%4) : (vector<16xf8E4M3FN>) -> ()
  return
}

func.func @truncf_f8E4M3FNUZ_to_f4E2M1FN(%codes : vector<16xi8>) {
  %x = arith.bitcast %codes : vector<16xi8> to vector<16xf8E4M3FNUZ>
  %0 = arith.truncf %x to_nearest_even
      : vector<16xf8E4M3FNUZ> to vector<16xf4E2M1FN>
  func.call @print_f4E2M1FN(%0) : (vector<16xf4E2M1FN>) -> ()
  %1 = arith.truncf %x upward : vector<16xf8E4M3FNUZ> to vector<16xf4E2M1FN>
  func.call @print_f4E2M1FN(%1) : (vector<16xf4E2M1FN>) -> ()
  %2 = arith.truncf %x downward : vector<16xf8E4M3FNUZ> to vector<16xf4E2M1FN>
  func.call @print_f4E2M1FN(%2) : (vector<16xf4E2M1FN>) -> ()
  return
}

func.func @truncf_f32_to_f8E8M0FNU(%bits : vector<16xi32>) {
  %x = arith.bitcast %bits : vector<16xi32> to vector<16xf32>
  %0 = arith.truncf %x to_nearest_even : vector<16xf32> to vector<16xf8E8M0FNU>
  func.call @print_f8E8M0FNU(%0) : (vector<16xf8E8M0FNU>) -> ()
  %1 = arith.truncf %x to_nearest_away : vector<16xf32> to vector<16xf8E8M0FNU>
  func.call @print_f8E8M0FNU(%1) : (vector<16xf8E8M0FNU>) -> ()
  %2 = arith.truncf %x toward_zero : vector<16xf32> to vector<16xf8E8M0FNU>
  func.call @print_f8E8M0FNU(%2) : (vector<16xf8E8M0FNU>) -> ()
  %3 = arith.truncf %x upward : vector<16xf32> to vector<16xf8E8M0FNU>
  func.call @print_f8E8M0FNU(%3) : (vector<16xf8E8M0FNU>) -> ()
  %4 = arith.truncf %x downward : vector<16xf32> to vector<16xf8E8M0FNU>
  func.call @print_f8E8M0FNU(%4) : (vector<16xf8E8M0FNU>) -> ()
  return
}

func.func @truncf_f64_to_f8E8M0FNU(%bits : vector<16xi64>) {
  %x = arith.bitcast %bits : vector<16xi64> to vector<16xf64>
  %0 = arith.truncf %x to_nearest_even : vector<16xf64> to vector<16xf8E8M0FNU>
  func.call @print_f8E8M0FNU(%0) : (vector<16xf8E8M0FNU>) -> ()
  %1 = arith.truncf %x to_nearest_away : vector<16xf64> to vector<16xf8E8M0FNU>
  func.call @print_f8E8M0FNU(%1) : (vector<16xf8E8M0FNU>) -> ()
  %2 = arith.truncf %x toward_zero : vector<16xf64> to vector<16xf8E8M0FNU>
  func.call @print_f8E8M0FNU(%2) : (vector<16xf8E8M0FNU>) -> ()
  %3 = arith.truncf %x upward : vector<16xf64> to vector<16xf8E8M0FNU>
  func.call @print_f8E8M0FNU(%3) : (vector<16xf8E8M0FNU>) -> ()
  %4 = arith.truncf %x downward : vector<16xf64> to vector<16xf8E8M0FNU>
  func.call @print_f8E8M0FNU(%4) : (vector<16xf8E8M0FNU>) -> ()
  %5 = arith.truncf %x : vector<16xf64> to vector<16xf8E8M0FNU>
  func.call @print_f8E8M0FNU(%5) : (vector<16xf8E8M0FNU>) -> ()
  return
}

func.func @truncf_bf16_to_f8E8M0FNU(%bits : vector<16xi16>) {
  %x = arith.bitcast %bits : vector<16xi16> to vector<16xbf16>
  %0 = arith.truncf %x to_nearest_even
      : vector<16xbf16> to vector<16xf8E8M0FNU>
  func.call @print_f8E8M0FNU(%0) : (vector<16xf8E8M0FNU>) -> ()
  %1 = arith.truncf %x upward : vector<16xbf16> to vector<16xf8E8M0FNU>
  func.call @print_f8E8M0FNU(%1) : (vector<16xf8E8M0FNU>) -> ()
  %2 = arith.truncf %x downward : vector<16xbf16> to vector<16xf8E8M0FNU>
  func.call @print_f8E8M0FNU(%2) : (vector<16xf8E8M0FNU>) -> ()
  return
}

func.func @truncf_f16_to_f8E8M0FNU(%bits : vector<16xi16>) {
  %x = arith.bitcast %bits : vector<16xi16> to vector<16xf16>
  %0 = arith.truncf %x to_nearest_even : vector<16xf16> to vector<16xf8E8M0FNU>
  func.call @print_f8E8M0FNU(%0) : (vector<16xf8E8M0FNU>) -> ()
  %1 = arith.truncf %x upward : vector<16xf16> to vector<16xf8E8M0FNU>
  func.call @print_f8E8M0FNU(%1) : (vector<16xf8E8M0FNU>) -> ()
  %2 = arith.truncf %x downward : vector<16xf16> to vector<16xf8E8M0FNU>
  func.call @print_f8E8M0FNU(%2) : (vector<16xf8E8M0FNU>) -> ()
  return
}

func.func @extf_f8E4M3FN(%codes : vector<8xi8>) {
  %x = arith.bitcast %codes : vector<8xi8> to vector<8xf8E4M3FN>
  %0 = arith.extf %x : vector<8xf8E4M3FN> to vector<8xf32>
  func.call @print_f32(%0) : (vector<8xf32>) -> ()
  %1 = arith.extf %x : vector<8xf8E4M3FN> to vector<8xf16>
  func.call @print_f16(%1) : (vector<8xf16>) -> ()
  %2 = arith.extf %x : vector<8xf8E4M3FN> to vector<8xbf16>
  func.call @print_bf16(%2) : (vector<8xbf16>) -> ()
  %3 = arith.extf %x : vector<8xf8E4M3FN> to vector<8xf64>
  func.call @print_f64(%3) : (vector<8xf64>) -> ()
  return
}

func.func @extf_f8E5M2(%codes : vector<8xi8>) {
  %x = arith.bitcast %codes : vector<8xi8> to vector<8xf8E5M2>
  %0 = arith.extf %x : vector<8xf8E5M2> to vector<8xf32>
  func.call @print_f32(%0) : (vector<8xf32>) -> ()
  %1 = arith.extf %x : vector<8xf8E5M2> to vector<8xf16>
  func.call @print_f16(%1) : (vector<8xf16>) -> ()
  %2 = arith.extf %x : vector<8xf8E5M2> to vector<8xbf16>
  func.call @print_bf16(%2) : (vector<8xbf16>) -> ()
  %3 = arith.extf %x : vector<8xf8E5M2> to vector<8xf64>
  func.call @print_f64(%3) : (vector<8xf64>) -> ()
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

func.func @extf_f4E2M1FN(%codes : vector<8xi8>) {
  %narrow = arith.trunci %codes : vector<8xi8> to vector<8xi4>
  %x = arith.bitcast %narrow : vector<8xi4> to vector<8xf4E2M1FN>
  %0 = arith.extf %x : vector<8xf4E2M1FN> to vector<8xf32>
  func.call @print_f32(%0) : (vector<8xf32>) -> ()
  %1 = arith.extf %x : vector<8xf4E2M1FN> to vector<8xf16>
  func.call @print_f16(%1) : (vector<8xf16>) -> ()
  %2 = arith.extf %x : vector<8xf4E2M1FN> to vector<8xbf16>
  func.call @print_bf16(%2) : (vector<8xbf16>) -> ()
  %3 = arith.extf %x : vector<8xf4E2M1FN> to vector<8xf64>
  func.call @print_f64(%3) : (vector<8xf64>) -> ()
  return
}

func.func @extf_f8E8M0FNU(%bits : vector<8xi8>) {
  %x = arith.bitcast %bits : vector<8xi8> to vector<8xf8E8M0FNU>
  %0 = arith.extf %x : vector<8xf8E8M0FNU> to vector<8xf32>
  func.call @print_f32(%0) : (vector<8xf32>) -> ()
  %1 = arith.extf %x : vector<8xf8E8M0FNU> to vector<8xbf16>
  func.call @print_bf16(%1) : (vector<8xbf16>) -> ()
  %2 = arith.extf %x : vector<8xf8E8M0FNU> to vector<8xf64>
  func.call @print_f64(%2) : (vector<8xf64>) -> ()
  return
}

func.func @extf_f4E2M1FN_to_narrow(%codes : vector<16xi8>) {
  %narrow = arith.trunci %codes : vector<16xi8> to vector<16xi4>
  %x = arith.bitcast %narrow : vector<16xi4> to vector<16xf4E2M1FN>
  %0 = arith.extf %x : vector<16xf4E2M1FN> to vector<16xf6E2M3FN>
  func.call @print_f6E2M3FN(%0) : (vector<16xf6E2M3FN>) -> ()
  %1 = arith.extf %x : vector<16xf4E2M1FN> to vector<16xf6E3M2FN>
  func.call @print_f6E3M2FN(%1) : (vector<16xf6E3M2FN>) -> ()
  %2 = arith.extf %x : vector<16xf4E2M1FN> to vector<16xf8E4M3FN>
  func.call @print_f8E4M3FN(%2) : (vector<16xf8E4M3FN>) -> ()
  %3 = arith.extf %x : vector<16xf4E2M1FN> to vector<16xf8E5M2>
  func.call @print_f8E5M2(%3) : (vector<16xf8E5M2>) -> ()
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

  // f4E2M1FN: u = 2^-1, s = 2^-1, M = 6 (0x7); no Inf or NaN.
  %to_f4E2M1FN = arith.constant dense<[
      0x3FA00000, 0xBFE00000, 0x3FA00008, 0xBF900000,
      0x3F400000, 0xBE800000, 0x3E000000, 0x80000000,
      0x40C00000, 0x40A00000, 0x40E00000, 0xC0F00000,
      0x7F800000, 0xFF800000, 0x7FE00000, 0xFF800001]> : vector<16xi32>
  // CHECK:      ( 2, 12, 3, 10, 2, 8, 0, 8, 7, 6, 7, 15, 7, 15, 7, 7 )
  // CHECK-NEXT: ( 3, 12, 3, 10, 2, 9, 0, 8, 7, 7, 7, 15, 7, 15, 7, 7 )
  // CHECK-NEXT: ( 2, 11, 2, 10, 1, 8, 0, 8, 7, 6, 7, 15, 7, 15, 7, 7 )
  // CHECK-NEXT: ( 3, 11, 3, 10, 2, 8, 1, 8, 7, 7, 7, 15, 7, 15, 7, 7 )
  // CHECK-NEXT: ( 2, 12, 2, 11, 1, 9, 0, 8, 7, 6, 7, 15, 7, 15, 7, 7 )
  func.call @truncf_f32_to_f4E2M1FN(%to_f4E2M1FN) : (vector<16xi32>) -> ()

  // f6E2M3FN: u = 2^-3, s = 2^-3, M = 7.5 (0x1f); no Inf or NaN.
  %to_f6E2M3FN = arith.constant dense<[
      0x3F880000, 0xBF980000, 0x3F880008, 0xBF840000,
      0x3E400000, 0xBD800000, 0x3D000000, 0x80000000,
      0x40F00000, 0x40E80000, 0x40F80000, 0xC0FC0000,
      0x7F800000, 0xFF800000, 0x7FE00000, 0xFF800001]> : vector<16xi32>
  // CHECK-NEXT: ( 8, 42, 9, 40, 2, 32, 0, 32, 31, 30, 31, 63, 31, 63, 31, 31 )
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

  // f8E4M3FN: u = 2^-3, s = 2^-9, M = 448 (0x7e); no Inf, NaN 0x7f/0xff.
  %to_f8E4M3FN = arith.constant dense<[
      0x3F880000, 0xBF980000, 0x3F880008, 0xBF840000,
      0x3B400000, 0xBA800000, 0x3A000000, 0x80000000,
      0x43E00000, 0x43D80000, 0x43E80000, 0xC3EC0000,
      0x7F800000, 0xFF800000, 0x7FE00000, 0xFF800001]> : vector<16xi32>
  // CHECK-NEXT: ( 56, 186, 57, 184, 2, 128, 0, 128, 126, 126, 126, 255, 127, 255, 127, 255 )
  // CHECK-NEXT: ( 57, 186, 57, 184, 2, 129, 0, 128, 126, 126, 127, 255, 127, 255, 127, 255 )
  // CHECK-NEXT: ( 56, 185, 56, 184, 1, 128, 0, 128, 126, 125, 126, 254, 127, 255, 127, 255 )
  // CHECK-NEXT: ( 57, 185, 57, 184, 2, 128, 1, 128, 126, 126, 127, 254, 127, 255, 127, 255 )
  // CHECK-NEXT: ( 56, 186, 56, 185, 1, 129, 0, 128, 126, 125, 126, 255, 127, 255, 127, 255 )
  func.call @truncf_f32_to_f8E4M3FN(%to_f8E4M3FN) : (vector<16xi32>) -> ()

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

  // f8E5M2: u = 2^-2, s = 2^-16, M = 57344 (0x7b); Inf 0x7c, quiet NaN 0x7e.
  %to_f8E5M2 = arith.constant dense<[
      0x3F900000, 0xBFB00000, 0x3F900008, 0xBF880000,
      0x37C00000, 0xB7000000, 0x36800000, 0x80000000,
      0x47600000, 0x47500000, 0x47700000, 0xC7780000,
      0x7F800000, 0xFF800000, 0x7FE00000, 0xFF800001]> : vector<16xi32>
  // CHECK-NEXT: ( 60, 190, 61, 188, 2, 128, 0, 128, 123, 122, 124, 252, 124, 252, 127, 254 )
  // CHECK-NEXT: ( 61, 190, 61, 188, 2, 129, 0, 128, 123, 123, 124, 252, 124, 252, 127, 254 )
  // CHECK-NEXT: ( 60, 189, 60, 188, 1, 128, 0, 128, 123, 122, 123, 251, 124, 252, 127, 254 )
  // CHECK-NEXT: ( 61, 189, 61, 188, 2, 128, 1, 128, 123, 123, 124, 251, 124, 252, 127, 254 )
  // CHECK-NEXT: ( 60, 190, 60, 189, 1, 129, 0, 128, 123, 122, 123, 252, 124, 252, 127, 254 )
  func.call @truncf_f32_to_f8E5M2(%to_f8E5M2) : (vector<16xi32>) -> ()

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

  // truncf from f16, bf16, f64 and a narrow source: to_nearest_even, upward and
  // downward, or all five modes (the *_modes and f8E4M3B11FNUZ helpers).

  // f16 -> f8E5M2 (same exponent range): 1.125 and -1.375 (ties),
  // 1.125+2^-10, -1.03125, 1.5*2^-16 (a subnormal tie), -2^-17 (half
  // the smallest subnormal), 2^-24, 2^-14-2^-24 (rounds up to the
  // smallest normal), 57344 (max), 53248 and 61440 (ties), -65504,
  // +-Inf, quiet NaN 0x7e01, signaling NaN 0xfd00.
  %f16_to_f8E5M2 = arith.constant dense<[
      0x3C80, 0xBD80, 0x3C81, 0xBC20,
      0x0180, 0x8080, 0x0001, 0x03FF,
      0x7B00, 0x7A80, 0x7B80, 0xFBFF,
      0x7C00, 0xFC00, 0x7E01, 0xFD00]> : vector<16xi16>
  // CHECK-NEXT: ( 60, 190, 61, 188, 2, 128, 0, 4, 123, 122, 124, 252, 124, 252, 126, 255 )
  // CHECK-NEXT: ( 61, 189, 61, 188, 2, 128, 1, 4, 123, 123, 124, 251, 124, 252, 126, 255 )
  // CHECK-NEXT: ( 60, 190, 60, 189, 1, 129, 0, 3, 123, 122, 123, 252, 124, 252, 126, 255 )
  func.call @truncf_f16_to_f8E5M2(%f16_to_f8E5M2) : (vector<16xi16>) -> ()

  // f16 -> f8E5M2 in every mode: 1.125 and -1.125 (ties), 1.125+2^-10,
  // 2^-14-2^-24 (carries into the smallest normal), -2^-24, 61408, 61440
  // (the tie above max), 65504, -61440, -65504, +-Inf, the signaling NaNs
  // 0x7c01 (its payload does not fit), 0xfc01 and 0x7d00, and -NaN 0xfe00.
  %f16_to_f8E5M2_modes = arith.constant dense<[
      0x3C80, 0x3C81, 0xBC80, 0x03FF,
      0x8001, 0x7B7F, 0x7B80, 0x7BFF,
      0xFB80, 0xFBFF, 0x7C00, 0xFC00,
      0x7C01, 0xFC01, 0x7D00, 0xFE00]> : vector<16xi16>
  // CHECK-NEXT: ( 60, 61, 188, 4, 128, 123, 124, 124, 252, 252, 124, 252, 126, 254, 127, 254 )
  // CHECK-NEXT: ( 61, 61, 189, 4, 128, 123, 124, 124, 252, 252, 124, 252, 126, 254, 127, 254 )
  // CHECK-NEXT: ( 60, 60, 188, 3, 128, 123, 123, 123, 251, 251, 124, 252, 126, 254, 127, 254 )
  // CHECK-NEXT: ( 61, 61, 188, 4, 128, 124, 124, 124, 251, 251, 124, 252, 126, 254, 127, 254 )
  // CHECK-NEXT: ( 60, 60, 189, 3, 129, 123, 123, 123, 252, 252, 124, 252, 126, 254, 127, 254 )
  func.call @truncf_f16_to_f8E5M2_modes(%f16_to_f8E5M2_modes)
      : (vector<16xi16>) -> ()

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

  // f16 -> f4E2M1FN (no Inf or NaN): -Inf, +Inf, -NaN, +NaN (quiet),
  // -NaN, +NaN (signaling), -6 (max), -7 (tie above max), -100, -65504,
  // -0, 1, 5 and -5 (ties), 0.25 (tie with 0), 0.75 (tie).
  %f16_to_f4E2M1FN = arith.constant dense<[
      0xFC00, 0x7C00, 0xFE00, 0x7E00,
      0xFC01, 0x7C01, 0xC600, 0xC700,
      0xD640, 0xFBFF, 0x8000, 0x3C00,
      0x4500, 0xC500, 0x3400, 0x3A00]> : vector<16xi16>
  // CHECK-NEXT: ( 15, 7, 7, 7, 7, 7, 15, 15, 15, 15, 8, 2, 6, 14, 0, 2 )
  // CHECK-NEXT: ( 15, 7, 7, 7, 7, 7, 15, 15, 15, 15, 8, 2, 7, 14, 1, 2 )
  // CHECK-NEXT: ( 15, 7, 7, 7, 7, 7, 15, 15, 15, 15, 8, 2, 6, 15, 0, 1 )
  func.call @truncf_f16_to_f4E2M1FN(%f16_to_f4E2M1FN) : (vector<16xi16>) -> ()

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

  // bf16 -> f8E5M2: 1.125 and -1.375 (ties), 1.1328125, -1.03125,
  // 1.5*2^-16 (subnormal tie), -2^-17, the smallest bf16 subnormal,
  // 2^-14, 57344 (max), 53248 and 61440 (ties), -63488, +-Inf, quiet
  // NaN 0x7fc1, signaling NaN 0xffa0.
  %bf16_to_f8E5M2 = arith.constant dense<[
      0x3F90, 0xBFB0, 0x3F91, 0xBF84,
      0x37C0, 0xB700, 0x0001, 0x3880,
      0x4760, 0x4750, 0x4770, 0xC778,
      0x7F80, 0xFF80, 0x7FC1, 0xFFA0]> : vector<16xi16>
  // CHECK-NEXT: ( 60, 190, 61, 188, 2, 128, 0, 4, 123, 122, 124, 252, 124, 252, 126, 255 )
  // CHECK-NEXT: ( 61, 189, 61, 188, 2, 128, 1, 4, 123, 123, 124, 251, 124, 252, 126, 255 )
  // CHECK-NEXT: ( 60, 190, 60, 189, 1, 129, 0, 4, 123, 122, 123, 252, 124, 252, 126, 255 )
  func.call @truncf_bf16_to_f8E5M2(%bf16_to_f8E5M2) : (vector<16xi16>) -> ()

  // f64 -> f8E4M3FN: 400+2^-30 and 1.0625+2^-40 (just above a tie;
  // rounding to f32 first would land on the tie), 1.0625 and -1.1875
  // (ties), 1+2^-52, -(1.0625-2^-52), 1.5*2^-9 (subnormal tie), +-the
  // smallest f64 subnormal, 448 (max), 464 (tie above max), 464+2^-44,
  // +-1e300, -Inf, NaN.
  %f64_to_f8E4M3FN = arith.constant dense<[
      0x4079000000004000, 0x3FF1000000001000,
      0x3FF1000000000000, 0xBFF3000000000000,
      0x3FF0000000000001, 0xBFF0FFFFFFFFFFFF,
      0x3F68000000000000, 0x0000000000000001,
      0x8000000000000001, 0x407C000000000000,
      0x407D000000000000, 0x407D000000000001,
      0x7E37E43C8800759C, 0xFE37E43C8800759C,
      0xFFF0000000000000, 0x7FF8000000000000]> : vector<16xi64>
  // CHECK-NEXT: ( 125, 57, 56, 186, 56, 184, 2, 0, 128, 126, 126, 127, 127, 255, 255, 127 )
  // CHECK-NEXT: ( 125, 57, 57, 185, 57, 184, 2, 1, 128, 126, 127, 127, 127, 254, 255, 127 )
  // CHECK-NEXT: ( 124, 56, 56, 186, 56, 185, 1, 0, 129, 126, 126, 126, 126, 255, 255, 127 )
  func.call @truncf_f64_to_f8E4M3FN(%f64_to_f8E4M3FN) : (vector<16xi64>) -> ()

  // f64 -> f4E2M1FN: 5, -5, 1.25 (ties), 1.25+2^-40, 3.5, -2.5,
  // 0.75 (ties), -0.25 (tie with -0), 0.25+2^-54, the smallest f64
  // subnormal, 7 (tie above max), -1e300, +-Inf, NaN, -0.
  %f64_to_f4E2M1FN = arith.constant dense<[
      0x4014000000000000, 0xC014000000000000,
      0x3FF4000000000000, 0x3FF4000000001000,
      0x400C000000000000, 0xC004000000000000,
      0x3FE8000000000000, 0xBFD0000000000000,
      0x3FD0000000000001, 0x0000000000000001,
      0x401C000000000000, 0xFE37E43C8800759C,
      0x7FF0000000000000, 0xFFF0000000000000,
      0x7FF8000000000000, 0x8000000000000000]> : vector<16xi64>
  // CHECK-NEXT: ( 6, 14, 2, 3, 6, 12, 2, 8, 1, 0, 7, 15, 7, 15, 7, 8 )
  // CHECK-NEXT: ( 7, 14, 3, 3, 6, 12, 2, 8, 1, 1, 7, 15, 7, 15, 7, 8 )
  // CHECK-NEXT: ( 6, 15, 2, 2, 5, 13, 1, 9, 0, 0, 7, 15, 7, 15, 7, 8 )
  func.call @truncf_f64_to_f4E2M1FN(%f64_to_f4E2M1FN) : (vector<16xi64>) -> ()

  // f128 -> f8E5M2 and f8E4M3FN, rounded to odd into f64 first. Rounding to
  // nearest through f64 instead would land on a tie or a value of the format
  // for most inputs: those within 2^-70 of one (1.125+2^-70, -(1.125+2^-70),
  // 1.25-2^-70, 1.375-2^-70, 1+2^-70, -(1+2^-70)), 2^-17+2^-90 (above the
  // subnormal tie), 61440-2^-58 and -(61440+2^-58) (around the overflow tie),
  // 1.125+2^-112; also 2^1100, -2^-1099 (beyond f64's range), -0, +Inf, a
  // quiet NaN with payload 0x123 and a negative signaling NaN.
  %f128_to_f8E5M2 = arith.constant dense<[
      0x3FFF2000000000000000040000000000, 0xBFFF2000000000000000040000000000,
      0x3FFF3FFFFFFFFFFFFFFFFC0000000000, 0x3FFF5FFFFFFFFFFFFFFFFC0000000000,
      0x3FFF0000000000000000040000000000, 0xBFFF0000000000000000040000000000,
      0x3FEE0000000000000000008000000000, 0x400EDFFFFFFFFFFFFFFFFF8000000000,
      0xC00EE000000000000000008000000000, 0x3FFF2000000000000000000000000001,
      0x444B0000000000000000000000000000, 0xBBB40000000000000000000000000000,
      0x80000000000000000000000000000000, 0x7FFF0000000000000000000000000000,
      0x7FFF8000000000000000000000000123, 0xFFFF4000000000000000000000000000
  ]> : vector<16xi128>
  // CHECK-NEXT: ( 61, 189, 61, 61, 60, 188, 1, 123, 252, 61, 124, 128, 128, 124, 126, 255 )
  // CHECK-NEXT: ( 61, 189, 61, 61, 60, 188, 1, 123, 252, 61, 124, 128, 128, 124, 126, 255 )
  // CHECK-NEXT: ( 60, 188, 60, 61, 60, 188, 0, 123, 251, 60, 123, 128, 128, 124, 126, 255 )
  // CHECK-NEXT: ( 61, 188, 61, 62, 61, 188, 1, 124, 251, 61, 124, 128, 128, 124, 126, 255 )
  // CHECK-NEXT: ( 60, 189, 60, 61, 60, 189, 0, 123, 252, 60, 123, 129, 128, 124, 126, 255 )
  func.call @truncf_f128_to_f8E5M2(%f128_to_f8E5M2) : (vector<16xi128>) -> ()

  // 1.0625+2^-70, -(1.0625+2^-70), 1.0625-2^-70, 1.125-2^-70, -(1+2^-70),
  // 1.1875-2^-70, -(2^-10-2^-90), 2^-10+2^-90 (around the subnormal tie),
  // 464+2^-60, 464-2^-60 (around the overflow tie), -(448+2^-60), 2^1100,
  // 2^-1099, 0, -Inf, a negative quiet NaN.
  %f128_to_f8E4M3FN = arith.constant dense<[
      0x3FFF1000000000000000040000000000, 0xBFFF1000000000000000040000000000,
      0x3FFF0FFFFFFFFFFFFFFFFC0000000000, 0x3FFF1FFFFFFFFFFFFFFFFC0000000000,
      0xBFFF0000000000000000040000000000, 0x3FFF2FFFFFFFFFFFFFFFFC0000000000,
      0xBFF4FFFFFFFFFFFFFFFFFFFE00000000, 0x3FF50000000000000000000100000000,
      0x4007D000000000000000100000000000, 0x4007CFFFFFFFFFFFFFFFF00000000000,
      0xC007C000000000000000100000000000, 0x444B0000000000000000000000000000,
      0x3BB40000000000000000000000000000, 0x00000000000000000000000000000000,
      0xFFFF0000000000000000000000000000, 0xFFFFC000000000000000000000000000
  ]> : vector<16xi128>
  // CHECK-NEXT: ( 57, 185, 56, 57, 184, 57, 128, 1, 127, 126, 254, 127, 0, 0, 255, 255 )
  // CHECK-NEXT: ( 57, 185, 56, 57, 184, 57, 128, 1, 127, 126, 254, 127, 0, 0, 255, 255 )
  // CHECK-NEXT: ( 56, 184, 56, 56, 184, 57, 128, 0, 126, 126, 254, 126, 0, 0, 255, 255 )
  // CHECK-NEXT: ( 57, 184, 57, 57, 184, 58, 128, 1, 127, 127, 254, 127, 1, 0, 255, 255 )
  // CHECK-NEXT: ( 56, 185, 56, 56, 185, 57, 129, 0, 126, 126, 255, 126, 0, 0, 255, 255 )
  func.call @truncf_f128_to_f8E4M3FN(%f128_to_f8E4M3FN)
      : (vector<16xi128>) -> ()

  // f8E4M3FNUZ -> f4E2M1FN (a narrow source is decoded exactly first):
  // 1.25, -2.5, 3.5, 5 (ties), 5.5, 10, 0.375, 0.25 (tie with 0),
  // -0.140625, 0.140625, +-240, NaN, 0, 1.5, 0.5.
  %f8E4M3FNUZ_to_f4E2M1FN = arith.constant dense<[
      0x42, 0xCA, 0x4E, 0x52, 0x53, 0x5A, 0x34, 0x30,
      0xA9, 0x29, 0x7F, 0xFF, 0x80, 0x00, 0x44, 0x38]> : vector<16xi8>
  // CHECK-NEXT: ( 2, 12, 6, 6, 7, 7, 1, 0, 8, 0, 7, 15, 7, 0, 3, 1 )
  // CHECK-NEXT: ( 3, 12, 6, 7, 7, 7, 1, 1, 8, 1, 7, 15, 7, 0, 3, 1 )
  // CHECK-NEXT: ( 2, 13, 5, 6, 6, 7, 0, 0, 9, 0, 7, 15, 7, 0, 3, 1 )
  func.call @truncf_f8E4M3FNUZ_to_f4E2M1FN(%f8E4M3FNUZ_to_f4E2M1FN)
      : (vector<16xi8>) -> ()

  // truncf into f8E8M0FNU: code c is 2^(c - 127), 255 the NaN; there is no
  // zero (what would round to zero is code 0) and no sign (x is rounded with
  // its sign, which is then dropped). One significant bit: a tie m = 1.5
  // rounds up in to_nearest_even. An overflow of x > 0 is the NaN in
  // to_nearest_* and upward, 254 (2^127) otherwise.

  // f32 (extended to f64 first) in every mode: 1, 1.5 (tie), 1.5-2^-23,
  // 3 and -3 (ties), -1.25, 2^-127, 1.5*2^-127 (tie), 2^-128, +-0,
  // max-finite, 2^127, +Inf, NaN, -1.5*2^-127.
  %f32_to_f8E8M0FNU = arith.constant dense<[
      0x3F800000, 0x3FC00000, 0x3FBFFFFF, 0x40400000,
      0xC0400000, 0xBFA00000, 0x00400000, 0x00600000,
      0x00200000, 0x00000000, 0x80000000, 0x7F7FFFFF,
      0x7F000000, 0x7F800000, 0x7FC00000, 0x80600000]> : vector<16xi32>
  // CHECK-NEXT: ( 127, 128, 127, 129, 129, 127, 0, 1, 0, 0, 0, 255, 254, 255, 255, 1 )
  // CHECK-NEXT: ( 127, 128, 127, 129, 129, 127, 0, 1, 0, 0, 0, 255, 254, 255, 255, 1 )
  // CHECK-NEXT: ( 127, 127, 127, 128, 128, 127, 0, 0, 0, 0, 0, 254, 254, 255, 255, 0 )
  // CHECK-NEXT: ( 127, 128, 128, 129, 128, 127, 0, 1, 0, 0, 0, 255, 254, 255, 255, 0 )
  // CHECK-NEXT: ( 127, 127, 127, 128, 129, 128, 0, 0, 0, 0, 0, 254, 254, 255, 255, 1 )
  func.call @truncf_f32_to_f8E8M0FNU(%f32_to_f8E8M0FNU) : (vector<16xi32>) -> ()

  // f64 in every mode, then without one (toward zero, from the f64 bits):
  // 2^-125-2^-178 (2^-125 if rounded to f32 first), 2^128, 1.5*2^127 (the
  // overflow tie), 1.5*2^127-2^75, 1+2^-52, 1.5, -(1.5-2^-52), 2^-127,
  // 1.5*2^-127, the smallest subnormal, 2^-1000, -2^-128, max-finite, +Inf,
  // NaN, 0.75.
  %f64_to_f8E8M0FNU = arith.constant dense<[
      0x381FFFFFFFFFFFFF, 0x47F0000000000000,
      0x47E8000000000000, 0x47E7FFFFFFFFFFFF,
      0x3FF0000000000001, 0x3FF8000000000000,
      0xBFF7FFFFFFFFFFFF, 0x3800000000000000,
      0x3808000000000000, 0x0000000000000001,
      0x0170000000000000, 0xB7F0000000000000,
      0x7FEFFFFFFFFFFFFF, 0x7FF0000000000000,
      0x7FF8000000000000, 0x3FE8000000000000]> : vector<16xi64>
  // CHECK-NEXT: ( 2, 255, 255, 254, 127, 128, 127, 0, 1, 0, 0, 0, 255, 255, 255, 127 )
  // CHECK-NEXT: ( 2, 255, 255, 254, 127, 128, 127, 0, 1, 0, 0, 0, 255, 255, 255, 127 )
  // CHECK-NEXT: ( 1, 254, 254, 254, 127, 127, 127, 0, 0, 0, 0, 0, 254, 255, 255, 126 )
  // CHECK-NEXT: ( 2, 255, 255, 255, 128, 128, 127, 0, 1, 0, 0, 0, 255, 255, 255, 127 )
  // CHECK-NEXT: ( 1, 254, 254, 254, 127, 127, 128, 0, 0, 0, 0, 0, 254, 255, 255, 126 )
  // CHECK-NEXT: ( 1, 254, 254, 254, 127, 127, 127, 0, 0, 0, 0, 0, 254, 255, 255, 126 )
  func.call @truncf_f64_to_f8E8M0FNU(%f64_to_f8E8M0FNU) : (vector<16xi64>) -> ()

  // bf16, to_nearest_even, upward and downward: 2^-127, 1.5*2^-127 (tie),
  // 2^-128, 2^-133, 2^-126-2^-133 (bf16 subnormals), 1.5, -3, 1, max-finite,
  // 2^127, 1.5*2^127, +Inf, NaN, -0, -1.25, 0.75.
  %bf16_to_f8E8M0FNU = arith.constant dense<[
      0x0040, 0x0060, 0x0020, 0x0001,
      0x007F, 0x3FC0, 0xC040, 0x3F80,
      0x7F7F, 0x7F00, 0x7F40, 0x7F80,
      0x7FC0, 0x8000, 0xBFA0, 0x3F40]> : vector<16xi16>
  // CHECK-NEXT: ( 0, 1, 0, 0, 1, 128, 129, 127, 255, 254, 255, 255, 255, 0, 127, 127 )
  // CHECK-NEXT: ( 0, 1, 0, 0, 1, 128, 128, 127, 255, 254, 255, 255, 255, 0, 127, 127 )
  // CHECK-NEXT: ( 0, 0, 0, 0, 0, 127, 129, 127, 254, 254, 254, 255, 255, 0, 128, 126 )
  func.call @truncf_bf16_to_f8E8M0FNU(%bf16_to_f8E8M0FNU)
      : (vector<16xi16>) -> ()

  // f16, the same modes: 1.5, -3, 1+2^-10, 2^-24, 1.5*2^-23 (tie), 65504,
  // +Inf, NaN, -0, 0.75, 2^-14, 5 and -5, 2^-14-2^-24, -1, 1.
  %f16_to_f8E8M0FNU = arith.constant dense<[
      0x3E00, 0xC200, 0x3C01, 0x0001,
      0x0003, 0x7BFF, 0x7C00, 0x7E00,
      0x8000, 0x3A00, 0x0400, 0x4500,
      0xC500, 0x03FF, 0xBC00, 0x3C00]> : vector<16xi16>
  // CHECK-NEXT: ( 128, 129, 127, 103, 105, 143, 255, 255, 0, 127, 113, 129, 129, 113, 127, 127 )
  // CHECK-NEXT: ( 128, 128, 128, 103, 105, 143, 255, 255, 0, 127, 113, 130, 129, 113, 127, 127 )
  // CHECK-NEXT: ( 127, 129, 127, 103, 104, 142, 255, 255, 0, 126, 113, 129, 130, 112, 127, 127 )
  func.call @truncf_f16_to_f8E8M0FNU(%f16_to_f8E8M0FNU) : (vector<16xi16>) -> ()

  // extf from each format; each @extf_* prints the f32, f16, bf16 and f64
  // results (@extf_f8E8M0FNU: f32, bf16 and f64).

  // f8E4M3FN: 2^-9, 7*2^-9, 2^-6, 448, -0, -1.125, NaN, -NaN.
  %from_f8E4M3FN = arith.constant dense<[
      0x01, 0x07, 0x08, 0x7E, 0x80, 0xB9, 0x7F, 0xFF]> : vector<8xi8>
  // CHECK-NEXT: ( 989855744, 1012924416, 1015021568, 1138753536, 2147483648, 3213885440, 2146435072, 4293918720 )
  // CHECK-NEXT: ( 6144, 8960, 9216, 24320, 32768, 48256, 32640, 65408 )
  // CHECK-NEXT: ( 15104, 15456, 15488, 17376, 32768, 49040, 32752, 65520 )
  // CHECK-NEXT: ( 4566650022153682944, 4579034921128951808, 4580160821035794432, 4646588915539509248, 9223372036854775808, 13831117405608214528, 9222809086901354496, 18446181123756130304 )
  func.call @extf_f8E4M3FN(%from_f8E4M3FN) : (vector<8xi8>) -> ()

  // f8E5M2: 2^-16, 3*2^-16, 2^-14, 57344, -Inf, +Inf, sNaN 0x7d, -qNaN 0xff.
  %from_f8E5M2 = arith.constant dense<[
      0x01, 0x03, 0x04, 0x7B, 0xFC, 0x7C, 0x7D, 0xFF]> : vector<8xi8>
  // CHECK-NEXT: ( 931135488, 943718400, 947912704, 1197473792, 4286578688, 2139095040, 2145386496, 4292870144 )
  // CHECK-NEXT: ( 256, 768, 1024, 31488, 64512, 31744, 32512, 65280 )
  // CHECK-NEXT: ( 14208, 14400, 14464, 18272, 65408, 32640, 32736, 65504 )
  // CHECK-NEXT: ( 4535124824762089472, 4541880224203145216, 4544132024016830464, 4678114112931102720, 18442240474082181120, 9218868437227405312, 9222246136947933184, 18445618173802708992 )
  func.call @extf_f8E5M2(%from_f8E5M2) : (vector<8xi8>) -> ()

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

  // f4E2M1FN: 0.5, 1, 3, 6, -6, -0, -0.5, -4.
  %from_f4E2M1FN = arith.constant dense<[
      0x1, 0x2, 0x5, 0x7, 0xF, 0x8, 0x9, 0xE]> : vector<8xi8>
  // CHECK-NEXT: ( 1056964608, 1065353216, 1077936128, 1086324736, 3233808384, 2147483648, 3204448256, 3229614080 )
  // CHECK-NEXT: ( 14336, 15360, 16896, 17920, 50688, 32768, 47104, 50176 )
  // CHECK-NEXT: ( 16128, 16256, 16448, 16576, 49344, 32768, 48896, 49280 )
  // CHECK-NEXT: ( 4602678819172646912, 4607182418800017408, 4613937818241073152, 4618441417868443648, 13841813454723219456, 9223372036854775808, 13826050856027422720, 13839561654909534208 )
  func.call @extf_f4E2M1FN(%from_f4E2M1FN) : (vector<8xi8>) -> ()

  // f8E8M0FNU: code c is 2^(c - 127): 2^-127 (an f32 and bf16 subnormal),
  // 2^-126, 2^-125, 1, 2, 2^65, 2^127, NaN (f32, bf16 and f64 only: its f16
  // result goes through an f32 to f16 truncf).
  %from_f8E8M0FNU = arith.constant dense<[
      0x00, 0x01, 0x02, 0x7F, 0x80, 0xC0, 0xFE, 0xFF]> : vector<8xi8>
  // CHECK-NEXT: ( 4194304, 8388608, 16777216, 1065353216, 1073741824, 1610612736, 2130706432, 2143289344 )
  // CHECK-NEXT: ( 64, 128, 256, 16256, 16384, 24576, 32512, 32704 )
  // CHECK-NEXT: ( 4035225266123964416, 4039728865751334912, 4044232465378705408, 4607182418800017408, 4611686018427387904, 4899916394579099648, 5179139571476070400, 9221120237041090560 )
  func.call @extf_f8E8M0FNU(%from_f8E8M0FNU) : (vector<8xi8>) -> ()

  // f4E2M1FN -> f6E2M3FN, f6E3M2FN, f8E4M3FN, f8E5M2 (exact),
  // every code.
  %all_f4E2M1FN = arith.constant dense<[
      0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7,
      0x8, 0x9, 0xA, 0xB, 0xC, 0xD, 0xE, 0xF]> : vector<16xi8>
  // CHECK-NEXT: ( 0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60 )
  // CHECK-NEXT: ( 0, 8, 12, 14, 16, 18, 20, 22, 32, 40, 44, 46, 48, 50, 52, 54 )
  // CHECK-NEXT: ( 0, 48, 56, 60, 64, 68, 72, 76, 128, 176, 184, 188, 192, 196, 200, 204 )
  // CHECK-NEXT: ( 0, 56, 60, 62, 64, 66, 68, 70, 128, 184, 188, 190, 192, 194, 196, 198 )
  func.call @extf_f4E2M1FN_to_narrow(%all_f4E2M1FN) : (vector<16xi8>) -> ()
  return
}
