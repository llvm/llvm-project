// RUN:   mlir-opt %s -pass-pipeline="builtin.module(func.func(math-expand-ops),convert-vector-to-scf,convert-scf-to-cf,convert-vector-to-llvm,convert-to-llvm,reconcile-unrealized-casts)" \
// RUN: | mlir-runner                                                      \
// RUN:     -e main -entry-point-result=void -O0                               \
// RUN:     -shared-libs=%mlir_c_runner_utils  \
// RUN:     -shared-libs=%mlir_runner_utils    \
// RUN:     -shared-libs=%mlir_float16_utils   \
// RUN: | FileCheck %s

// -------------------------------------------------------------------------- //
// exp2f.
// -------------------------------------------------------------------------- //
func.func @func_exp2f(%a : f64) {
  %r = math.exp2 %a : f64
  vector.print %r : f64
  return
}

func.func @exp2f() {
  // CHECK: 2
  %a = arith.constant 1.0 : f64
  call @func_exp2f(%a) : (f64) -> ()

  // CHECK-NEXT: 4
  %b = arith.constant 2.0 : f64
  call @func_exp2f(%b) : (f64) -> ()

  // CHECK-NEXT: 5.65685
  %c = arith.constant 2.5 : f64
  call @func_exp2f(%c) : (f64) -> ()

  // CHECK-NEXT: 0.29730
  %d = arith.constant -1.75 : f64
  call @func_exp2f(%d) : (f64) -> ()

  // CHECK-NEXT: 1.09581
  %e = arith.constant 0.132 : f64
  call @func_exp2f(%e) : (f64) -> ()

  // CHECK-NEXT: inf
  %f1 = arith.constant 0.00 : f64
  %f2 = arith.constant 1.00 : f64
  %f = arith.divf %f2, %f1 : f64
  call @func_exp2f(%f) : (f64) -> ()

  // CHECK-NEXT: inf
  %g = arith.constant 5038939.0 : f64
  call @func_exp2f(%g) : (f64) -> ()

  // CHECK-NEXT: 0
  %neg_inf = arith.constant 0xff80000000000000 : f64
  call @func_exp2f(%neg_inf) : (f64) -> ()

  // CHECK-NEXT: inf
  %i = arith.constant 0x7fc0000000000000 : f64
  call @func_exp2f(%i) : (f64) -> ()
  return
}

// -------------------------------------------------------------------------- //
// round.
// -------------------------------------------------------------------------- //
func.func @func_roundf(%a : f32) {
  %r = math.round %a : f32
  vector.print %r : f32
  return
}

func.func @func_roundf$bitcast_result_to_int(%a : f32) {
  %b = math.round %a : f32
  %c = arith.bitcast %b : f32 to i32
  vector.print %c : i32
  return
}

func.func @func_roundf$vector(%a : vector<1xf32>) {
  %b = math.round %a : vector<1xf32>
  vector.print %b : vector<1xf32>
  return
}

func.func @roundf() {
  // CHECK-NEXT: 4
  %a = arith.constant 3.8 : f32
  call @func_roundf(%a) : (f32) -> ()

  // CHECK-NEXT: -4
  %b = arith.constant -3.8 : f32
  call @func_roundf(%b) : (f32) -> ()

  // CHECK-NEXT: -4
  %c = arith.constant -4.2 : f32
  call @func_roundf(%c) : (f32) -> ()

  // CHECK-NEXT: -495
  %d = arith.constant -495.0 : f32
  call @func_roundf(%d) : (f32) -> ()

  // CHECK-NEXT: 495
  %e = arith.constant 495.0 : f32
  call @func_roundf(%e) : (f32) -> ()

  // CHECK-NEXT: 9
  %f = arith.constant 8.5 : f32
  call @func_roundf(%f) : (f32) -> ()

  // CHECK-NEXT: -9
  %g = arith.constant -8.5 : f32
  call @func_roundf(%g) : (f32) -> ()

  // CHECK-NEXT: -0
  %h = arith.constant -0.4 : f32
  call @func_roundf(%h) : (f32) -> ()

  // Special values: 0, -0, inf, -inf, nan, -nan
  %cNeg0 = arith.constant -0.0 : f32
  %c0 = arith.constant 0.0 : f32
  %cInfInt = arith.constant 0x7f800000 : i32
  %cInf = arith.bitcast %cInfInt : i32 to f32
  %cNegInfInt = arith.constant 0xff800000 : i32
  %cNegInf = arith.bitcast %cNegInfInt : i32 to f32
  %cNanInt = arith.constant 0x7fc00000 : i32
  %cNan = arith.bitcast %cNanInt : i32 to f32
  %cNegNanInt = arith.constant 0xffc00000 : i32
  %cNegNan = arith.bitcast %cNegNanInt : i32 to f32

  // CHECK-NEXT: -0
  call @func_roundf(%cNeg0) : (f32) -> ()
  // CHECK-NEXT: 0
  call @func_roundf(%c0) : (f32) -> ()
  // CHECK-NEXT: inf
  call @func_roundf(%cInf) : (f32) -> ()
  // CHECK-NEXT: -inf
  call @func_roundf(%cNegInf) : (f32) -> ()
  // Per IEEE 754-2008, sign is not required when printing a negative NaN, so
  // print as an int to ensure input NaN is left unchanged.
  // CHECK-NEXT: 2143289344
  // CHECK-NEXT: 2143289344
  call @func_roundf$bitcast_result_to_int(%cNan) : (f32) -> ()
  vector.print %cNanInt : i32
  // CHECK-NEXT: -4194304
  // CHECK-NEXT: -4194304
  call @func_roundf$bitcast_result_to_int(%cNegNan) : (f32) -> ()
  vector.print %cNegNanInt : i32

  // Very large values (greater than INT_64_MAX)
  %c2To100 = arith.constant 1.268e30 : f32 // 2^100
  // CHECK-NEXT: 1.268e+30
  call @func_roundf(%c2To100) : (f32) -> ()

  // Values above and below 2^23 = 8388608
  %c8388606_5 = arith.constant 8388606.5 : f32
  %c8388607 = arith.constant 8388607.0 : f32
  %c8388607_5 = arith.constant 8388607.5 : f32
  %c8388608 = arith.constant 8388608.0 : f32
  %c8388609 = arith.constant 8388609.0 : f32

  // Bitcast result to int to avoid printing in scientific notation,
  // which does not display all significant digits.

  // CHECK-NEXT: 1258291198
  // hex: 0x4AFFFFFE
  call @func_roundf$bitcast_result_to_int(%c8388606_5) : (f32) -> ()
  // CHECK-NEXT: 1258291198
  // hex: 0x4AFFFFFE
  call @func_roundf$bitcast_result_to_int(%c8388607) : (f32) -> ()
  // CHECK-NEXT: 1258291200
  // hex: 0x4B000000
  call @func_roundf$bitcast_result_to_int(%c8388607_5) : (f32) -> ()
  // CHECK-NEXT: 1258291200
  // hex: 0x4B000000
  call @func_roundf$bitcast_result_to_int(%c8388608) : (f32) -> ()
  // CHECK-NEXT: 1258291201
  // hex: 0x4B000001
  call @func_roundf$bitcast_result_to_int(%c8388609) : (f32) -> ()

  // Check that vector type works
  %cVec = arith.constant dense<[0.5]> : vector<1xf32>
  // CHECK-NEXT: ( 1 )
  call @func_roundf$vector(%cVec) : (vector<1xf32>) -> ()

  return
}

// -------------------------------------------------------------------------- //
// pow.
// -------------------------------------------------------------------------- //
func.func @func_powff64(%a : f64, %b : f64) {
  %r = math.powf %a, %b : f64
  vector.print %r : f64
  return
}

func.func @func_powff32(%a : f32, %b : f32) {
  %r = math.powf %a, %b : f32
  vector.print %r : f32
  return
}

func.func @powf() {
  // CHECK-NEXT: 16
  %a   = arith.constant 4.0 : f64
  %a_p = arith.constant 2.0 : f64
  call @func_powff64(%a, %a_p) : (f64, f64) -> ()

  // CHECK-NEXT: 2.343
  %b   = arith.constant 2.343 : f64
  %b_p = arith.constant 1.000 : f64
  call @func_powff64(%b, %b_p) : (f64, f64) -> ()

  // CHECK-NEXT: 0.176171
  %c   = arith.constant 4.25 : f64
  %c_p = arith.constant -1.2  : f64
  call @func_powff64(%c, %c_p) : (f64, f64) -> ()

  // CHECK-NEXT: 1
  %d   = arith.constant 4.385 : f64
  %d_p = arith.constant 0.00 : f64
  call @func_powff64(%d, %d_p) : (f64, f64) -> ()

  // CHECK-NEXT: 6.62637
  %e    = arith.constant 4.835 : f64
  %e_p  = arith.constant 1.2 : f64
  call @func_powff64(%e, %e_p) : (f64, f64) -> ()

  // CHECK-NEXT: nan
  %f = arith.constant 1.0 : f64
  %f_p = arith.constant 0x7fffffffffffffff : f64
  call @func_powff64(%f, %f_p) : (f64, f64) -> ()

  // CHECK-NEXT: inf
  %g   = arith.constant 29385.0 : f64
  %g_p = arith.constant 23598.0 : f64
  call @func_powff64(%g, %g_p) : (f64, f64) -> ()

  // CHECK-NEXT: -nan
  %h = arith.constant 1.0 : f64
  %h_p = arith.constant 0xfff0000001000000 : f64
  call @func_powff64(%h, %h_p) : (f64, f64) -> ()

  // CHECK-NEXT: -nan
  %i = arith.constant 1.0 : f32
  %i_p = arith.constant 0xffffffff : f32
  call @func_powff32(%i, %i_p) : (f32, f32) -> ()

  // CHECK-NEXT: 1
  %j = arith.constant 0.000 : f32
  %j_r = math.powf %j, %j : f32
  vector.print %j_r : f32

  // CHECK-NEXT: 4
  %k = arith.constant -2.0 : f32
  %k_p = arith.constant 2.0 : f32
  %k_r = math.powf %k, %k_p : f32
  vector.print %k_r : f32

  // CHECK-NEXT: 0.25
  %l = arith.constant -2.0 : f32
  %l_p = arith.constant -2.0 : f32
  %l_r = math.powf %k, %l_p : f32
  vector.print %l_r : f32
  return
}

// -------------------------------------------------------------------------- //
// roundeven.
// -------------------------------------------------------------------------- //

func.func @func_roundeven32(%a : f32) {
  %b = math.roundeven %a : f32
  vector.print %b : f32
  return
}

func.func @func_roundeven32$bitcast_result_to_int(%a : f32) {
  %b = math.roundeven %a : f32
  %c = arith.bitcast %b : f32 to i32
  vector.print %c : i32
  return
}

func.func @func_roundeven32$vector(%a : vector<1xf32>) {
  %b = math.roundeven %a : vector<1xf32>
  vector.print %b : vector<1xf32>
  return
}

func.func @roundeven32() {
  %c0_25 = arith.constant 0.25 : f32
  %c0_5 = arith.constant 0.5 : f32
  %c0_75 = arith.constant 0.75 : f32
  %c1 = arith.constant 1.0 : f32
  %c1_25 = arith.constant 1.25 : f32
  %c1_5 = arith.constant 1.5 : f32
  %c1_75 = arith.constant 1.75 : f32
  %c2 = arith.constant 2.0 : f32
  %c2_25 = arith.constant 2.25 : f32
  %c2_5 = arith.constant 2.5 : f32
  %c2_75 = arith.constant 2.75 : f32
  %c3 = arith.constant 3.0 : f32
  %c3_25 = arith.constant 3.25 : f32
  %c3_5 = arith.constant 3.5 : f32
  %c3_75 = arith.constant 3.75 : f32

  %cNeg0_25 = arith.constant -0.25 : f32
  %cNeg0_5 = arith.constant -0.5 : f32
  %cNeg0_75 = arith.constant -0.75 : f32
  %cNeg1 = arith.constant -1.0 : f32
  %cNeg1_25 = arith.constant -1.25 : f32
  %cNeg1_5 = arith.constant -1.5 : f32
  %cNeg1_75 = arith.constant -1.75 : f32
  %cNeg2 = arith.constant -2.0 : f32
  %cNeg2_25 = arith.constant -2.25 : f32
  %cNeg2_5 = arith.constant -2.5 : f32
  %cNeg2_75 = arith.constant -2.75 : f32
  %cNeg3 = arith.constant -3.0 : f32
  %cNeg3_25 = arith.constant -3.25 : f32
  %cNeg3_5 = arith.constant -3.5 : f32
  %cNeg3_75 = arith.constant -3.75 : f32

  // CHECK-NEXT: 0
  call @func_roundeven32(%c0_25) : (f32) -> ()
  // CHECK-NEXT: 0
  call @func_roundeven32(%c0_5) : (f32) -> ()
  // CHECK-NEXT: 1
  call @func_roundeven32(%c0_75) : (f32) -> ()
  // CHECK-NEXT: 1
  call @func_roundeven32(%c1) : (f32) -> ()
  // CHECK-NEXT: 1
  call @func_roundeven32(%c1_25) : (f32) -> ()
  // CHECK-NEXT: 2
  call @func_roundeven32(%c1_5) : (f32) -> ()
  // CHECK-NEXT: 2
  call @func_roundeven32(%c1_75) : (f32) -> ()
  // CHECK-NEXT: 2
  call @func_roundeven32(%c2) : (f32) -> ()
  // CHECK-NEXT: 2
  call @func_roundeven32(%c2_25) : (f32) -> ()
  // CHECK-NEXT: 2
  call @func_roundeven32(%c2_5) : (f32) -> ()
  // CHECK-NEXT: 3
  call @func_roundeven32(%c2_75) : (f32) -> ()
  // CHECK-NEXT: 3
  call @func_roundeven32(%c3) : (f32) -> ()
  // CHECK-NEXT: 3
  call @func_roundeven32(%c3_25) : (f32) -> ()
  // CHECK-NEXT: 4
  call @func_roundeven32(%c3_5) : (f32) -> ()
  // CHECK-NEXT: 4
  call @func_roundeven32(%c3_75) : (f32) -> ()

  // CHECK-NEXT: -0
  call @func_roundeven32(%cNeg0_25) : (f32) -> ()
  // CHECK-NEXT: -0
  call @func_roundeven32(%cNeg0_5) : (f32) -> ()
  // CHECK-NEXT: -1
  call @func_roundeven32(%cNeg0_75) : (f32) -> ()
  // CHECK-NEXT: -1
  call @func_roundeven32(%cNeg1) : (f32) -> ()
  // CHECK-NEXT: -1
  call @func_roundeven32(%cNeg1_25) : (f32) -> ()
  // CHECK-NEXT: -2
  call @func_roundeven32(%cNeg1_5) : (f32) -> ()
  // CHECK-NEXT: -2
  call @func_roundeven32(%cNeg1_75) : (f32) -> ()
  // CHECK-NEXT: -2
  call @func_roundeven32(%cNeg2) : (f32) -> ()
  // CHECK-NEXT: -2
  call @func_roundeven32(%cNeg2_25) : (f32) -> ()
  // CHECK-NEXT: -2
  call @func_roundeven32(%cNeg2_5) : (f32) -> ()
  // CHECK-NEXT: -3
  call @func_roundeven32(%cNeg2_75) : (f32) -> ()
  // CHECK-NEXT: -3
  call @func_roundeven32(%cNeg3) : (f32) -> ()
  // CHECK-NEXT: -3
  call @func_roundeven32(%cNeg3_25) : (f32) -> ()
  // CHECK-NEXT: -4
  call @func_roundeven32(%cNeg3_5) : (f32) -> ()
  // CHECK-NEXT: -4
  call @func_roundeven32(%cNeg3_75) : (f32) -> ()

  // Special values: 0, -0, inf, -inf, nan, -nan
  %cNeg0 = arith.constant -0.0 : f32
  %c0 = arith.constant 0.0 : f32
  %cInfInt = arith.constant 0x7f800000 : i32
  %cInf = arith.bitcast %cInfInt : i32 to f32
  %cNegInfInt = arith.constant 0xff800000 : i32
  %cNegInf = arith.bitcast %cNegInfInt : i32 to f32
  %cNanInt = arith.constant 0x7fc00000 : i32
  %cNan = arith.bitcast %cNanInt : i32 to f32
  %cNegNanInt = arith.constant 0xffc00000 : i32
  %cNegNan = arith.bitcast %cNegNanInt : i32 to f32

  // CHECK-NEXT: -0
  call @func_roundeven32(%cNeg0) : (f32) -> ()
  // CHECK-NEXT: 0
  call @func_roundeven32(%c0) : (f32) -> ()
  // CHECK-NEXT: inf
  call @func_roundeven32(%cInf) : (f32) -> ()
  // CHECK-NEXT: -inf
  call @func_roundeven32(%cNegInf) : (f32) -> ()
  // Per IEEE 754-2008, sign is not required when printing a negative NaN, so
  // print as an int to ensure input NaN is left unchanged.
  // CHECK-NEXT: 2143289344
  // CHECK-NEXT: 2143289344
  call @func_roundeven32$bitcast_result_to_int(%cNan) : (f32) -> ()
  vector.print %cNanInt : i32
  // CHECK-NEXT: -4194304
  // CHECK-NEXT: -4194304
  call @func_roundeven32$bitcast_result_to_int(%cNegNan) : (f32) -> ()
  vector.print %cNegNanInt : i32


  // Values above and below 2^23 = 8388608
  %c8388606_5 = arith.constant 8388606.5 : f32
  %c8388607 = arith.constant 8388607.0 : f32
  %c8388607_5 = arith.constant 8388607.5 : f32
  %c8388608 = arith.constant 8388608.0 : f32
  %c8388609 = arith.constant 8388609.0 : f32

  // Bitcast result to int to avoid printing in scientific notation,
  // which does not display all significant digits.

  // CHECK-NEXT: 1258291196
  // hex: 0x4AFFFFFC
  call @func_roundeven32$bitcast_result_to_int(%c8388606_5) : (f32) -> ()
  // CHECK-NEXT: 1258291198
  // hex: 0x4AFFFFFE
  call @func_roundeven32$bitcast_result_to_int(%c8388607) : (f32) -> ()
  // CHECK-NEXT: 1258291200
  // hex: 0x4B000000
  call @func_roundeven32$bitcast_result_to_int(%c8388607_5) : (f32) -> ()
  // CHECK-NEXT: 1258291200
  // hex: 0x4B000000
  call @func_roundeven32$bitcast_result_to_int(%c8388608) : (f32) -> ()
  // CHECK-NEXT: 1258291201
  // hex: 0x4B000001
  call @func_roundeven32$bitcast_result_to_int(%c8388609) : (f32) -> ()


  // Check that vector type works
  %cVec = arith.constant dense<[0.5]> : vector<1xf32>
  // CHECK-NEXT: ( 0 )
  call @func_roundeven32$vector(%cVec) : (vector<1xf32>) -> ()
  return
}

func.func @func_roundeven64(%a : f64) {
  %b = math.roundeven %a : f64
  vector.print %b : f64
  return
}

func.func @func_roundeven64$bitcast_result_to_int(%a : f64) {
  %b = math.roundeven %a : f64
  %c = arith.bitcast %b : f64 to i64
  vector.print %c : i64
  return
}

func.func @func_roundeven64$vector(%a : vector<1xf64>) {
  %b = math.roundeven %a : vector<1xf64>
  vector.print %b : vector<1xf64>
  return
}

func.func @roundeven64() {
  %c0_25 = arith.constant 0.25 : f64
  %c0_5 = arith.constant 0.5 : f64
  %c0_75 = arith.constant 0.75 : f64
  %c1 = arith.constant 1.0 : f64
  %c1_25 = arith.constant 1.25 : f64
  %c1_5 = arith.constant 1.5 : f64
  %c1_75 = arith.constant 1.75 : f64
  %c2 = arith.constant 2.0 : f64
  %c2_25 = arith.constant 2.25 : f64
  %c2_5 = arith.constant 2.5 : f64
  %c2_75 = arith.constant 2.75 : f64
  %c3 = arith.constant 3.0 : f64
  %c3_25 = arith.constant 3.25 : f64
  %c3_5 = arith.constant 3.5 : f64
  %c3_75 = arith.constant 3.75 : f64

  %cNeg0_25 = arith.constant -0.25 : f64
  %cNeg0_5 = arith.constant -0.5 : f64
  %cNeg0_75 = arith.constant -0.75 : f64
  %cNeg1 = arith.constant -1.0 : f64
  %cNeg1_25 = arith.constant -1.25 : f64
  %cNeg1_5 = arith.constant -1.5 : f64
  %cNeg1_75 = arith.constant -1.75 : f64
  %cNeg2 = arith.constant -2.0 : f64
  %cNeg2_25 = arith.constant -2.25 : f64
  %cNeg2_5 = arith.constant -2.5 : f64
  %cNeg2_75 = arith.constant -2.75 : f64
  %cNeg3 = arith.constant -3.0 : f64
  %cNeg3_25 = arith.constant -3.25 : f64
  %cNeg3_5 = arith.constant -3.5 : f64
  %cNeg3_75 = arith.constant -3.75 : f64

  // CHECK-NEXT: 0
  call @func_roundeven64(%c0_25) : (f64) -> ()
  // CHECK-NEXT: 0
  call @func_roundeven64(%c0_5) : (f64) -> ()
  // CHECK-NEXT: 1
  call @func_roundeven64(%c0_75) : (f64) -> ()
  // CHECK-NEXT: 1
  call @func_roundeven64(%c1) : (f64) -> ()
  // CHECK-NEXT: 1
  call @func_roundeven64(%c1_25) : (f64) -> ()
  // CHECK-NEXT: 2
  call @func_roundeven64(%c1_5) : (f64) -> ()
  // CHECK-NEXT: 2
  call @func_roundeven64(%c1_75) : (f64) -> ()
  // CHECK-NEXT: 2
  call @func_roundeven64(%c2) : (f64) -> ()
  // CHECK-NEXT: 2
  call @func_roundeven64(%c2_25) : (f64) -> ()
  // CHECK-NEXT: 2
  call @func_roundeven64(%c2_5) : (f64) -> ()
  // CHECK-NEXT: 3
  call @func_roundeven64(%c2_75) : (f64) -> ()
  // CHECK-NEXT: 3
  call @func_roundeven64(%c3) : (f64) -> ()
  // CHECK-NEXT: 3
  call @func_roundeven64(%c3_25) : (f64) -> ()
  // CHECK-NEXT: 4
  call @func_roundeven64(%c3_5) : (f64) -> ()
  // CHECK-NEXT: 4
  call @func_roundeven64(%c3_75) : (f64) -> ()

  // CHECK-NEXT: -0
  call @func_roundeven64(%cNeg0_25) : (f64) -> ()
  // CHECK-NEXT: -0
  call @func_roundeven64(%cNeg0_5) : (f64) -> ()
  // CHECK-NEXT: -1
  call @func_roundeven64(%cNeg0_75) : (f64) -> ()
  // CHECK-NEXT: -1
  call @func_roundeven64(%cNeg1) : (f64) -> ()
  // CHECK-NEXT: -1
  call @func_roundeven64(%cNeg1_25) : (f64) -> ()
  // CHECK-NEXT: -2
  call @func_roundeven64(%cNeg1_5) : (f64) -> ()
  // CHECK-NEXT: -2
  call @func_roundeven64(%cNeg1_75) : (f64) -> ()
  // CHECK-NEXT: -2
  call @func_roundeven64(%cNeg2) : (f64) -> ()
  // CHECK-NEXT: -2
  call @func_roundeven64(%cNeg2_25) : (f64) -> ()
  // CHECK-NEXT: -2
  call @func_roundeven64(%cNeg2_5) : (f64) -> ()
  // CHECK-NEXT: -3
  call @func_roundeven64(%cNeg2_75) : (f64) -> ()
  // CHECK-NEXT: -3
  call @func_roundeven64(%cNeg3) : (f64) -> ()
  // CHECK-NEXT: -3
  call @func_roundeven64(%cNeg3_25) : (f64) -> ()
  // CHECK-NEXT: -4
  call @func_roundeven64(%cNeg3_5) : (f64) -> ()
  // CHECK-NEXT: -4
  call @func_roundeven64(%cNeg3_75) : (f64) -> ()

  // Special values: 0, -0, inf, -inf, nan, -nan
  %cNeg0 = arith.constant -0.0 : f64
  %c0 = arith.constant 0.0 : f64
  %cInfInt = arith.constant 0x7FF0000000000000 : i64
  %cInf = arith.bitcast %cInfInt : i64 to f64
  %cNegInfInt = arith.constant 0xFFF0000000000000 : i64
  %cNegInf = arith.bitcast %cNegInfInt : i64 to f64
  %cNanInt = arith.constant 0x7FF0000000000001 : i64
  %cNan = arith.bitcast %cNanInt : i64 to f64
  %cNegNanInt = arith.constant 0xFFF0000000000001 : i64
  %cNegNan = arith.bitcast %cNegNanInt : i64 to f64

  // CHECK-NEXT: -0
  call @func_roundeven64(%cNeg0) : (f64) -> ()
  // CHECK-NEXT: 0
  call @func_roundeven64(%c0) : (f64) -> ()
  // CHECK-NEXT: inf
  call @func_roundeven64(%cInf) : (f64) -> ()
  // CHECK-NEXT: -inf
  call @func_roundeven64(%cNegInf) : (f64) -> ()

  // Values above and below 2^52 = 4503599627370496
  %c4503599627370494_5 = arith.constant 4503599627370494.5 : f64
  %c4503599627370495 = arith.constant 4503599627370495.0 : f64
  %c4503599627370495_5 = arith.constant 4503599627370495.5 : f64
  %c4503599627370496 = arith.constant 4503599627370496.0 : f64
  %c4503599627370497 = arith.constant 4503599627370497.0 : f64

  // Bitcast result to int to avoid printing in scientific notation,
  // which does not display all significant digits.

  // CHECK-NEXT: 4841369599423283196
  // hex: 0x432ffffffffffffc
  call @func_roundeven64$bitcast_result_to_int(%c4503599627370494_5) : (f64) -> ()
  // CHECK-NEXT: 4841369599423283198
  // hex: 0x432ffffffffffffe
  call @func_roundeven64$bitcast_result_to_int(%c4503599627370495) : (f64) -> ()
  // CHECK-NEXT: 4841369599423283200
  // hex: 0x4330000000000000
  call @func_roundeven64$bitcast_result_to_int(%c4503599627370495_5) : (f64) -> ()
  // CHECK-NEXT: 4841369599423283200
  // hex: 0x10000000000000
  call @func_roundeven64$bitcast_result_to_int(%c4503599627370496) : (f64) -> ()
  // CHECK-NEXT: 4841369599423283201
  // hex: 0x10000000000001
  call @func_roundeven64$bitcast_result_to_int(%c4503599627370497) : (f64) -> ()

  // Check that vector type works
  %cVec = arith.constant dense<[0.5]> : vector<1xf64>
  // CHECK-NEXT: ( 0 )
  call @func_roundeven64$vector(%cVec) : (vector<1xf64>) -> ()
  return
}

func.func @roundeven() {
  call @roundeven32() : () -> ()
  call @roundeven64() : () -> ()
  return
}

// -------------------------------------------------------------------------- //
// Sinh.
// -------------------------------------------------------------------------- //

func.func @sinh_f32(%a : f32) {
  %r = math.sinh %a : f32
  vector.print %r : f32
  return
}

func.func @sinh_4xf32(%a : vector<4xf32>) {
  %r = math.sinh %a : vector<4xf32>
  vector.print %r : vector<4xf32>
  return
}

func.func @sinh_8xf32(%a : vector<8xf32>) {
  %r = math.sinh %a : vector<8xf32>
  vector.print %r : vector<8xf32>
  return
}

func.func @sinh() {
  // CHECK: 1.60192
  %f0 = arith.constant 1.25 : f32
  call @sinh_f32(%f0) : (f32) -> ()

  // CHECK: 0.252612, 0.822317, 1.1752, 1.60192
  %v1 = arith.constant dense<[0.25, 0.75, 1.0, 1.25]> : vector<4xf32>
  call @sinh_4xf32(%v1) : (vector<4xf32>) -> ()

  // CHECK: 0.100167, 0.201336, 0.30452, 0.410752, 0.521095, 0.636654, 0.758584, 0.888106
  %v2 = arith.constant dense<[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]> : vector<8xf32>
  call @sinh_8xf32(%v2) : (vector<8xf32>) -> ()

  // CHECK: -0.100167, -0.201336, -0.30452, -0.410752, -0.521095, -0.636654, -0.758584, -0.888106
  %v3 = arith.constant dense<[-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8]> : vector<8xf32>
  call @sinh_8xf32(%v3) : (vector<8xf32>) -> ()

  // CHECK: nan
  %nan = arith.constant 0x7fc00000 : f32
  call @sinh_f32(%nan) : (f32) -> ()

 return
}

// -------------------------------------------------------------------------- //
// Cosh.
// -------------------------------------------------------------------------- //

func.func @cosh_f32(%a : f32) {
  %r = math.cosh %a : f32
  vector.print %r : f32
  return
}

func.func @cosh_4xf32(%a : vector<4xf32>) {
  %r = math.cosh %a : vector<4xf32>
  vector.print %r : vector<4xf32>
  return
}

func.func @cosh_8xf32(%a : vector<8xf32>) {
  %r = math.cosh %a : vector<8xf32>
  vector.print %r : vector<8xf32>
  return
}

func.func @cosh() {
  // CHECK: 1.88842
  %f0 = arith.constant 1.25 : f32
  call @cosh_f32(%f0) : (f32) -> ()

  // CHECK: 1.03141, 1.29468, 1.54308, 1.88842
  %v1 = arith.constant dense<[0.25, 0.75, 1.0, 1.25]> : vector<4xf32>
  call @cosh_4xf32(%v1) : (vector<4xf32>) -> ()

  // CHECK: 1.005, 1.02007, 1.04534, 1.08107, 1.12763, 1.18547, 1.25517, 1.33743
  %v2 = arith.constant dense<[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]> : vector<8xf32>
  call @cosh_8xf32(%v2) : (vector<8xf32>) -> ()

  // CHECK: 1.005, 1.02007, 1.04534, 1.08107, 1.12763, 1.18547, 1.25517, 1.33743
  %v3 = arith.constant dense<[-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8]> : vector<8xf32>
  call @cosh_8xf32(%v3) : (vector<8xf32>) -> ()

  // CHECK: nan
  %nan = arith.constant 0x7fc00000 : f32
  call @cosh_f32(%nan) : (f32) -> ()

 return
}

// -------------------------------------------------------------------------- //
// Tanh.
// -------------------------------------------------------------------------- //

func.func @tanh_8xf32(%a : vector<8xf32>) {
  %r = math.tanh %a : vector<8xf32>
  vector.print %r : vector<8xf32>
  return
}

func.func @tanh() {
  // CHECK: -1, -0.761594, -0.291313, 0, 0.291313, 0.761594, 1, 1
  %v3 = arith.constant dense<[0xff800000, -1.0, -0.3, 0.0, 0.3, 1.0, 10.0, 0x7f800000]> : vector<8xf32>
  call @tanh_8xf32(%v3) : (vector<8xf32>) -> ()

 return
}

// -------------------------------------------------------------------------- //
// Asinh.
// -------------------------------------------------------------------------- //

func.func @asinh_f32(%a : f32) {
  %r = math.asinh %a : f32
  vector.print %r : f32
  return
}

func.func @asinh_3xf32(%a : vector<3xf32>) {
  %r = math.asinh %a : vector<3xf32>
  vector.print %r : vector<3xf32>
  return
}

func.func @asinh() {
  // CHECK: 0
  %zero = arith.constant 0.0 : f32
  call @asinh_f32(%zero) : (f32) -> ()

  // CHECK: 0.881374
  %cst1 = arith.constant 1.0 : f32
  call @asinh_f32(%cst1) : (f32) -> ()

  // CHECK: -0.881374
  %cst2 = arith.constant -1.0 : f32
  call @asinh_f32(%cst2) : (f32) -> ()

  // CHECK: 1.81845
  %cst3 = arith.constant 3.0 : f32
  call @asinh_f32(%cst3) : (f32) -> ()

  // CHECK: 0.247466, 0.790169, 1.44364
  %vec_x = arith.constant dense<[0.25, 0.875, 2.0]> : vector<3xf32>
  call @asinh_3xf32(%vec_x) : (vector<3xf32>) -> ()

  return
}

// -------------------------------------------------------------------------- //
// Acosh.
// -------------------------------------------------------------------------- //

func.func @acosh_f32(%a : f32) {
  %r = math.acosh %a : f32
  vector.print %r : f32
  return
}

func.func @acosh_3xf32(%a : vector<3xf32>) {
  %r = math.acosh %a : vector<3xf32>
  vector.print %r : vector<3xf32>
  return
}

func.func @acosh() {
  // CHECK: 0
  %zero = arith.constant 1.0 : f32
  call @acosh_f32(%zero) : (f32) -> ()

  // CHECK: 1.31696
  %cst1 = arith.constant 2.0 : f32
  call @acosh_f32(%cst1) : (f32) -> ()

  // CHECK: 2.99322
  %cst2 = arith.constant 10.0 : f32
  call @acosh_f32(%cst2) : (f32) -> ()

  // CHECK: 0.962424, 1.76275, 2.47789
  %vec_x = arith.constant dense<[1.5, 3.0, 6.0]> : vector<3xf32>
  call @acosh_3xf32(%vec_x) : (vector<3xf32>) -> ()

  return
}

// -------------------------------------------------------------------------- //
// Atanh.
// -------------------------------------------------------------------------- //

func.func @atanh_f32(%a : f32) {
  %r = math.atanh %a : f32
  vector.print %r : f32
  return
}

func.func @atanh_3xf32(%a : vector<3xf32>) {
  %r = math.atanh %a : vector<3xf32>
  vector.print %r : vector<3xf32>
  return
}

func.func @atanh() {
  // CHECK: 0
  %zero = arith.constant 0.0 : f32
  call @atanh_f32(%zero) : (f32) -> ()

  // CHECK: 0.549306
  %cst1 = arith.constant 0.5 : f32
  call @atanh_f32(%cst1) : (f32) -> ()

  // CHECK: -0.549306
  %cst2 = arith.constant -0.5 : f32
  call @atanh_f32(%cst2) : (f32) -> ()

  // CHECK: inf
  %cst3 = arith.constant 1.0 : f32
  call @atanh_f32(%cst3) : (f32) -> ()

  // CHECK: 0.255413, 0.394229, 2.99448
  %vec_x = arith.constant dense<[0.25, 0.375, 0.995]> : vector<3xf32>
  call @atanh_3xf32(%vec_x) : (vector<3xf32>) -> ()

  return
}

// -------------------------------------------------------------------------- //
// Rsqrt.
// -------------------------------------------------------------------------- //

func.func @rsqrt_f32(%a : f32) {
  %r = math.rsqrt %a : f32
  vector.print %r : f32
  return
}

func.func @rsqrt_3xf32(%a : vector<3xf32>) {
  %r = math.rsqrt %a : vector<3xf32>
  vector.print %r : vector<3xf32>
  return
}

func.func @rsqrt() {
  // CHECK: 1
  %zero = arith.constant 1.0 : f32
  call @rsqrt_f32(%zero) : (f32) -> ()

  // CHECK: 0.707107
  %cst1 = arith.constant 2.0 : f32
  call @rsqrt_f32(%cst1) : (f32) -> ()

  // CHECK: inf
  %cst2 = arith.constant 0.0 : f32
  call @rsqrt_f32(%cst2) : (f32) -> ()

  // CHECK: nan
  %cst3 = arith.constant -1.0 : f32
  call @rsqrt_f32(%cst3) : (f32) -> ()

  // CHECK: 0.5, 1.41421, 0.57735
  %vec_x = arith.constant dense<[4.0, 0.5, 3.0]> : vector<3xf32>
  call @rsqrt_3xf32(%vec_x) : (vector<3xf32>) -> ()

  return
}

func.func @main() {
  call @exp2f() : () -> ()
  call @roundf() : () -> ()
  call @powf() : () -> ()
  call @roundeven() : () -> ()
  call @sinh() : () -> ()
  call @cosh() : () -> ()
  call @tanh() : () -> ()
  call @asinh() : () -> ()
  call @acosh() : () -> ()
  call @atanh() : () -> ()
  call @rsqrt() : () -> ()
  return
}
