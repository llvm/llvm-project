// RUN:   mlir-opt %s -pass-pipeline="builtin.module(func.func(test-expand-math,convert-arith-to-llvm),convert-vector-to-scf,convert-scf-to-cf,convert-cf-to-llvm,convert-vector-to-llvm,func.func(convert-math-to-llvm),convert-func-to-llvm,reconcile-unrealized-casts)" \
// RUN: | mlir-cpu-runner                                                      \
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

func.func @powf() {
  // CHECK-NEXT: 16
  %a   = arith.constant 4.0 : f64
  %a_p = arith.constant 2.0 : f64
  call @func_powff64(%a, %a_p) : (f64, f64) -> ()

  // CHECK-NEXT: nan
  %b   = arith.constant -3.0 : f64
  %b_p = arith.constant 3.0 : f64
  call @func_powff64(%b, %b_p) : (f64, f64) -> ()

  // CHECK-NEXT: 2.343
  %c   = arith.constant 2.343 : f64
  %c_p = arith.constant 1.000 : f64
  call @func_powff64(%c, %c_p) : (f64, f64) -> ()

  // CHECK-NEXT: 0.176171
  %d   = arith.constant 4.25 : f64
  %d_p = arith.constant -1.2  : f64
  call @func_powff64(%d, %d_p) : (f64, f64) -> ()

  // CHECK-NEXT: 1
  %e   = arith.constant 4.385 : f64
  %e_p = arith.constant 0.00 : f64
  call @func_powff64(%e, %e_p) : (f64, f64) -> ()

  // CHECK-NEXT: 6.62637
  %f    = arith.constant 4.835 : f64
  %f_p  = arith.constant 1.2 : f64
  call @func_powff64(%f, %f_p) : (f64, f64) -> ()

  // CHECK-NEXT: nan
  %g    = arith.constant 0xff80000000000000 : f64
  call @func_powff64(%g, %g) : (f64, f64) -> ()

  // CHECK-NEXT: nan
  %h = arith.constant 0x7fffffffffffffff : f64
  call @func_powff64(%h, %h) : (f64, f64) -> ()

  // CHECK-NEXT: nan
  %i = arith.constant 1.0 : f64
  call @func_powff64(%i, %h) : (f64, f64) -> ()

  // CHECK-NEXT: inf
  %j   = arith.constant 29385.0 : f64
  %j_p = arith.constant 23598.0 : f64
  call @func_powff64(%j, %j_p) : (f64, f64) -> ()
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

func.func @main() {
  call @exp2f() : () -> ()
  call @roundf() : () -> ()
  call @powf() : () -> ()
  call @roundeven() : () -> ()
  return
}
