// RUN:   mlir-opt %s -pass-pipeline="builtin.module(func.func(test-math-polynomial-approximation),convert-vector-to-scf,convert-scf-to-cf,convert-vector-to-llvm,convert-to-llvm,reconcile-unrealized-casts)" \
// RUN: | mlir-runner                                                      \
// RUN:     -e main -entry-point-result=void -O0                               \
// RUN:     -shared-libs=%mlir_c_runner_utils  \
// RUN:     -shared-libs=%mlir_runner_utils    \
// RUN: | FileCheck %s

// -------------------------------------------------------------------------- //
// Tanh.
// -------------------------------------------------------------------------- //

func.func @tanh_f32(%a : f32) {
  %r = math.tanh %a : f32
  vector.print %r : f32
  return
}

func.func @tanh_4xf32(%a : vector<4xf32>) {
  %r = math.tanh %a : vector<4xf32>
  vector.print %r : vector<4xf32>
  return
}

func.func @tanh_8xf32(%a : vector<8xf32>) {
  %r = math.tanh %a : vector<8xf32>
  vector.print %r : vector<8xf32>
  return
}

func.func @tanh() {
  // CHECK: 0.848284
  %f0 = arith.constant 1.25 : f32
  call @tanh_f32(%f0) : (f32) -> ()

  // CHECK: 0.244919, 0.635149, 0.761594, 0.848284
  %v1 = arith.constant dense<[0.25, 0.75, 1.0, 1.25]> : vector<4xf32>
  call @tanh_4xf32(%v1) : (vector<4xf32>) -> ()

  // CHECK: 0.099668, 0.197375, 0.291313, 0.379949, 0.462117, 0.53705, 0.604368, 0.664037
  %v2 = arith.constant dense<[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]> : vector<8xf32>
  call @tanh_8xf32(%v2) : (vector<8xf32>) -> ()

  // CHECK: nan
  %nan = arith.constant 0x7fc00000 : f32
  call @tanh_f32(%nan) : (f32) -> ()

 return
}

// -------------------------------------------------------------------------- //
// Log.
// -------------------------------------------------------------------------- //

func.func @log_f32(%a : f32) {
  %r = math.log %a : f32
  vector.print %r : f32
  return
}

func.func @log_4xf32(%a : vector<4xf32>) {
  %r = math.log %a : vector<4xf32>
  vector.print %r : vector<4xf32>
  return
}

func.func @log_8xf32(%a : vector<8xf32>) {
  %r = math.log %a : vector<8xf32>
  vector.print %r : vector<8xf32>
  return
}

func.func @log() {
  // CHECK: 2.64704
  %f1 = arith.constant 14.112233 : f32
  call @log_f32(%f1) : (f32) -> ()

  // CHECK: -1.38629, -0.287682, 0, 0.223144
  %v1 = arith.constant dense<[0.25, 0.75, 1.0, 1.25]> : vector<4xf32>
  call @log_4xf32(%v1) : (vector<4xf32>) -> ()

  // CHECK: -2.30259, -1.60944, -1.20397, -0.916291, -0.693147, -0.510826, -0.356675, -0.223144
  %v2 = arith.constant dense<[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]> : vector<8xf32>
  call @log_8xf32(%v2) : (vector<8xf32>) -> ()

  // CHECK: -inf
  %zero = arith.constant 0.0 : f32
  call @log_f32(%zero) : (f32) -> ()

  // CHECK: nan
  %nan = arith.constant 0x7fc00000 : f32
  call @log_f32(%nan) : (f32) -> ()

  // CHECK: inf
  %inf = arith.constant 0x7f800000 : f32
  call @log_f32(%inf) : (f32) -> ()

  // CHECK: -inf, nan, inf, 0.693147
  %special_vec = arith.constant dense<[0.0, -1.0, 0x7f800000, 2.0]> : vector<4xf32>
  call @log_4xf32(%special_vec) : (vector<4xf32>) -> ()

  return
}

func.func @log2_f32(%a : f32) {
  %r = math.log2 %a : f32
  vector.print %r : f32
  return
}

func.func @log2_4xf32(%a : vector<4xf32>) {
  %r = math.log2 %a : vector<4xf32>
  vector.print %r : vector<4xf32>
  return
}

func.func @log2_8xf32(%a : vector<8xf32>) {
  %r = math.log2 %a : vector<8xf32>
  vector.print %r : vector<8xf32>
  return
}

func.func @log2() {
  // CHECK: 3.81887
  %f0 = arith.constant 14.112233 : f32
  call @log2_f32(%f0) : (f32) -> ()

  // CHECK: -2, -0.415037, 0, 0.321928
  %v1 = arith.constant dense<[0.25, 0.75, 1.0, 1.25]> : vector<4xf32>
  call @log2_4xf32(%v1) : (vector<4xf32>) -> ()

  // CHECK: -3.32193, -2.32193, -1.73697, -1.32193, -1, -0.736966, -0.514573, -0.321928
  %v2 = arith.constant dense<[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]> : vector<8xf32>
  call @log2_8xf32(%v2) : (vector<8xf32>) -> ()

  // CHECK: -inf
  %zero = arith.constant 0.0 : f32
  call @log2_f32(%zero) : (f32) -> ()

  // CHECK: nan
  %neg_one = arith.constant -1.0 : f32
  call @log2_f32(%neg_one) : (f32) -> ()

  // CHECK: inf
  %inf = arith.constant 0x7f800000 : f32
  call @log2_f32(%inf) : (f32) -> ()

  // CHECK: -inf, nan, inf, 1.58496
  %special_vec = arith.constant dense<[0.0, -1.0, 0x7f800000, 3.0]> : vector<4xf32>
  call @log2_4xf32(%special_vec) : (vector<4xf32>) -> ()

  return
}

func.func @log1p_f32(%a : f32) {
  %r = math.log1p %a : f32
  vector.print %r : f32
  return
}

func.func @log1p_4xf32(%a : vector<4xf32>) {
  %r = math.log1p %a : vector<4xf32>
  vector.print %r : vector<4xf32>
  return
}

func.func @log1p_8xf32(%a : vector<8xf32>) {
  %r = math.log1p %a : vector<8xf32>
  vector.print %r : vector<8xf32>
  return
}

func.func @log1p() {
  // CHECK: 0.00995033
  %f0 = arith.constant 0.01 : f32
  call @log1p_f32(%f0) : (f32) -> ()


  // CHECK: -4.60517, -0.693147, 0, 1.38629
  %v1 = arith.constant dense<[-0.99, -0.5, 0.0, 3.0]> : vector<4xf32>
  call @log1p_4xf32(%v1) : (vector<4xf32>) -> ()

  // CHECK: 0.0953102, 0.182322, 0.262364, 0.336472, 0.405465, 0.470004, 0.530628, 0.587787
  %v2 = arith.constant dense<[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]> : vector<8xf32>
  call @log1p_8xf32(%v2) : (vector<8xf32>) -> ()

  // CHECK: -inf
  %neg_one = arith.constant -1.0 : f32
  call @log1p_f32(%neg_one) : (f32) -> ()

  // CHECK: nan
  %neg_two = arith.constant -2.0 : f32
  call @log1p_f32(%neg_two) : (f32) -> ()

  // CHECK: inf
  %inf = arith.constant 0x7f800000 : f32
  call @log1p_f32(%inf) : (f32) -> ()

  // CHECK: -inf, nan, inf, 9.99995e-06
  %special_vec = arith.constant dense<[-1.0, -1.1, 0x7f800000, 0.00001]> : vector<4xf32>
  call @log1p_4xf32(%special_vec) : (vector<4xf32>) -> ()

  return
}

// -------------------------------------------------------------------------- //
// Erf.
// -------------------------------------------------------------------------- //
func.func @erf_f32(%a : f32) {
  %r = math.erf %a : f32
  vector.print %r : f32
  return
}

func.func @erf_4xf32(%a : vector<4xf32>) {
  %r = math.erf %a : vector<4xf32>
  vector.print %r : vector<4xf32>
  return
}

func.func @erf() {
  // CHECK: -0.000274406
  %val1 = arith.constant -2.431864e-4 : f32
  call @erf_f32(%val1) : (f32) -> ()

  // CHECK: 0.742095
  %val2 = arith.constant 0.79999 : f32
  call @erf_f32(%val2) : (f32) -> ()

  // CHECK: 0.742101
  %val3 = arith.constant 0.8 : f32
  call @erf_f32(%val3) : (f32) -> ()

  // CHECK: 0.995322
  %val4 = arith.constant 1.99999 : f32
  call @erf_f32(%val4) : (f32) -> ()

  // CHECK: 0.995322
  %val5 = arith.constant 2.0 : f32
  call @erf_f32(%val5) : (f32) -> ()

  // CHECK: 1
  %val6 = arith.constant 3.74999 : f32
  call @erf_f32(%val6) : (f32) -> ()

  // CHECK: 1
  %val7 = arith.constant 3.75 : f32
  call @erf_f32(%val7) : (f32) -> ()

  // CHECK: -1
  %negativeInf = arith.constant 0xff800000 : f32
  call @erf_f32(%negativeInf) : (f32) -> ()

  // CHECK: -1, -1, -0.913759, -0.731446
  %vecVals1 = arith.constant dense<[-3.4028235e+38, -4.54318, -1.2130899, -7.8234202e-01]> : vector<4xf32>
  call @erf_4xf32(%vecVals1) : (vector<4xf32>) -> ()

  // CHECK: -1.3264e-38, 0, 1.3264e-38, 0.121319
  %vecVals2 = arith.constant dense<[-1.1754944e-38, 0.0, 1.1754944e-38, 1.0793410e-01]> : vector<4xf32>
  call @erf_4xf32(%vecVals2) : (vector<4xf32>) -> ()

  // CHECK: 0.919477, 0.999069, 1, 1
  %vecVals3 = arith.constant dense<[1.23578, 2.34093, 3.82342, 3.4028235e+38]> : vector<4xf32>
  call @erf_4xf32(%vecVals3) : (vector<4xf32>) -> ()

  // CHECK: 1
  %inf = arith.constant 0x7f800000 : f32
  call @erf_f32(%inf) : (f32) -> ()

  // CHECK: nan
  %nan = arith.constant 0x7fc00000 : f32
  call @erf_f32(%nan) : (f32) -> ()

  return
}

// -------------------------------------------------------------------------- //
// Erfc.
// -------------------------------------------------------------------------- //
func.func @erfc_f32(%a : f32) {
  %r = math.erfc %a : f32
  vector.print %r : f32
  return
}

func.func @erfc_4xf32(%a : vector<4xf32>) {
  %r = math.erfc %a : vector<4xf32>
  vector.print %r : vector<4xf32>
  return
}

func.func @erfc() {
  // CHECK: 1.00027
  %val1 = arith.constant -2.431864e-4 : f32
  call @erfc_f32(%val1) : (f32) -> ()

  // CHECK: 0.257905
  %val2 = arith.constant 0.79999 : f32
  call @erfc_f32(%val2) : (f32) -> ()

  // CHECK: 0.257899
  %val3 = arith.constant 0.8 : f32
  call @erfc_f32(%val3) : (f32) -> ()

  // CHECK: 0.00467794
  %val4 = arith.constant 1.99999 : f32
  call @erfc_f32(%val4) : (f32) -> ()

  // CHECK: 0.00467774
  %val5 = arith.constant 2.0 : f32
  call @erfc_f32(%val5) : (f32) -> ()

  // CHECK: 1.13736e-07
  %val6 = arith.constant 3.74999 : f32
  call @erfc_f32(%val6) : (f32) -> ()

  // CHECK: 1.13727e-07
  %val7 = arith.constant 3.75 : f32
  call @erfc_f32(%val7) : (f32) -> ()

  // CHECK: 2
  %negativeInf = arith.constant 0xff800000 : f32
  call @erfc_f32(%negativeInf) : (f32) -> ()

  // CHECK: 2, 2, 1.91376, 1.73145
  %vecVals1 = arith.constant dense<[-3.4028235e+38, -4.54318, -1.2130899, -7.8234202e-01]> : vector<4xf32>
  call @erfc_4xf32(%vecVals1) : (vector<4xf32>) -> ()

  // CHECK: 1, 1, 1, 0.878681
  %vecVals2 = arith.constant dense<[-1.1754944e-38, 0.0, 1.1754944e-38, 1.0793410e-01]> : vector<4xf32>
  call @erfc_4xf32(%vecVals2) : (vector<4xf32>) -> ()

  // CHECK: 0.0805235, 0.000931045, 6.40418e-08, 0
  %vecVals3 = arith.constant dense<[1.23578, 2.34093, 3.82342, 3.4028235e+38]> : vector<4xf32>
  call @erfc_4xf32(%vecVals3) : (vector<4xf32>) -> ()

  // CHECK: 0
  %inf = arith.constant 0x7f800000 : f32
  call @erfc_f32(%inf) : (f32) -> ()

  // CHECK: nan
  %nan = arith.constant 0x7fc00000 : f32
  call @erfc_f32(%nan) : (f32) -> ()

  return
}

// -------------------------------------------------------------------------- //
// Exp.
// -------------------------------------------------------------------------- //
func.func @exp_f32(%a : f32) {
  %r = math.exp %a : f32
  vector.print %r : f32
  return
}

func.func @exp_4xf32(%a : vector<4xf32>) {
  %r = math.exp %a : vector<4xf32>
  vector.print %r : vector<4xf32>
  return
}

func.func @exp() {
  // CHECK: 2.71828
  %f0 = arith.constant 1.0 : f32
  call @exp_f32(%f0) : (f32) -> ()

  // CHECK: 0.778801, 2.117, 2.71828, 3.85743
  %v1 = arith.constant dense<[-0.25, 0.75, 1.0, 1.35]> : vector<4xf32>
  call @exp_4xf32(%v1) : (vector<4xf32>) -> ()

  // CHECK: 1
  %zero = arith.constant 0.0 : f32
  call @exp_f32(%zero) : (f32) -> ()

  // CHECK: 0, 1.38879e-11, 7.20049e+10, inf
  %special_vec = arith.constant dense<[-89.0, -25.0, 25.0, 89.0]> : vector<4xf32>
  call @exp_4xf32(%special_vec) : (vector<4xf32>) -> ()

  // CHECK: inf
  %inf = arith.constant 0x7f800000 : f32
  call @exp_f32(%inf) : (f32) -> ()

  // CHECK: 0
  %negative_inf = arith.constant 0xff800000 : f32
  call @exp_f32(%negative_inf) : (f32) -> ()

  // CHECK: nan
  %nan = arith.constant 0x7fc00000 : f32
  call @exp_f32(%nan) : (f32) -> ()

  return
}

func.func @expm1_f32(%a : f32) {
  %r = math.expm1 %a : f32
  vector.print %r : f32
  return
}

func.func @expm1_3xf32(%a : vector<3xf32>) {
  %r = math.expm1 %a : vector<3xf32>
  vector.print %r : vector<3xf32>
  return
}

func.func @expm1_4xf32(%a : vector<4xf32>) {
  %r = math.expm1 %a : vector<4xf32>
  vector.print %r : vector<4xf32>
  return
}

func.func @expm1_8xf32(%a : vector<8xf32>) {
  %r = math.expm1 %a : vector<8xf32>
  vector.print %r : vector<8xf32>
  return
}

func.func @expm1() {
  // CHECK: 1e-10
  %f0 = arith.constant 1.0e-10 : f32
  call @expm1_f32(%f0) : (f32) -> ()

  // CHECK: -0.00995017, 0.0100502, 0.648721, 6.38906
  %v1 = arith.constant dense<[-0.01, 0.01, 0.5, 2.0]> : vector<4xf32>
  call @expm1_4xf32(%v1) : (vector<4xf32>) -> ()

  // CHECK: -0.181269, 0, 0.221403, 0.491825, 0.822119, 1.22554, 1.71828, 2.32012
  %v2 = arith.constant dense<[-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]> : vector<8xf32>
  call @expm1_8xf32(%v2) : (vector<8xf32>) -> ()

  // CHECK: -1
  %neg_inf = arith.constant 0xff800000 : f32
  call @expm1_f32(%neg_inf) : (f32) -> ()

  // CHECK: inf
  %inf = arith.constant 0x7f800000 : f32
  call @expm1_f32(%inf) : (f32) -> ()

  // CHECK: -1, inf, 1e-10
  %special_vec = arith.constant dense<[0xff800000, 0x7f800000, 1.0e-10]> : vector<3xf32>
  call @expm1_3xf32(%special_vec) : (vector<3xf32>) -> ()

  // CHECK: nan
  %nan = arith.constant 0x7fc00000 : f32
  call @expm1_f32(%nan) : (f32) -> ()

  return
}
// -------------------------------------------------------------------------- //
// Sin.
// -------------------------------------------------------------------------- //
func.func @sin_f32(%a : f32) {
  %r = math.sin %a : f32
  vector.print %r : f32
  return
}

func.func @sin_3xf32(%a : vector<3xf32>) {
  %r = math.sin %a : vector<3xf32>
  vector.print %r : vector<3xf32>
  return
}

func.func @sin() {
  // CHECK: 0
  %zero = arith.constant 0.0 : f32
  call @sin_f32(%zero) : (f32) -> ()

  // CHECK: 0.707107
  %pi_over_4 = arith.constant 0.78539816339 : f32
  call @sin_f32(%pi_over_4) : (f32) -> ()

  // CHECK: 1
  %pi_over_2 = arith.constant 1.57079632679 : f32
  call @sin_f32(%pi_over_2) : (f32) -> ()

  // CHECK: 0
  %pi = arith.constant 3.14159265359 : f32
  call @sin_f32(%pi) : (f32) -> ()

  // CHECK: -1
  %pi_3_over_2 = arith.constant 4.71238898038 : f32
  call @sin_f32(%pi_3_over_2) : (f32) -> ()

  // CHECK: 0, 0.866025, -1
  %vec_x = arith.constant dense<[9.42477796077, 2.09439510239, -1.57079632679]> : vector<3xf32>
  call @sin_3xf32(%vec_x) : (vector<3xf32>) -> ()

  return
}

// -------------------------------------------------------------------------- //
// cos.
// -------------------------------------------------------------------------- //
func.func @cos_f32(%a : f32) {
  %r = math.cos %a : f32
  vector.print %r : f32
  return
}

func.func @cos_3xf32(%a : vector<3xf32>) {
  %r = math.cos %a : vector<3xf32>
  vector.print %r : vector<3xf32>
  return
}

func.func @cos() {
  // CHECK: 1
  %zero = arith.constant 0.0 : f32
  call @cos_f32(%zero) : (f32) -> ()

  // CHECK: 0.707107
  %pi_over_4 = arith.constant 0.78539816339 : f32
  call @cos_f32(%pi_over_4) : (f32) -> ()

  // CHECK: 0
  %pi_over_2 = arith.constant 1.57079632679 : f32
  call @cos_f32(%pi_over_2) : (f32) -> ()

  // CHECK: -1
  %pi = arith.constant 3.14159265359 : f32
  call @cos_f32(%pi) : (f32) -> ()

  // CHECK: 0
  %pi_3_over_2 = arith.constant 4.71238898038 : f32
  call @cos_f32(%pi_3_over_2) : (f32) -> ()

  // CHECK: -1, -0.5, 0
  %vec_x = arith.constant dense<[9.42477796077, 2.09439510239, -1.57079632679]> : vector<3xf32>
  call @cos_3xf32(%vec_x) : (vector<3xf32>) -> ()

  return
}

// -------------------------------------------------------------------------- //
// Asin.
// -------------------------------------------------------------------------- //
func.func @asin_f32(%a : f32) {
  %r = math.asin %a : f32
  vector.print %r : f32
  return
}

func.func @asin_3xf32(%a : vector<3xf32>) {
  %r = math.asin %a : vector<3xf32>
  vector.print %r : vector<3xf32>
  return
}

func.func @asin() {
  // CHECK: 0
  %zero = arith.constant 0.0 : f32
  call @asin_f32(%zero) : (f32) -> ()

  // CHECK: -0.597406
  %cst1 = arith.constant -0.5625 : f32
  call @asin_f32(%cst1) : (f32) -> ()

  // CHECK: -0.384397
  %cst2 = arith.constant -0.375 : f32
  call @asin_f32(%cst2) : (f32) -> ()

  // CHECK: -0.25268
  %cst3 = arith.constant -0.25 : f32
  call @asin_f32(%cst3) : (f32) -> ()

  // CHECK: -1.1197
  %cst4 = arith.constant -0.90 : f32
  call @asin_f32(%cst4) : (f32) -> ()

  // CHECK: 0.25268, 0.384397, 0.597406
  %vec_x = arith.constant dense<[0.25, 0.375, 0.5625]> : vector<3xf32>
  call @asin_3xf32(%vec_x) : (vector<3xf32>) -> ()

  return
}

// -------------------------------------------------------------------------- //
// Acos.
// -------------------------------------------------------------------------- //
func.func @acos_f32(%a : f32) {
  %r = math.acos %a : f32
  vector.print %r : f32
  return
}

func.func @acos_3xf32(%a : vector<3xf32>) {
  %r = math.acos %a : vector<3xf32>
  vector.print %r : vector<3xf32>
  return
}

func.func @acos() {
  // CHECK: 1.5708
  %zero = arith.constant 0.0 : f32
  call @acos_f32(%zero) : (f32) -> ()

  // CHECK: 2.1682
  %cst1 = arith.constant -0.5625 : f32
  call @acos_f32(%cst1) : (f32) -> ()

  // CHECK: 1.95519
  %cst2 = arith.constant -0.375 : f32
  call @acos_f32(%cst2) : (f32) -> ()

  // CHECK: 1.82348
  %cst3 = arith.constant -0.25 : f32
  call @acos_f32(%cst3) : (f32) -> ()

  // CHECK: 1.31812, 1.1864, 0.97339
  %vec_x = arith.constant dense<[0.25, 0.375, 0.5625]> : vector<3xf32>
  call @acos_3xf32(%vec_x) : (vector<3xf32>) -> ()

  return
}

// -------------------------------------------------------------------------- //
// Atan.
// -------------------------------------------------------------------------- //
func.func @atan_f32(%a : f32) {
  %r = math.atan %a : f32
  vector.print %r : f32
  return
}

func.func @atan() {
  // CHECK: -0.785398
  %0 = arith.constant -1.0 : f32
  call @atan_f32(%0) : (f32) -> ()

  // CHECK: 0.785398
  %1 = arith.constant 1.0 : f32
  call @atan_f32(%1) : (f32) -> ()

  // CHECK: -0.463648
  %2 = arith.constant -0.5 : f32
  call @atan_f32(%2) : (f32) -> ()

  // CHECK: 0.463648
  %3 = arith.constant 0.5 : f32
  call @atan_f32(%3) : (f32) -> ()

  // CHECK: 0
  %4 = arith.constant 0.0 : f32
  call @atan_f32(%4) : (f32) -> ()

  // CHECK: -1.10715
  %5 = arith.constant -2.0 : f32
  call @atan_f32(%5) : (f32) -> ()

  // CHECK: 1.10715
  %6 = arith.constant 2.0 : f32
  call @atan_f32(%6) : (f32) -> ()

  return
}


// -------------------------------------------------------------------------- //
// Atan2.
// -------------------------------------------------------------------------- //
func.func @atan2_f32(%a : f32, %b : f32) {
  %r = math.atan2 %a, %b : f32
  vector.print %r : f32
  return
}

func.func @atan2() {
  %zero = arith.constant 0.0 : f32
  %one = arith.constant 1.0 : f32
  %two = arith.constant 2.0 : f32
  %neg_one = arith.constant -1.0 : f32
  %neg_two = arith.constant -2.0 : f32

  // CHECK: 0
  call @atan2_f32(%zero, %one) : (f32, f32) -> ()

  // CHECK: 1.5708
  call @atan2_f32(%one, %zero) : (f32, f32) -> ()

  // CHECK: 3.14159
  call @atan2_f32(%zero, %neg_one) : (f32, f32) -> ()

  // CHECK: -1.5708
  call @atan2_f32(%neg_one, %zero) : (f32, f32) -> ()

  // CHECK: nan
  call @atan2_f32(%zero, %zero) : (f32, f32) -> ()

  // CHECK: 1.10715
  call @atan2_f32(%two, %one) : (f32, f32) -> ()

  // CHECK: 2.03444
  %x6 = arith.constant -1.0 : f32
  %y6 = arith.constant 2.0 : f32
  call @atan2_f32(%two, %neg_one) : (f32, f32) -> ()

  // CHECK: -2.03444
  call @atan2_f32(%neg_two, %neg_one) : (f32, f32) -> ()

  // CHECK: -1.10715
  call @atan2_f32(%neg_two, %one) : (f32, f32) -> ()

  // CHECK: 0.463648
  call @atan2_f32(%one, %two) : (f32, f32) -> ()

  // CHECK: 2.67795
  %x10 = arith.constant -2.0 : f32
  %y10 = arith.constant 1.0 : f32
  call @atan2_f32(%one, %neg_two) : (f32, f32) -> ()

  // CHECK: -2.67795
  %x11 = arith.constant -2.0 : f32
  %y11 = arith.constant -1.0 : f32
  call @atan2_f32(%neg_one, %neg_two) : (f32, f32) -> ()

  // CHECK: -0.463648
  call @atan2_f32(%neg_one, %two) : (f32, f32) -> ()

  return
}


// -------------------------------------------------------------------------- //
// Cbrt.
// -------------------------------------------------------------------------- //

func.func @cbrt_f32(%a : f32) {
  %r = math.cbrt %a : f32
  vector.print %r : f32
  return
}

func.func @cbrt() {
  // CHECK: 1
  %a = arith.constant 1.0 : f32
  call @cbrt_f32(%a) : (f32) -> ()

  // CHECK: -1
  %b = arith.constant -1.0 : f32
  call @cbrt_f32(%b) : (f32) -> ()

  // CHECK: 0
  %c = arith.constant 0.0 : f32
  call @cbrt_f32(%c) : (f32) -> ()

  // CHECK: -0
  %d = arith.constant -0.0 : f32
  call @cbrt_f32(%d) : (f32) -> ()

  // CHECK: 10
  %e = arith.constant 1000.0 : f32
  call @cbrt_f32(%e) : (f32) -> ()

  // CHECK: -10
  %f = arith.constant -1000.0 : f32
  call @cbrt_f32(%f) : (f32) -> ()

  // CHECK: 2.57128
  %g = arith.constant 17.0 : f32
  call @cbrt_f32(%g) : (f32) -> ()

  return
}

// -------------------------------------------------------------------------- //
// floor.
// -------------------------------------------------------------------------- //
func.func @func_floorf32(%a : f32) {
  %r = math.floor %a : f32
  vector.print %r : f32
  return
}

func.func @floorf() {
  // CHECK: 3
  %a = arith.constant 3.8 : f32
  call @func_floorf32(%a) : (f32) -> ()

  // CHECK: -4
  %b = arith.constant -3.8 : f32
  call @func_floorf32(%b) : (f32) -> ()

  // CHECK: 0
  %c = arith.constant 0.0 : f32
  call @func_floorf32(%c) : (f32) -> ()

  // CHECK: -5
  %d = arith.constant -4.2 : f32
  call @func_floorf32(%d) : (f32) -> ()

  // CHECK: -2
  %e = arith.constant -2.0 : f32
  call @func_floorf32(%e) : (f32) -> ()

  // CHECK: 2
  %f = arith.constant 2.0 : f32
  call @func_floorf32(%f) : (f32) -> ()

  return
}

// -------------------------------------------------------------------------- //
// ceil.
// -------------------------------------------------------------------------- //
func.func @func_ceilf32(%a : f32) {
  %r = math.ceil %a : f32
  vector.print %r : f32
  return
}

func.func @ceilf() {
  // CHECK: 4
  %a = arith.constant 3.8 : f32
  call @func_ceilf32(%a) : (f32) -> ()

  // CHECK: -3
  %b = arith.constant -3.8 : f32
  call @func_ceilf32(%b) : (f32) -> ()

  // CHECK: 0
  %c = arith.constant 0.0 : f32
  call @func_ceilf32(%c) : (f32) -> ()

  // CHECK: -4
  %d = arith.constant -4.2 : f32
  call @func_ceilf32(%d) : (f32) -> ()

  // CHECK: -495
  %e = arith.constant -495.0 : f32
  call @func_ceilf32(%e) : (f32) -> ()

  // CHECK: 495
  %f = arith.constant 495.0 : f32
  call @func_ceilf32(%f) : (f32) -> ()

  return
}

func.func @main() {
  call @tanh(): () -> ()
  call @log(): () -> ()
  call @log2(): () -> ()
  call @log1p(): () -> ()
  call @erf(): () -> ()
  call @erfc(): () -> ()
  call @exp(): () -> ()
  call @expm1(): () -> ()
  call @sin(): () -> ()
  call @cos(): () -> ()
  call @asin(): () -> ()
  call @acos(): () -> ()
  call @atan() : () -> ()
  call @atan2() : () -> ()
  call @cbrt() : () -> ()
  call @floorf() : () -> ()
  call @ceilf() : () -> ()
  return
}
