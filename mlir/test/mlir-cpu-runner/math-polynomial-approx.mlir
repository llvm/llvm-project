// RUN:   mlir-opt %s -test-math-polynomial-approximation                      \
// RUN:               -convert-arith-to-llvm                                   \
// RUN:               -convert-vector-to-llvm                                  \
// RUN:               -convert-math-to-llvm                                    \
// RUN:               -convert-std-to-llvm                                     \
// RUN:               -reconcile-unrealized-casts                              \
// RUN: | mlir-cpu-runner                                                      \
// RUN:     -e main -entry-point-result=void -O0                               \
// RUN:     -shared-libs=%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext  \
// RUN:     -shared-libs=%linalg_test_lib_dir/libmlir_runner_utils%shlibext    \
// RUN: | FileCheck %s

// -------------------------------------------------------------------------- //
// Tanh.
// -------------------------------------------------------------------------- //
func @tanh() {
  // CHECK: 0.848284
  %0 = arith.constant 1.25 : f32
  %1 = math.tanh %0 : f32
  vector.print %1 : f32

  // CHECK: 0.244919, 0.635149, 0.761594, 0.848284
  %2 = arith.constant dense<[0.25, 0.75, 1.0, 1.25]> : vector<4xf32>
  %3 = math.tanh %2 : vector<4xf32>
  vector.print %3 : vector<4xf32>

  // CHECK: 0.099668, 0.197375, 0.291313, 0.379949, 0.462117, 0.53705, 0.604368, 0.664037
  %4 = arith.constant dense<[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]> : vector<8xf32>
  %5 = math.tanh %4 : vector<8xf32>
  vector.print %5 : vector<8xf32>

  return
}

// -------------------------------------------------------------------------- //
// Log.
// -------------------------------------------------------------------------- //
func @log() {
  // CHECK: 2.64704
  %0 = arith.constant 14.112233 : f32
  %1 = math.log %0 : f32
  vector.print %1 : f32

  // CHECK: -1.38629, -0.287682, 0, 0.223144
  %2 = arith.constant dense<[0.25, 0.75, 1.0, 1.25]> : vector<4xf32>
  %3 = math.log %2 : vector<4xf32>
  vector.print %3 : vector<4xf32>

  // CHECK: -2.30259, -1.60944, -1.20397, -0.916291, -0.693147, -0.510826, -0.356675, -0.223144
  %4 = arith.constant dense<[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]> : vector<8xf32>
  %5 = math.log %4 : vector<8xf32>
  vector.print %5 : vector<8xf32>

  // CHECK: -inf
  %zero = arith.constant 0.0 : f32
  %log_zero = math.log %zero : f32
  vector.print %log_zero : f32

  // CHECK: nan
  %neg_one = arith.constant -1.0 : f32
  %log_neg_one = math.log %neg_one : f32
  vector.print %log_neg_one : f32

  // CHECK: inf
  %inf = arith.constant 0x7f800000 : f32
  %log_inf = math.log %inf : f32
  vector.print %log_inf : f32

  // CHECK: -inf, nan, inf, 0.693147
  %special_vec = arith.constant dense<[0.0, -1.0, 0x7f800000, 2.0]> : vector<4xf32>
  %log_special_vec = math.log %special_vec : vector<4xf32>
  vector.print %log_special_vec : vector<4xf32>

  return
}

func @log2() {
  // CHECK: 3.81887
  %0 = arith.constant 14.112233 : f32
  %1 = math.log2 %0 : f32
  vector.print %1 : f32

  // CHECK: -2, -0.415037, 0, 0.321928
  %2 = arith.constant dense<[0.25, 0.75, 1.0, 1.25]> : vector<4xf32>
  %3 = math.log2 %2 : vector<4xf32>
  vector.print %3 : vector<4xf32>

  // CHECK: -3.32193, -2.32193, -1.73697, -1.32193, -1, -0.736966, -0.514573, -0.321928
  %4 = arith.constant dense<[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]> : vector<8xf32>
  %5 = math.log2 %4 : vector<8xf32>
  vector.print %5 : vector<8xf32>

  // CHECK: -inf
  %zero = arith.constant 0.0 : f32
  %log_zero = math.log2 %zero : f32
  vector.print %log_zero : f32

  // CHECK: nan
  %neg_one = arith.constant -1.0 : f32
  %log_neg_one = math.log2 %neg_one : f32
  vector.print %log_neg_one : f32

  // CHECK: inf
  %inf = arith.constant 0x7f800000 : f32
  %log_inf = math.log2 %inf : f32
  vector.print %log_inf : f32

  // CHECK: -inf, nan, inf, 1.58496
  %special_vec = arith.constant dense<[0.0, -1.0, 0x7f800000, 3.0]> : vector<4xf32>
  %log_special_vec = math.log2 %special_vec : vector<4xf32>
  vector.print %log_special_vec : vector<4xf32>

  return
}

func @log1p() {
  // CHECK: 0.00995033
  %0 = arith.constant 0.01 : f32
  %1 = math.log1p %0 : f32
  vector.print %1 : f32

  // CHECK: -4.60517, -0.693147, 0, 1.38629
  %2 = arith.constant dense<[-0.99, -0.5, 0.0, 3.0]> : vector<4xf32>
  %3 = math.log1p %2 : vector<4xf32>
  vector.print %3 : vector<4xf32>

  // CHECK: 0.0953102, 0.182322, 0.262364, 0.336472, 0.405465, 0.470004, 0.530628, 0.587787
  %4 = arith.constant dense<[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]> : vector<8xf32>
  %5 = math.log1p %4 : vector<8xf32>
  vector.print %5 : vector<8xf32>

  // CHECK: -inf
  %neg_one = arith.constant -1.0 : f32
  %log_neg_one = math.log1p %neg_one : f32
  vector.print %log_neg_one : f32

  // CHECK: nan
  %neg_two = arith.constant -2.0 : f32
  %log_neg_two = math.log1p %neg_two : f32
  vector.print %log_neg_two : f32

  // CHECK: inf
  %inf = arith.constant 0x7f800000 : f32
  %log_inf = math.log1p %inf : f32
  vector.print %log_inf : f32

  // CHECK: -inf, nan, inf, 9.99995e-06
  %special_vec = arith.constant dense<[-1.0, -1.1, 0x7f800000, 0.00001]> : vector<4xf32>
  %log_special_vec = math.log1p %special_vec : vector<4xf32>
  vector.print %log_special_vec : vector<4xf32>

  return
}

// -------------------------------------------------------------------------- //
// Erf.
// -------------------------------------------------------------------------- //
func @erf() {
  // CHECK: -0.000274406
  %val1 = arith.constant -2.431864e-4 : f32
  %erfVal1 = math.erf %val1 : f32
  vector.print %erfVal1 : f32

  // CHECK: 0.742095
  %val2 = arith.constant 0.79999 : f32
  %erfVal2 = math.erf %val2 : f32
  vector.print %erfVal2 : f32

  // CHECK: 0.742101
  %val3 = arith.constant 0.8 : f32
  %erfVal3 = math.erf %val3 : f32
  vector.print %erfVal3 : f32

  // CHECK: 0.995322
  %val4 = arith.constant 1.99999 : f32
  %erfVal4 = math.erf %val4 : f32
  vector.print %erfVal4 : f32

  // CHECK: 0.995322
  %val5 = arith.constant 2.0 : f32
  %erfVal5 = math.erf %val5 : f32
  vector.print %erfVal5 : f32

  // CHECK: 1
  %val6 = arith.constant 3.74999 : f32
  %erfVal6 = math.erf %val6 : f32
  vector.print %erfVal6 : f32

  // CHECK: 1
  %val7 = arith.constant 3.75 : f32
  %erfVal7 = math.erf %val7 : f32
  vector.print %erfVal7 : f32

  // CHECK: -1
  %negativeInf = arith.constant 0xff800000 : f32
  %erfNegativeInf = math.erf %negativeInf : f32
  vector.print %erfNegativeInf : f32

  // CHECK: -1, -1, -0.913759, -0.731446
  %vecVals1 = arith.constant dense<[-3.4028235e+38, -4.54318, -1.2130899, -7.8234202e-01]> : vector<4xf32>
  %erfVecVals1 = math.erf %vecVals1 : vector<4xf32>
  vector.print %erfVecVals1 : vector<4xf32>

  // CHECK: -1.3264e-38, 0, 1.3264e-38, 0.121319
  %vecVals2 = arith.constant dense<[-1.1754944e-38, 0.0, 1.1754944e-38, 1.0793410e-01]> : vector<4xf32>
  %erfVecVals2 = math.erf %vecVals2 : vector<4xf32>
  vector.print %erfVecVals2 : vector<4xf32>

  // CHECK: 0.919477, 0.999069, 1, 1
  %vecVals3 = arith.constant dense<[1.23578, 2.34093, 3.82342, 3.4028235e+38]> : vector<4xf32>
  %erfVecVals3 = math.erf %vecVals3 : vector<4xf32>
  vector.print %erfVecVals3 : vector<4xf32>

  // CHECK: 1
  %inf = arith.constant 0x7f800000 : f32
  %erfInf = math.erf %inf : f32
  vector.print %erfInf : f32

  // CHECK: nan
  %nan = arith.constant 0x7fc00000 : f32
  %erfNan = math.erf %nan : f32
  vector.print %erfNan : f32

  return
}

// -------------------------------------------------------------------------- //
// Exp.
// -------------------------------------------------------------------------- //
func @exp() {
  // CHECK: 2.71828
  %0 = arith.constant 1.0 : f32
  %1 = math.exp %0 : f32
  vector.print %1 : f32

  // CHECK: 0.778802, 2.117, 2.71828, 3.85742
  %2 = arith.constant dense<[-0.25, 0.75, 1.0, 1.35]> : vector<4xf32>
  %3 = math.exp %2 : vector<4xf32>
  vector.print %3 : vector<4xf32>

  // CHECK: 1
  %zero = arith.constant 0.0 : f32
  %exp_zero = math.exp %zero : f32
  vector.print %exp_zero : f32

  // CHECK: 1.17549e-38, 1.38879e-11, 7.20049e+10, inf
  %special_vec = arith.constant dense<[-89.0, -25.0, 25.0, 89.0]> : vector<4xf32>
  %exp_special_vec = math.exp %special_vec : vector<4xf32>
  vector.print %exp_special_vec : vector<4xf32>

  // CHECK: inf
  %inf = arith.constant 0x7f800000 : f32
  %exp_inf = math.exp %inf : f32
  vector.print %exp_inf : f32

  // CHECK: 0
  %negative_inf = arith.constant 0xff800000 : f32
  %exp_negative_inf = math.exp %negative_inf : f32
  vector.print %exp_negative_inf : f32

  // CHECK: nan
  %nan = arith.constant 0x7fc00000 : f32
  %exp_nan = math.exp %nan : f32
  vector.print %exp_nan : f32

  return
}

func @expm1() {
  // CHECK: 1e-10
  %0 = arith.constant 1.0e-10 : f32
  %1 = math.expm1 %0 : f32
  vector.print %1 : f32

  // CHECK: -0.00995016, 0.0100502, 0.648721, 6.38905
  %2 = arith.constant dense<[-0.01, 0.01, 0.5, 2.0]> : vector<4xf32>
  %3 = math.expm1 %2 : vector<4xf32>
  vector.print %3 : vector<4xf32>

  // CHECK: -0.181269, 0, 0.221403, 0.491825, 0.822119, 1.22554, 1.71828, 2.32012
  %4 = arith.constant dense<[-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]> : vector<8xf32>
  %5 = math.expm1 %4 : vector<8xf32>
  vector.print %5 : vector<8xf32>

  // CHECK: -1
  %neg_inf = arith.constant 0xff800000 : f32
  %expm1_neg_inf = math.expm1 %neg_inf : f32
  vector.print %expm1_neg_inf : f32

  // CHECK: inf
  %inf = arith.constant 0x7f800000 : f32
  %expm1_inf = math.expm1 %inf : f32
  vector.print %expm1_inf : f32

  // CHECK: -1, inf, 1e-10
  %special_vec = arith.constant dense<[0xff800000, 0x7f800000, 1.0e-10]> : vector<3xf32>
  %log_special_vec = math.expm1 %special_vec : vector<3xf32>
  vector.print %log_special_vec : vector<3xf32>

  // CHECK: nan
  %nan = arith.constant 0x7fc00000 : f32
  %exp_nan = math.expm1 %nan : f32
  vector.print %exp_nan : f32

  return
}
// -------------------------------------------------------------------------- //
// Sin.
// -------------------------------------------------------------------------- //
func @sin() {
  // CHECK: 0
  %0 = arith.constant 0.0 : f32
  %sin_0 = math.sin %0 : f32
  vector.print %sin_0 : f32

  // CHECK: 0.707107
  %pi_over_4 = arith.constant 0.78539816339 : f32
  %sin_pi_over_4 = math.sin %pi_over_4 : f32
  vector.print %sin_pi_over_4 : f32

  // CHECK: 1
  %pi_over_2 = arith.constant 1.57079632679 : f32
  %sin_pi_over_2 = math.sin %pi_over_2 : f32
  vector.print %sin_pi_over_2 : f32


  // CHECK: 0
  %pi = arith.constant 3.14159265359 : f32
  %sin_pi = math.sin %pi : f32
  vector.print %sin_pi : f32

  // CHECK: -1
  %pi_3_over_2 = arith.constant 4.71238898038 : f32
  %sin_pi_3_over_2 = math.sin %pi_3_over_2 : f32
  vector.print %sin_pi_3_over_2 : f32

  // CHECK: 0, 0.866025, -1
  %vec_x = arith.constant dense<[9.42477796077, 2.09439510239, -1.57079632679]> : vector<3xf32>
  %sin_vec_x = math.sin %vec_x : vector<3xf32>
  vector.print %sin_vec_x : vector<3xf32>

  return
}

// -------------------------------------------------------------------------- //
// cos.
// -------------------------------------------------------------------------- //

func @cos() {
  // CHECK: 1
  %0 = arith.constant 0.0 : f32
  %cos_0 = math.cos %0 : f32
  vector.print %cos_0 : f32

  // CHECK: 0.707107
  %pi_over_4 = arith.constant 0.78539816339 : f32
  %cos_pi_over_4 = math.cos %pi_over_4 : f32
  vector.print %cos_pi_over_4 : f32

  //// CHECK: 0
  %pi_over_2 = arith.constant 1.57079632679 : f32
  %cos_pi_over_2 = math.cos %pi_over_2 : f32
  vector.print %cos_pi_over_2 : f32

  /// CHECK: -1
  %pi = arith.constant 3.14159265359 : f32
  %cos_pi = math.cos %pi : f32
  vector.print %cos_pi : f32

  // CHECK: 0
  %pi_3_over_2 = arith.constant 4.71238898038 : f32
  %cos_pi_3_over_2 = math.cos %pi_3_over_2 : f32
  vector.print %cos_pi_3_over_2 : f32

  // CHECK: -1, -0.5, 0
  %vec_x = arith.constant dense<[9.42477796077, 2.09439510239, -1.57079632679]> : vector<3xf32>
  %cos_vec_x = math.cos %vec_x : vector<3xf32>
  vector.print %cos_vec_x : vector<3xf32>


  return
}

// -------------------------------------------------------------------------- //
// Atan.
// -------------------------------------------------------------------------- //

func @atan() {
  // CHECK: -0.785184
  %0 = arith.constant -1.0 : f32
  %atan_0 = math.atan %0 : f32
  vector.print %atan_0 : f32

  // CHECK: 0.785184
  %1 = arith.constant 1.0 : f32
  %atan_1 = math.atan %1 : f32
  vector.print %atan_1 : f32

  // CHECK: -0.463643
  %2 = arith.constant -0.5 : f32
  %atan_2 = math.atan %2 : f32
  vector.print %atan_2 : f32

  // CHECK: 0.463643
  %3 = arith.constant 0.5 : f32
  %atan_3 = math.atan %3 : f32
  vector.print %atan_3 : f32

  // CHECK: 0
  %4 = arith.constant 0.0 : f32
  %atan_4 = math.atan %4 : f32
  vector.print %atan_4 : f32

  // CHECK: -1.10715
  %5 = arith.constant -2.0 : f32
  %atan_5 = math.atan %5 : f32
  vector.print %atan_5 : f32

  // CHECK: 1.10715
  %6 = arith.constant 2.0 : f32
  %atan_6 = math.atan %6 : f32
  vector.print %atan_6 : f32

  return
}


// -------------------------------------------------------------------------- //
// Atan2.
// -------------------------------------------------------------------------- //

func @atan2() {
  %zero = arith.constant 0.0 : f32
  %one = arith.constant 1.0 : f32
  %two = arith.constant 2.0 : f32
  %neg_one = arith.constant -1.0 : f32
  %neg_two = arith.constant -2.0 : f32

  // CHECK: 0
  %atan2_0 = math.atan2 %zero, %one : f32
  vector.print %atan2_0 : f32

  // CHECK: 1.5708
  %atan2_1 = math.atan2 %one, %zero : f32
  vector.print %atan2_1 : f32

  // CHECK: 3.14159
  %atan2_2 = math.atan2 %zero, %neg_one : f32
  vector.print %atan2_2 : f32

  // CHECK: -1.5708
  %atan2_3 = math.atan2 %neg_one, %zero : f32
  vector.print %atan2_3 : f32

  // CHECK: nan
  %atan2_4 = math.atan2 %zero, %zero : f32
  vector.print %atan2_4 : f32

  // CHECK: 1.10715
  %atan2_5 = math.atan2 %two, %one : f32
  vector.print %atan2_5 : f32

  // CHECK: 2.03444
  %x6 = arith.constant -1.0 : f32
  %y6 = arith.constant 2.0 : f32
  %atan2_6 = math.atan2 %two, %neg_one : f32
  vector.print %atan2_6 : f32

  // CHECK: -2.03444
  %atan2_7 = math.atan2 %neg_two, %neg_one : f32
  vector.print %atan2_7 : f32

  // CHECK: -1.10715
  %atan2_8 = math.atan2 %neg_two, %one : f32
  vector.print %atan2_8 : f32

  // CHECK: 0.463643
  %atan2_9 = math.atan2 %one, %two : f32
  vector.print %atan2_9 : f32

  // CHECK: 2.67795
  %x10 = arith.constant -2.0 : f32
  %y10 = arith.constant 1.0 : f32
  %atan2_10 = math.atan2 %one, %neg_two : f32
  vector.print %atan2_10 : f32

  // CHECK: -2.67795
  %x11 = arith.constant -2.0 : f32
  %y11 = arith.constant -1.0 : f32
  %atan2_11 = math.atan2 %neg_one, %neg_two : f32
  vector.print %atan2_11 : f32

  // CHECK: -0.463643
  %atan2_12 = math.atan2 %neg_one, %two : f32
  vector.print %atan2_12 : f32

  return
}


func @main() {
  call @tanh(): () -> ()
  call @log(): () -> ()
  call @log2(): () -> ()
  call @log1p(): () -> ()
  call @erf(): () -> ()
  call @exp(): () -> ()
  call @expm1(): () -> ()
  call @sin(): () -> ()
  call @cos(): () -> ()
  call @atan() : () -> ()
  call @atan2() : () -> ()
  return
}
