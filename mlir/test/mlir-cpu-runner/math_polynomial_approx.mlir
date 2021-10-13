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

// XFAIL: s390x
// (see https://bugs.llvm.org/show_bug.cgi?id=51204)

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


func @main() {
  call @tanh(): () -> ()
  call @log(): () -> ()
  call @log2(): () -> ()
  call @log1p(): () -> ()
  call @exp(): () -> ()
  call @expm1(): () -> ()
  call @sin(): () -> ()
  call @cos(): () -> ()
  return
}
