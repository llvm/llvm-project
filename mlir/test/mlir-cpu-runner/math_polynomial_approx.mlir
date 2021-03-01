// RUN:   mlir-opt %s -test-math-polynomial-approximation                      \
// RUN:               -convert-vector-to-llvm                                  \
// RUN:               -convert-std-to-llvm                                     \
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
  %0 = constant 1.25 : f32
  %1 = math.tanh %0 : f32
  vector.print %1 : f32

  // CHECK: 0.244919, 0.635149, 0.761594, 0.848284
  %2 = constant dense<[0.25, 0.75, 1.0, 1.25]> : vector<4xf32>
  %3 = math.tanh %2 : vector<4xf32>
  vector.print %3 : vector<4xf32>

  // CHECK: 0.099668, 0.197375, 0.291313, 0.379949, 0.462117, 0.53705, 0.604368, 0.664037
  %4 = constant dense<[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]> : vector<8xf32>
  %5 = math.tanh %4 : vector<8xf32>
  vector.print %5 : vector<8xf32>

  return
}

// -------------------------------------------------------------------------- //
// Log.
// -------------------------------------------------------------------------- //
func @log() {
  // CHECK: 2.64704
  %0 = constant 14.112233 : f32
  %1 = math.log %0 : f32
  vector.print %1 : f32

  // CHECK: -1.38629, -0.287682, 0, 0.223144
  %2 = constant dense<[0.25, 0.75, 1.0, 1.25]> : vector<4xf32>
  %3 = math.log %2 : vector<4xf32>
  vector.print %3 : vector<4xf32>

  // CHECK: -2.30259, -1.60944, -1.20397, -0.916291, -0.693147, -0.510826, -0.356675, -0.223144
  %4 = constant dense<[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]> : vector<8xf32>
  %5 = math.log %4 : vector<8xf32>
  vector.print %5 : vector<8xf32>

  // CHECK: -inf
  %zero = constant 0.0 : f32
  %log_zero = math.log %zero : f32
  vector.print %log_zero : f32

  // CHECK: nan
  %neg_one = constant -1.0 : f32
  %log_neg_one = math.log %neg_one : f32
  vector.print %log_neg_one : f32

  // CHECK: inf
  %inf = constant 0x7f800000 : f32
  %log_inf = math.log %inf : f32
  vector.print %log_inf : f32

  // CHECK: -inf, nan, inf, 0.693147
  %special_vec = constant dense<[0.0, -1.0, 0x7f800000, 2.0]> : vector<4xf32>
  %log_special_vec = math.log %special_vec : vector<4xf32>
  vector.print %log_special_vec : vector<4xf32>

  return
}

func @main() {
  call @tanh(): () -> ()
  call @log(): () -> ()
  return
}
