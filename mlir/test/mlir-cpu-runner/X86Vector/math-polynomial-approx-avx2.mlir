// RUN:   mlir-opt %s -test-math-polynomial-approximation="enable-avx2"        \
// RUN:               -convert-arith-to-llvm                                   \
// RUN:               -convert-vector-to-llvm="enable-x86vector"               \
// RUN:               -convert-math-to-llvm                                    \
// RUN:               -convert-func-to-llvm                                     \
// RUN:               -reconcile-unrealized-casts                              \
// RUN: | mlir-cpu-runner                                                      \
// RUN:     -e main -entry-point-result=void -O0                               \
// RUN:     -shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext  \
// RUN:     -shared-libs=%mlir_lib_dir/libmlir_runner_utils%shlibext    \
// RUN: | FileCheck %s

// -------------------------------------------------------------------------- //
// rsqrt.
// -------------------------------------------------------------------------- //

func.func @rsqrt() {
  // Sanity-check that the scalar rsqrt still works OK.
  // CHECK: inf
  %0 = arith.constant 0.0 : f32
  %rsqrt_0 = math.rsqrt %0 : f32
  vector.print %rsqrt_0 : f32
  // CHECK: 0.707107
  %two = arith.constant 2.0: f32
  %rsqrt_two = math.rsqrt %two : f32
  vector.print %rsqrt_two : f32

  // Check that the vectorized approximation is reasonably accurate.
  // CHECK: 0.707107, 0.707107, 0.707107, 0.707107, 0.707107, 0.707107, 0.707107, 0.707107
  %vec8 = arith.constant dense<2.0> : vector<8xf32>
  %rsqrt_vec8 = math.rsqrt %vec8 : vector<8xf32>
  vector.print %rsqrt_vec8 : vector<8xf32>

  return
}

func.func @main() {
  call @rsqrt(): () -> ()
  return
}
