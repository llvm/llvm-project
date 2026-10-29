// RUN:   mlir-opt %s -transform-interpreter                                   \
// RUN:               -test-transform-dialect-erase-schedule                   \
// RUN:               -convert-vector-to-scf                                   \
// RUN:               -convert-scf-to-cf                                       \
// RUN:               -convert-arith-to-llvm                                   \
// RUN:               -convert-cf-to-llvm                                   \
// RUN:               -convert-vector-to-llvm="enable-x86"               \
// RUN:               -convert-math-to-llvm                                    \
// RUN:               -convert-func-to-llvm                                    \
// RUN:               -reconcile-unrealized-casts                              \
// RUN: | mlir-runner                                                      \
// RUN:     -e main -entry-point-result=void -O0                               \
// RUN:     -shared-libs=%mlir_c_runner_utils  \
// RUN:     -shared-libs=%mlir_runner_utils    \
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

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.math.polynomial_approximation enable_avx2
    } : !transform.any_op
    transform.yield
  }
}
