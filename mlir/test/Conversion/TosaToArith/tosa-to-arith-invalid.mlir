// RUN: mlir-opt --split-input-file --tosa-to-arith="include-apply-rescale=true use-32-bit=true" %s -verify-diagnostics

// CHECK-LABEL: @apply_scale_unsupported_inexact_round
func.func @apply_scale_unsupported_inexact_round(%arg0 : i64, %arg1 : i32, %arg2 : i8) -> (i32) {
  // expected-error@+1 {{failed to legalize operation 'tosa.apply_scale'}}
  %res = tosa.apply_scale %arg0, %arg1, %arg2 {rounding_mode = INEXACT_ROUND} : (i64, i32, i8) -> i32
  return %res : i32
}
