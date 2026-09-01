// RUN: mlir-opt -split-input-file -verify-diagnostics %s

// Verify that test.with_bounds with mismatched attribute width (e.g., i64
// bounds for an i8 result) is rejected as invalid IR.
// See: https://github.com/llvm/llvm-project/issues/120882
func.func @with_bounds_mismatched_width() -> i8 {
  // expected-error@+1 {{'test.with_bounds' op bound attribute width (64) does not match result type width (8)}}
  %0 = test.with_bounds { umin = 10 : i64, umax = 15 : i64,
                           smin = 10 : i64, smax = 15 : i64 } : i8
  %1 = test.reflect_bounds %0 : i8
  return %1 : i8
}
