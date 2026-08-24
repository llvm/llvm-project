// RUN: mlir-opt -split-input-file -verify-diagnostics %s

// Verify that test.with_bounds with mismatched attribute width (e.g., i64
// bounds for an i8 result) is rejected as invalid IR.
// See: https://github.com/llvm/llvm-project/issues/120882
func.func @with_bounds_mismatched_width() -> i8 {
  // expected-error@+1 {{'test.with_bounds' op 'umin' bound attribute width (64) does not match result type width (8)}}
  %0 = test.with_bounds { umin = 10 : i64, umax = 15 : i64,
                           smin = 10 : i64, smax = 15 : i64 } : i8
  %1 = test.reflect_bounds %0 : i8
  return %1 : i8
}

// Verify that test.with_bounds with mismatched attribute width (e.g., i64
// bounds for an i8 result) is rejected as invalid IR. The old implementation
// only compared bitwidth of umin and result, so umax, smin, and smax must
// also be checked.
// See: https://github.com/llvm/llvm-project/issues/203855
func.func @mismatched_umax_bound_bitwidth() -> i32 {
  // expected-error@+1 {{'test.with_bounds' op 'umax' bound attribute width (8) does not match result type width (32)}}
  %0 = test.with_bounds {
    umin = 0 : i32,
    umax = 127 : i8,
    smin = 0 : i8,
    smax = 127 : i8
  } : i32
  return %0 : i32
}

func.func @mismatched_smin_bound_bitwidth() -> i32 {
  // expected-error@+1 {{'test.with_bounds' op 'smin' bound attribute width (8) does not match result type width (32)}}
  %0 = test.with_bounds {
    umin = 0 : i32,
    umax = 127 : i32,
    smin = 0 : i8,
    smax = 127 : i8
  } : i32
  return %0 : i32
}

func.func @mismatched_smax_bound_bitwidth() -> i32 {
  // expected-error@+1 {{'test.with_bounds' op 'smax' bound attribute width (8) does not match result type width (32)}}
  %0 = test.with_bounds {
    umin = 0 : i32,
    umax = 127 : i32,
    smin = 0 : i32,
    smax = 127 : i8
  } : i32
  return %0 : i32
}
