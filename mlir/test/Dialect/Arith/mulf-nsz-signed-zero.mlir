// RUN: mlir-opt %s -canonicalize --split-input-file | FileCheck %s

// Both operands constant, nnan+nsz: fold with APFloat (IEEE signed zeros),
// not the algebraic "X * 0 -> +0" identity.
// CHECK-LABEL: func.func @mulf_neg_const_times_pos_zero_nnan_nsz
// CHECK: %[[NZ:.*]] = arith.constant -0.000000e+00 : f32
// CHECK: return %[[NZ]]
func.func @mulf_neg_const_times_pos_zero_nnan_nsz() -> f32 {
  %neg = arith.constant -4.630000e+01 : f32
  %z = arith.constant 0.000000e+00 : f32
  %0 = arith.mulf %neg, %z fastmath<nnan,nsz> : f32
  return %0 : f32
}

// -----

// Same constants, no fastmath flags: APFloat multiply also yields -0.0.
// CHECK-LABEL: func.func @mulf_neg_const_times_pos_zero_ieee
// CHECK: %[[NZ:.*]] = arith.constant -0.000000e+00 : f32
// CHECK: return %[[NZ]]
func.func @mulf_neg_const_times_pos_zero_ieee() -> f32 {
  %neg = arith.constant -4.630000e+01 : f32
  %z = arith.constant 0.000000e+00 : f32
  %0 = arith.mulf %neg, %z : f32
  return %0 : f32
}

// -----

// Non-constant X still uses the nnan+nsz identity and folds to the zero.
// CHECK-LABEL: func.func @mulf_unknown_times_pos_zero_nnan_nsz
// CHECK: %[[Z:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: return %[[Z]]
func.func @mulf_unknown_times_pos_zero_nnan_nsz(%arg0: f32) -> f32 {
  %z = arith.constant 0.000000e+00 : f32
  %0 = arith.mulf %arg0, %z fastmath<nnan,nsz> : f32
  return %0 : f32
}
