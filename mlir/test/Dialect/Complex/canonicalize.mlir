// RUN: mlir-opt %s -canonicalize="test-convergence" | FileCheck %s

// CHECK-LABEL: func @create_of_real_and_imag
// CHECK-SAME: (%[[CPLX:.*]]: complex<f32>)
func.func @create_of_real_and_imag(%cplx: complex<f32>) -> complex<f32> {
  // CHECK-NEXT: return %[[CPLX]] : complex<f32>
  %real = complex.re %cplx : complex<f32>
  %imag = complex.im %cplx : complex<f32>
  %complex = complex.create %real, %imag : complex<f32>
  return %complex : complex<f32>
}

// CHECK-LABEL: func @create_of_real_and_imag_different_operand
// CHECK-SAME: (%[[CPLX:.*]]: complex<f32>, %[[CPLX2:.*]]: complex<f32>)
func.func @create_of_real_and_imag_different_operand(
    %cplx: complex<f32>, %cplx2 : complex<f32>) -> complex<f32> {
  // CHECK-NEXT: %[[REAL:.*]] = complex.re %[[CPLX]] : complex<f32>
  // CHECK-NEXT: %[[IMAG:.*]] = complex.im %[[CPLX2]] : complex<f32>
  // CHECK-NEXT: %[[COMPLEX:.*]] = complex.create %[[REAL]], %[[IMAG]] : complex<f32>
  %real = complex.re %cplx : complex<f32>
  %imag = complex.im %cplx2 : complex<f32>
  %complex = complex.create %real, %imag : complex<f32>
  return %complex: complex<f32>
}

// CHECK-LABEL: func @real_of_const(
func.func @real_of_const() -> f32 {
  // CHECK: %[[CST:.*]] = arith.constant 1.000000e+00 : f32
  // CHECK-NEXT: return %[[CST]] : f32
  %complex = complex.constant [1.0 : f32, 0.0 : f32] : complex<f32>
  %1 = complex.re %complex : complex<f32>
  return %1 : f32
}

// CHECK-LABEL: func @real_of_create_op(
func.func @real_of_create_op() -> f32 {
  // CHECK: %[[CST:.*]] = arith.constant 1.000000e+00 : f32
  // CHECK-NEXT: return %[[CST]] : f32
  %real = arith.constant 1.0 : f32
  %imag = arith.constant 0.0 : f32
  %complex = complex.create %real, %imag : complex<f32>
  %1 = complex.re %complex : complex<f32>
  return %1 : f32
}

// CHECK-LABEL: func @imag_of_const(
func.func @imag_of_const() -> f32 {
  // CHECK: %[[CST:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK-NEXT: return %[[CST]] : f32
  %complex = complex.constant [1.0 : f32, 0.0 : f32] : complex<f32>
  %1 = complex.im %complex : complex<f32>
  return %1 : f32
}

// CHECK-LABEL: func @imag_of_create_op(
func.func @imag_of_create_op() -> f32 {
  // CHECK: %[[CST:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK-NEXT: return %[[CST]] : f32
  %real = arith.constant 1.0 : f32
  %imag = arith.constant 0.0 : f32
  %complex = complex.create %real, %imag : complex<f32>
  %1 = complex.im %complex : complex<f32>
  return %1 : f32
}

// CHECK-LABEL: func @complex_add_sub_lhs
func.func @complex_add_sub_lhs() -> complex<f32> {
  %complex1 = complex.constant [1.0 : f32, 0.0 : f32] : complex<f32>
  %complex2 = complex.constant [0.0 : f32, 2.0 : f32] : complex<f32>
  // CHECK: %[[CPLX:.*]] = complex.constant [1.000000e+00 : f32, 0.000000e+00 : f32] : complex<f32>
  // CHECK-NEXT: return %[[CPLX:.*]] : complex<f32>
  %sub = complex.sub %complex1, %complex2 : complex<f32>
  %add = complex.add %sub, %complex2 : complex<f32>
  return %add : complex<f32>
}

// CHECK-LABEL: func @complex_add_sub_rhs
func.func @complex_add_sub_rhs() -> complex<f32> {
  %complex1 = complex.constant [1.0 : f32, 0.0 : f32] : complex<f32>
  %complex2 = complex.constant [0.0 : f32, 2.0 : f32] : complex<f32>
  // CHECK: %[[CPLX:.*]] = complex.constant [1.000000e+00 : f32, 0.000000e+00 : f32] : complex<f32>
  // CHECK-NEXT: return %[[CPLX:.*]] : complex<f32>
  %sub = complex.sub %complex1, %complex2 : complex<f32>
  %add = complex.add %complex2, %sub : complex<f32>
  return %add : complex<f32>
}

// CHECK-LABEL: func @complex_neg_neg
func.func @complex_neg_neg() -> complex<f32> {
  %complex1 = complex.constant [1.0 : f32, 0.0 : f32] : complex<f32>
  // CHECK: %[[CPLX:.*]] = complex.constant [1.000000e+00 : f32, 0.000000e+00 : f32] : complex<f32>
  // CHECK-NEXT: return %[[CPLX:.*]] : complex<f32>
  %neg1 = complex.neg %complex1 : complex<f32>
  %neg2 = complex.neg %neg1 : complex<f32>
  return %neg2 : complex<f32>
}

// CHECK-LABEL: func @complex_log_exp
func.func @complex_log_exp() -> complex<f32> {
  %complex1 = complex.constant [1.0 : f32, 0.0 : f32] : complex<f32>
  // CHECK: %[[CPLX:.*]] = complex.constant [1.000000e+00 : f32, 0.000000e+00 : f32] : complex<f32>
  // CHECK-NEXT: return %[[CPLX:.*]] : complex<f32>
  %exp = complex.exp %complex1 : complex<f32>
  %log = complex.log %exp : complex<f32>
  return %log : complex<f32>
}

// CHECK-LABEL: func @complex_exp_log
func.func @complex_exp_log() -> complex<f32> {
  %complex1 = complex.constant [1.0 : f32, 0.0 : f32] : complex<f32>
  // CHECK: %[[CPLX:.*]] = complex.constant [1.000000e+00 : f32, 0.000000e+00 : f32] : complex<f32>
  // CHECK-NEXT: return %[[CPLX:.*]] : complex<f32>
  %log = complex.log %complex1 : complex<f32>
  %exp = complex.exp %log : complex<f32>
  return %exp : complex<f32>
}

// CHECK-LABEL: func @complex_conj_conj
func.func @complex_conj_conj() -> complex<f32> {
  %complex1 = complex.constant [1.0 : f32, 0.0 : f32] : complex<f32>
  // CHECK: %[[CPLX:.*]] = complex.constant [1.000000e+00 : f32, 0.000000e+00 : f32] : complex<f32>
  // CHECK-NEXT: return %[[CPLX:.*]] : complex<f32>
  %conj1 = complex.conj %complex1 : complex<f32>
  %conj2 = complex.conj %conj1 : complex<f32>
  return %conj2 : complex<f32>
}

// CHECK-LABEL: func @complex_add_zero
func.func @complex_add_zero() -> complex<f32> {
  %complex1 = complex.constant [1.0 : f32, 0.0 : f32] : complex<f32>
  %complex2 = complex.constant [0.0 : f32, 0.0 : f32] : complex<f32>
  // CHECK: %[[CPLX:.*]] = complex.constant [1.000000e+00 : f32, 0.000000e+00 : f32] : complex<f32>
  // CHECK-NEXT: return %[[CPLX:.*]] : complex<f32>
  %add = complex.add %complex1, %complex2 : complex<f32>
  return %add : complex<f32>
}

// CHECK-LABEL: func @complex_sub_add_lhs
func.func @complex_sub_add_lhs() -> complex<f32> {
  %complex1 = complex.constant [1.0 : f32, 0.0 : f32] : complex<f32>
  %complex2 = complex.constant [0.0 : f32, 2.0 : f32] : complex<f32>
  // CHECK: %[[CPLX:.*]] = complex.constant [1.000000e+00 : f32, 0.000000e+00 : f32] : complex<f32>
  // CHECK-NEXT: return %[[CPLX:.*]] : complex<f32>
  %add = complex.add %complex1, %complex2 : complex<f32>
  %sub = complex.sub %add, %complex2 : complex<f32>
  return %sub : complex<f32>
}

// CHECK-LABEL: func @complex_sub_zero
func.func @complex_sub_zero() -> complex<f32> {
  %complex1 = complex.constant [1.0 : f32, 0.0 : f32] : complex<f32>
  %complex2 = complex.constant [0.0 : f32, 0.0 : f32] : complex<f32>
  // CHECK: %[[CPLX:.*]] = complex.constant [1.000000e+00 : f32, 0.000000e+00 : f32] : complex<f32>
  // CHECK-NEXT: return %[[CPLX:.*]] : complex<f32>
  %sub = complex.sub %complex1, %complex2 : complex<f32>
  return %sub : complex<f32>
}

// CHECK-LABEL: func @re_neg
//  CHECK-SAME: (%[[ARG0:.*]]: f32, %[[ARG1:.*]]: f32)
func.func @re_neg(%arg0: f32, %arg1: f32) -> f32 {
  %create = complex.create %arg0, %arg1: complex<f32>
  // CHECK: %[[NEG:.*]] = arith.negf %[[ARG0]]
  %neg = complex.neg %create : complex<f32>
  %re = complex.re %neg : complex<f32>
  // CHECK-NEXT: return %[[NEG]]
  return %re : f32
}

// CHECK-LABEL: func @im_neg
//  CHECK-SAME: (%[[ARG0:.*]]: f32, %[[ARG1:.*]]: f32)
func.func @im_neg(%arg0: f32, %arg1: f32) -> f32 {
  %create = complex.create %arg0, %arg1: complex<f32>
  // CHECK: %[[NEG:.*]] = arith.negf %[[ARG1]]
  %neg = complex.neg %create : complex<f32>
  %im = complex.im %neg : complex<f32>
  // CHECK-NEXT: return %[[NEG]]
  return %im : f32
}

// CHECK-LABEL: func @mul_one_f16
//  CHECK-SAME: (%[[ARG0:.*]]: f16, %[[ARG1:.*]]: f16) -> complex<f16>
func.func @mul_one_f16(%arg0: f16, %arg1: f16) -> complex<f16> {
  %create = complex.create %arg0, %arg1: complex<f16>  
  %one = complex.constant [1.0 : f16, 0.0 : f16] : complex<f16>
  %mul = complex.mul %create, %one : complex<f16>
  // CHECK: %[[CREATE:.*]] = complex.create %[[ARG0]], %[[ARG1]] : complex<f16>
  // CHECK-NEXT: return %[[CREATE]]
  return %mul : complex<f16>
}

// CHECK-LABEL: func @mul_one_f32
//  CHECK-SAME: (%[[ARG0:.*]]: f32, %[[ARG1:.*]]: f32) -> complex<f32>
func.func @mul_one_f32(%arg0: f32, %arg1: f32) -> complex<f32> {
  %create = complex.create %arg0, %arg1: complex<f32>  
  %one = complex.constant [1.0 : f32, 0.0 : f32] : complex<f32>
  %mul = complex.mul %create, %one : complex<f32>
  // CHECK: %[[CREATE:.*]] = complex.create %[[ARG0]], %[[ARG1]] : complex<f32>
  // CHECK-NEXT: return %[[CREATE]]
  return %mul : complex<f32>
}

// CHECK-LABEL: func @mul_one_f64
//  CHECK-SAME: (%[[ARG0:.*]]: f64, %[[ARG1:.*]]: f64) -> complex<f64>
func.func @mul_one_f64(%arg0: f64, %arg1: f64) -> complex<f64> {
  %create = complex.create %arg0, %arg1: complex<f64>  
  %one = complex.constant [1.0 : f64, 0.0 : f64] : complex<f64>
  %mul = complex.mul %create, %one : complex<f64>
  // CHECK: %[[CREATE:.*]] = complex.create %[[ARG0]], %[[ARG1]] : complex<f64>
  // CHECK-NEXT: return %[[CREATE]]
  return %mul : complex<f64>
}

// CHECK-LABEL: func @mul_one_f80
//  CHECK-SAME: (%[[ARG0:.*]]: f80, %[[ARG1:.*]]: f80) -> complex<f80>
func.func @mul_one_f80(%arg0: f80, %arg1: f80) -> complex<f80> {
  %create = complex.create %arg0, %arg1: complex<f80>  
  %one = complex.constant [1.0 : f80, 0.0 : f80] : complex<f80>
  %mul = complex.mul %create, %one : complex<f80>
  // CHECK: %[[CREATE:.*]] = complex.create %[[ARG0]], %[[ARG1]] : complex<f80>
  // CHECK-NEXT: return %[[CREATE]]
  return %mul : complex<f80>
}

// CHECK-LABEL: func @mul_one_f128
//  CHECK-SAME: (%[[ARG0:.*]]: f128, %[[ARG1:.*]]: f128) -> complex<f128>
func.func @mul_one_f128(%arg0: f128, %arg1: f128) -> complex<f128> {
  %create = complex.create %arg0, %arg1: complex<f128>  
  %one = complex.constant [1.0 : f128, 0.0 : f128] : complex<f128>
  %mul = complex.mul %create, %one : complex<f128>
  // CHECK: %[[CREATE:.*]] = complex.create %[[ARG0]], %[[ARG1]] : complex<f128>
  // CHECK-NEXT: return %[[CREATE]]
  return %mul : complex<f128>
}

// CHECK-LABEL: func @fold_between_complex
//  CHECK-SAME: %[[ARG0:.*]]: complex<f32>
func.func @fold_between_complex(%arg0 : complex<f32>) -> complex<f32> {
  %0 = complex.bitcast %arg0 : complex<f32> to i64
  %1 = complex.bitcast %0 : i64 to complex<f32>
  // CHECK: return %[[ARG0]] : complex<f32>
  func.return %1 : complex<f32>
}

// CHECK-LABEL: func @fold_between_i64
//  CHECK-SAME: %[[ARG0:.*]]: i64
func.func @fold_between_i64(%arg0 : i64) -> i64 {
  %0 = complex.bitcast %arg0 : i64 to complex<f32>
  %1 = complex.bitcast %0 : complex<f32> to i64
  // CHECK: return %[[ARG0]] : i64
  func.return %1 : i64
}

// CHECK-LABEL: func @canon_arith_bitcast
//  CHECK-SAME: %[[ARG0:.*]]: f64
func.func @canon_arith_bitcast(%arg0 : f64) -> i64 {
  %0 = complex.bitcast %arg0 : f64 to complex<f32>
  %1 = complex.bitcast %0 : complex<f32> to i64
  // CHECK: %[[R0:.+]] = arith.bitcast %[[ARG0]]
  // CHECK: return %[[R0]] : i64
  func.return %1 : i64
}


// CHECK-LABEL: func @double_bitcast
//  CHECK-SAME: %[[ARG0:.*]]: f64
func.func @double_bitcast(%arg0 : f64) -> complex<f32> {
  // CHECK: %[[R0:.+]] = complex.bitcast %[[ARG0]]
  %0 = arith.bitcast %arg0 : f64 to i64
  %1 = complex.bitcast %0 : i64 to complex<f32>
  // CHECK: return %[[R0]] : complex<f32>
  func.return %1 : complex<f32>
}

// CHECK-LABEL: func @double_reverse_bitcast
//  CHECK-SAME: %[[ARG0:.*]]: complex<f32>
func.func @double_reverse_bitcast(%arg0 : complex<f32>) -> f64 {
  // CHECK: %[[R0:.+]] = complex.bitcast %[[ARG0]]
  %0 = complex.bitcast %arg0 : complex<f32> to i64
  %1 = arith.bitcast %0 : i64 to f64
  // CHECK: return %[[R0]] : f64
  func.return %1 : f64
}


// CHECK-LABEL: func @div_one_f16
//  CHECK-SAME: (%[[ARG0:.*]]: f16, %[[ARG1:.*]]: f16) -> complex<f16>
func.func @div_one_f16(%arg0: f16, %arg1: f16) -> complex<f16> {
  %create = complex.create %arg0, %arg1: complex<f16>  
  %one = complex.constant [1.0 : f16, 0.0 : f16] : complex<f16>
  %div = complex.div %create, %one : complex<f16>
  // CHECK: %[[CREATE:.*]] = complex.create %[[ARG0]], %[[ARG1]] : complex<f16>
  // CHECK-NEXT: return %[[CREATE]]
  return %div : complex<f16>
}

// CHECK-LABEL: func @div_one_f32
//  CHECK-SAME: (%[[ARG0:.*]]: f32, %[[ARG1:.*]]: f32) -> complex<f32>
func.func @div_one_f32(%arg0: f32, %arg1: f32) -> complex<f32> {
  %create = complex.create %arg0, %arg1: complex<f32>  
  %one = complex.constant [1.0 : f32, 0.0 : f32] : complex<f32>
  %div = complex.div %create, %one : complex<f32>
  // CHECK: %[[CREATE:.*]] = complex.create %[[ARG0]], %[[ARG1]] : complex<f32>
  // CHECK-NEXT: return %[[CREATE]]
  return %div : complex<f32>
}

// CHECK-LABEL: func @div_one_f64
//  CHECK-SAME: (%[[ARG0:.*]]: f64, %[[ARG1:.*]]: f64) -> complex<f64>
func.func @div_one_f64(%arg0: f64, %arg1: f64) -> complex<f64> {
  %create = complex.create %arg0, %arg1: complex<f64>  
  %one = complex.constant [1.0 : f64, 0.0 : f64] : complex<f64>
  %div = complex.div %create, %one : complex<f64>
  // CHECK: %[[CREATE:.*]] = complex.create %[[ARG0]], %[[ARG1]] : complex<f64>
  // CHECK-NEXT: return %[[CREATE]]
  return %div : complex<f64>
}

// CHECK-LABEL: func @div_one_f80
//  CHECK-SAME: (%[[ARG0:.*]]: f80, %[[ARG1:.*]]: f80) -> complex<f80>
func.func @div_one_f80(%arg0: f80, %arg1: f80) -> complex<f80> {
  %create = complex.create %arg0, %arg1: complex<f80>  
  %one = complex.constant [1.0 : f80, 0.0 : f80] : complex<f80>
  %div = complex.div %create, %one : complex<f80>
  // CHECK: %[[CREATE:.*]] = complex.create %[[ARG0]], %[[ARG1]] : complex<f80>
  // CHECK-NEXT: return %[[CREATE]]
  return %div : complex<f80>
}

// CHECK-LABEL: func @div_one_f128
//  CHECK-SAME: (%[[ARG0:.*]]: f128, %[[ARG1:.*]]: f128) -> complex<f128>
func.func @div_one_f128(%arg0: f128, %arg1: f128) -> complex<f128> {
  %create = complex.create %arg0, %arg1: complex<f128>  
  %one = complex.constant [1.0 : f128, 0.0 : f128] : complex<f128>
  %div = complex.div %create, %one : complex<f128>
  // CHECK: %[[CREATE:.*]] = complex.create %[[ARG0]], %[[ARG1]] : complex<f128>
  // CHECK-NEXT: return %[[CREATE]]
  return %div : complex<f128>
}
