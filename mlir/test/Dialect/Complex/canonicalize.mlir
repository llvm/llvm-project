// RUN: mlir-opt %s -canonicalize | FileCheck %s

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