// RUN: mlir-opt %s -pass-pipeline="builtin.func(convert-complex-to-standard)" | FileCheck %s

// CHECK-LABEL: func @complex_abs
// CHECK-SAME: %[[ARG:.*]]: complex<f32>
func @complex_abs(%arg: complex<f32>) -> f32 {
  %abs = complex.abs %arg: complex<f32>
  return %abs : f32
}
// CHECK: %[[REAL:.*]] = complex.re %[[ARG]] : complex<f32>
// CHECK: %[[IMAG:.*]] = complex.im %[[ARG]] : complex<f32>
// CHECK-DAG: %[[REAL_SQ:.*]] = arith.mulf %[[REAL]], %[[REAL]] : f32
// CHECK-DAG: %[[IMAG_SQ:.*]] = arith.mulf %[[IMAG]], %[[IMAG]] : f32
// CHECK: %[[SQ_NORM:.*]] = arith.addf %[[REAL_SQ]], %[[IMAG_SQ]] : f32
// CHECK: %[[NORM:.*]] = math.sqrt %[[SQ_NORM]] : f32
// CHECK: return %[[NORM]] : f32

// CHECK-LABEL: func @complex_add
// CHECK-SAME: (%[[LHS:.*]]: complex<f32>, %[[RHS:.*]]: complex<f32>)
func @complex_add(%lhs: complex<f32>, %rhs: complex<f32>) -> complex<f32> {
  %add = complex.add %lhs, %rhs: complex<f32>
  return %add : complex<f32>
}
// CHECK: %[[REAL_LHS:.*]] = complex.re %[[LHS]] : complex<f32>
// CHECK: %[[REAL_RHS:.*]] = complex.re %[[RHS]] : complex<f32>
// CHECK: %[[RESULT_REAL:.*]] = arith.addf %[[REAL_LHS]], %[[REAL_RHS]] : f32
// CHECK: %[[IMAG_LHS:.*]] = complex.im %[[LHS]] : complex<f32>
// CHECK: %[[IMAG_RHS:.*]] = complex.im %[[RHS]] : complex<f32>
// CHECK: %[[RESULT_IMAG:.*]] = arith.addf %[[IMAG_LHS]], %[[IMAG_RHS]] : f32
// CHECK: %[[RESULT:.*]] = complex.create %[[RESULT_REAL]], %[[RESULT_IMAG]] : complex<f32>
// CHECK: return %[[RESULT]] : complex<f32>

// CHECK-LABEL: func @complex_div
// CHECK-SAME: (%[[LHS:.*]]: complex<f32>, %[[RHS:.*]]: complex<f32>)
func @complex_div(%lhs: complex<f32>, %rhs: complex<f32>) -> complex<f32> {
  %div = complex.div %lhs, %rhs : complex<f32>
  return %div : complex<f32>
}
// CHECK: %[[LHS_REAL:.*]] = complex.re %[[LHS]] : complex<f32>
// CHECK: %[[LHS_IMAG:.*]] = complex.im %[[LHS]] : complex<f32>
// CHECK: %[[RHS_REAL:.*]] = complex.re %[[RHS]] : complex<f32>
// CHECK: %[[RHS_IMAG:.*]] = complex.im %[[RHS]] : complex<f32>

// CHECK: %[[RHS_REAL_IMAG_RATIO:.*]] = arith.divf %[[RHS_REAL]], %[[RHS_IMAG]] : f32
// CHECK: %[[RHS_REAL_TIMES_RHS_REAL_IMAG_RATIO:.*]] = arith.mulf %[[RHS_REAL_IMAG_RATIO]], %[[RHS_REAL]] : f32
// CHECK: %[[RHS_REAL_IMAG_DENOM:.*]] = arith.addf %[[RHS_IMAG]], %[[RHS_REAL_TIMES_RHS_REAL_IMAG_RATIO]] : f32
// CHECK: %[[LHS_REAL_TIMES_RHS_REAL_IMAG_RATIO:.*]] = arith.mulf %[[LHS_REAL]], %[[RHS_REAL_IMAG_RATIO]] : f32
// CHECK: %[[REAL_NUMERATOR_1:.*]] = arith.addf %[[LHS_REAL_TIMES_RHS_REAL_IMAG_RATIO]], %[[LHS_IMAG]] : f32
// CHECK: %[[RESULT_REAL_1:.*]] = arith.divf %[[REAL_NUMERATOR_1]], %[[RHS_REAL_IMAG_DENOM]] : f32
// CHECK: %[[LHS_IMAG_TIMES_RHS_REAL_IMAG_RATIO:.*]] = arith.mulf %[[LHS_IMAG]], %[[RHS_REAL_IMAG_RATIO]] : f32
// CHECK: %[[IMAG_NUMERATOR_1:.*]] = arith.subf %[[LHS_IMAG_TIMES_RHS_REAL_IMAG_RATIO]], %[[LHS_REAL]] : f32
// CHECK: %[[RESULT_IMAG_1:.*]] = arith.divf %[[IMAG_NUMERATOR_1]], %[[RHS_REAL_IMAG_DENOM]] : f32

// CHECK: %[[RHS_IMAG_REAL_RATIO:.*]] = arith.divf %[[RHS_IMAG]], %[[RHS_REAL]] : f32
// CHECK: %[[RHS_IMAG_TIMES_RHS_IMAG_REAL_RATIO:.*]] = arith.mulf %[[RHS_IMAG_REAL_RATIO]], %[[RHS_IMAG]] : f32
// CHECK: %[[RHS_IMAG_REAL_DENOM:.*]] = arith.addf %[[RHS_REAL]], %[[RHS_IMAG_TIMES_RHS_IMAG_REAL_RATIO]] : f32
// CHECK: %[[LHS_IMAG_TIMES_RHS_IMAG_REAL_RATIO:.*]] = arith.mulf %[[LHS_IMAG]], %[[RHS_IMAG_REAL_RATIO]] : f32
// CHECK: %[[REAL_NUMERATOR_2:.*]] = arith.addf %[[LHS_REAL]], %[[LHS_IMAG_TIMES_RHS_IMAG_REAL_RATIO]] : f32
// CHECK: %[[RESULT_REAL_2:.*]] = arith.divf %[[REAL_NUMERATOR_2]], %[[RHS_IMAG_REAL_DENOM]] : f32
// CHECK: %[[LHS_REAL_TIMES_RHS_IMAG_REAL_RATIO:.*]] = arith.mulf %[[LHS_REAL]], %[[RHS_IMAG_REAL_RATIO]] : f32
// CHECK: %[[IMAG_NUMERATOR_2:.*]] = arith.subf %[[LHS_IMAG]], %[[LHS_REAL_TIMES_RHS_IMAG_REAL_RATIO]] : f32
// CHECK: %[[RESULT_IMAG_2:.*]] = arith.divf %[[IMAG_NUMERATOR_2]], %[[RHS_IMAG_REAL_DENOM]] : f32

// Case 1. Zero denominator, numerator contains at most one NaN value.
// CHECK: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[RHS_REAL_ABS:.*]] = math.abs %[[RHS_REAL]] : f32
// CHECK: %[[RHS_REAL_ABS_IS_ZERO:.*]] = arith.cmpf oeq, %[[RHS_REAL_ABS]], %[[ZERO]] : f32
// CHECK: %[[RHS_IMAG_ABS:.*]] = math.abs %[[RHS_IMAG]] : f32
// CHECK: %[[RHS_IMAG_ABS_IS_ZERO:.*]] = arith.cmpf oeq, %[[RHS_IMAG_ABS]], %[[ZERO]] : f32
// CHECK: %[[LHS_REAL_IS_NOT_NAN:.*]] = arith.cmpf ord, %[[LHS_REAL]], %[[ZERO]] : f32
// CHECK: %[[LHS_IMAG_IS_NOT_NAN:.*]] = arith.cmpf ord, %[[LHS_IMAG]], %[[ZERO]] : f32
// CHECK: %[[LHS_CONTAINS_NOT_NAN_VALUE:.*]] = arith.ori %[[LHS_REAL_IS_NOT_NAN]], %[[LHS_IMAG_IS_NOT_NAN]] : i1
// CHECK: %[[RHS_IS_ZERO:.*]] = arith.andi %[[RHS_REAL_ABS_IS_ZERO]], %[[RHS_IMAG_ABS_IS_ZERO]] : i1
// CHECK: %[[RESULT_IS_INFINITY:.*]] = arith.andi %[[LHS_CONTAINS_NOT_NAN_VALUE]], %[[RHS_IS_ZERO]] : i1
// CHECK: %[[INF:.*]] = arith.constant 0x7F800000 : f32
// CHECK: %[[INF_WITH_SIGN_OF_RHS_REAL:.*]] = math.copysign %[[INF]], %[[RHS_REAL]] : f32
// CHECK: %[[INFINITY_RESULT_REAL:.*]] = arith.mulf %[[INF_WITH_SIGN_OF_RHS_REAL]], %[[LHS_REAL]] : f32
// CHECK: %[[INFINITY_RESULT_IMAG:.*]] = arith.mulf %[[INF_WITH_SIGN_OF_RHS_REAL]], %[[LHS_IMAG]] : f32

// Case 2. Infinite numerator, finite denominator.
// CHECK: %[[RHS_REAL_FINITE:.*]] = arith.cmpf one, %[[RHS_REAL_ABS]], %[[INF]] : f32
// CHECK: %[[RHS_IMAG_FINITE:.*]] = arith.cmpf one, %[[RHS_IMAG_ABS]], %[[INF]] : f32
// CHECK: %[[RHS_IS_FINITE:.*]] = arith.andi %[[RHS_REAL_FINITE]], %[[RHS_IMAG_FINITE]] : i1
// CHECK: %[[LHS_REAL_ABS:.*]] = math.abs %[[LHS_REAL]] : f32
// CHECK: %[[LHS_REAL_INFINITE:.*]] = arith.cmpf oeq, %[[LHS_REAL_ABS]], %[[INF]] : f32
// CHECK: %[[LHS_IMAG_ABS:.*]] = math.abs %[[LHS_IMAG]] : f32
// CHECK: %[[LHS_IMAG_INFINITE:.*]] = arith.cmpf oeq, %[[LHS_IMAG_ABS]], %[[INF]] : f32
// CHECK: %[[LHS_IS_INFINITE:.*]] = arith.ori %[[LHS_REAL_INFINITE]], %[[LHS_IMAG_INFINITE]] : i1
// CHECK: %[[INF_NUM_FINITE_DENOM:.*]] = arith.andi %[[LHS_IS_INFINITE]], %[[RHS_IS_FINITE]] : i1
// CHECK: %[[ONE:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[LHS_REAL_IS_INF:.*]] = arith.select %[[LHS_REAL_INFINITE]], %[[ONE]], %[[ZERO]] : f32
// CHECK: %[[LHS_REAL_IS_INF_WITH_SIGN:.*]] = math.copysign %[[LHS_REAL_IS_INF]], %[[LHS_REAL]] : f32
// CHECK: %[[LHS_IMAG_IS_INF:.*]] = arith.select %[[LHS_IMAG_INFINITE]], %[[ONE]], %[[ZERO]] : f32
// CHECK: %[[LHS_IMAG_IS_INF_WITH_SIGN:.*]] = math.copysign %[[LHS_IMAG_IS_INF]], %[[LHS_IMAG]] : f32
// CHECK: %[[LHS_REAL_IS_INF_WITH_SIGN_TIMES_RHS_REAL:.*]] = arith.mulf %[[LHS_REAL_IS_INF_WITH_SIGN]], %[[RHS_REAL]] : f32
// CHECK: %[[LHS_IMAG_IS_INF_WITH_SIGN_TIMES_RHS_IMAG:.*]] = arith.mulf %[[LHS_IMAG_IS_INF_WITH_SIGN]], %[[RHS_IMAG]] : f32
// CHECK: %[[INF_MULTIPLICATOR_1:.*]] = arith.addf %[[LHS_REAL_IS_INF_WITH_SIGN_TIMES_RHS_REAL]], %[[LHS_IMAG_IS_INF_WITH_SIGN_TIMES_RHS_IMAG]] : f32
// CHECK: %[[RESULT_REAL_3:.*]] = arith.mulf %[[INF]], %[[INF_MULTIPLICATOR_1]] : f32
// CHECK: %[[LHS_REAL_IS_INF_WITH_SIGN_TIMES_RHS_IMAG:.*]] = arith.mulf %[[LHS_REAL_IS_INF_WITH_SIGN]], %[[RHS_IMAG]] : f32
// CHECK: %[[LHS_IMAG_IS_INF_WITH_SIGN_TIMES_RHS_REAL:.*]] = arith.mulf %[[LHS_IMAG_IS_INF_WITH_SIGN]], %[[RHS_REAL]] : f32
// CHECK: %[[INF_MULTIPLICATOR_2:.*]] = arith.subf %[[LHS_IMAG_IS_INF_WITH_SIGN_TIMES_RHS_REAL]], %[[LHS_REAL_IS_INF_WITH_SIGN_TIMES_RHS_IMAG]] : f32
// CHECK: %[[RESULT_IMAG_3:.*]] = arith.mulf %[[INF]], %[[INF_MULTIPLICATOR_2]] : f32

// Case 3. Finite numerator, infinite denominator.
// CHECK: %[[LHS_REAL_FINITE:.*]] = arith.cmpf one, %[[LHS_REAL_ABS]], %[[INF]] : f32
// CHECK: %[[LHS_IMAG_FINITE:.*]] = arith.cmpf one, %[[LHS_IMAG_ABS]], %[[INF]] : f32
// CHECK: %[[LHS_IS_FINITE:.*]] = arith.andi %[[LHS_REAL_FINITE]], %[[LHS_IMAG_FINITE]] : i1
// CHECK: %[[RHS_REAL_INFINITE:.*]] = arith.cmpf oeq, %[[RHS_REAL_ABS]], %[[INF]] : f32
// CHECK: %[[RHS_IMAG_INFINITE:.*]] = arith.cmpf oeq, %[[RHS_IMAG_ABS]], %[[INF]] : f32
// CHECK: %[[RHS_IS_INFINITE:.*]] = arith.ori %[[RHS_REAL_INFINITE]], %[[RHS_IMAG_INFINITE]] : i1
// CHECK: %[[FINITE_NUM_INFINITE_DENOM:.*]] = arith.andi %[[LHS_IS_FINITE]], %[[RHS_IS_INFINITE]] : i1
// CHECK: %[[RHS_REAL_IS_INF:.*]] = arith.select %[[RHS_REAL_INFINITE]], %[[ONE]], %[[ZERO]] : f32
// CHECK: %[[RHS_REAL_IS_INF_WITH_SIGN:.*]] = math.copysign %[[RHS_REAL_IS_INF]], %[[RHS_REAL]] : f32
// CHECK: %[[RHS_IMAG_IS_INF:.*]] = arith.select %[[RHS_IMAG_INFINITE]], %[[ONE]], %[[ZERO]] : f32
// CHECK: %[[RHS_IMAG_IS_INF_WITH_SIGN:.*]] = math.copysign %[[RHS_IMAG_IS_INF]], %[[RHS_IMAG]] : f32
// CHECK: %[[RHS_REAL_IS_INF_WITH_SIGN_TIMES_LHS_REAL:.*]] = arith.mulf %[[LHS_REAL]], %[[RHS_REAL_IS_INF_WITH_SIGN]] : f32
// CHECK: %[[RHS_IMAG_IS_INF_WITH_SIGN_TIMES_LHS_IMAG:.*]] = arith.mulf %[[LHS_IMAG]], %[[RHS_IMAG_IS_INF_WITH_SIGN]] : f32
// CHECK: %[[ZERO_MULTIPLICATOR_1:.*]] = arith.addf %[[RHS_REAL_IS_INF_WITH_SIGN_TIMES_LHS_REAL]], %[[RHS_IMAG_IS_INF_WITH_SIGN_TIMES_LHS_IMAG]] : f32
// CHECK: %[[RESULT_REAL_4:.*]] = arith.mulf %[[ZERO]], %[[ZERO_MULTIPLICATOR_1]] : f32
// CHECK: %[[RHS_REAL_IS_INF_WITH_SIGN_TIMES_LHS_IMAG:.*]] = arith.mulf %[[LHS_IMAG]], %[[RHS_REAL_IS_INF_WITH_SIGN]] : f32
// CHECK: %[[RHS_IMAG_IS_INF_WITH_SIGN_TIMES_LHS_REAL:.*]] = arith.mulf %[[LHS_REAL]], %[[RHS_IMAG_IS_INF_WITH_SIGN]] : f32
// CHECK: %[[ZERO_MULTIPLICATOR_2:.*]] = arith.subf %[[RHS_REAL_IS_INF_WITH_SIGN_TIMES_LHS_IMAG]], %[[RHS_IMAG_IS_INF_WITH_SIGN_TIMES_LHS_REAL]] : f32
// CHECK: %[[RESULT_IMAG_4:.*]] = arith.mulf %[[ZERO]], %[[ZERO_MULTIPLICATOR_2]] : f32

// CHECK: %[[REAL_ABS_SMALLER_THAN_IMAG_ABS:.*]] = arith.cmpf olt, %[[RHS_REAL_ABS]], %[[RHS_IMAG_ABS]] : f32
// CHECK: %[[RESULT_REAL:.*]] = arith.select %[[REAL_ABS_SMALLER_THAN_IMAG_ABS]], %[[RESULT_REAL_1]], %[[RESULT_REAL_2]] : f32
// CHECK: %[[RESULT_IMAG:.*]] = arith.select %[[REAL_ABS_SMALLER_THAN_IMAG_ABS]], %[[RESULT_IMAG_1]], %[[RESULT_IMAG_2]] : f32
// CHECK: %[[RESULT_REAL_SPECIAL_CASE_3:.*]] = arith.select %[[FINITE_NUM_INFINITE_DENOM]], %[[RESULT_REAL_4]], %[[RESULT_REAL]] : f32
// CHECK: %[[RESULT_IMAG_SPECIAL_CASE_3:.*]] = arith.select %[[FINITE_NUM_INFINITE_DENOM]], %[[RESULT_IMAG_4]], %[[RESULT_IMAG]] : f32
// CHECK: %[[RESULT_REAL_SPECIAL_CASE_2:.*]] = arith.select %[[INF_NUM_FINITE_DENOM]], %[[RESULT_REAL_3]], %[[RESULT_REAL_SPECIAL_CASE_3]] : f32
// CHECK: %[[RESULT_IMAG_SPECIAL_CASE_2:.*]] = arith.select %[[INF_NUM_FINITE_DENOM]], %[[RESULT_IMAG_3]], %[[RESULT_IMAG_SPECIAL_CASE_3]] : f32
// CHECK: %[[RESULT_REAL_SPECIAL_CASE_1:.*]] = arith.select %[[RESULT_IS_INFINITY]], %[[INFINITY_RESULT_REAL]], %[[RESULT_REAL_SPECIAL_CASE_2]] : f32
// CHECK: %[[RESULT_IMAG_SPECIAL_CASE_1:.*]] = arith.select %[[RESULT_IS_INFINITY]], %[[INFINITY_RESULT_IMAG]], %[[RESULT_IMAG_SPECIAL_CASE_2]] : f32
// CHECK: %[[RESULT_REAL_IS_NAN:.*]] = arith.cmpf uno, %[[RESULT_REAL]], %[[ZERO]] : f32
// CHECK: %[[RESULT_IMAG_IS_NAN:.*]] = arith.cmpf uno, %[[RESULT_IMAG]], %[[ZERO]] : f32
// CHECK: %[[RESULT_IS_NAN:.*]] = arith.andi %[[RESULT_REAL_IS_NAN]], %[[RESULT_IMAG_IS_NAN]] : i1
// CHECK: %[[RESULT_REAL_WITH_SPECIAL_CASES:.*]] = arith.select %[[RESULT_IS_NAN]], %[[RESULT_REAL_SPECIAL_CASE_1]], %[[RESULT_REAL]] : f32
// CHECK: %[[RESULT_IMAG_WITH_SPECIAL_CASES:.*]] = arith.select %[[RESULT_IS_NAN]], %[[RESULT_IMAG_SPECIAL_CASE_1]], %[[RESULT_IMAG]] : f32
// CHECK: %[[RESULT:.*]] = complex.create %[[RESULT_REAL_WITH_SPECIAL_CASES]], %[[RESULT_IMAG_WITH_SPECIAL_CASES]] : complex<f32>
// CHECK: return %[[RESULT]] : complex<f32>

// CHECK-LABEL: func @complex_eq
// CHECK-SAME: %[[LHS:.*]]: complex<f32>, %[[RHS:.*]]: complex<f32>
func @complex_eq(%lhs: complex<f32>, %rhs: complex<f32>) -> i1 {
  %eq = complex.eq %lhs, %rhs: complex<f32>
  return %eq : i1
}
// CHECK: %[[REAL_LHS:.*]] = complex.re %[[LHS]] : complex<f32>
// CHECK: %[[IMAG_LHS:.*]] = complex.im %[[LHS]] : complex<f32>
// CHECK: %[[REAL_RHS:.*]] = complex.re %[[RHS]] : complex<f32>
// CHECK: %[[IMAG_RHS:.*]] = complex.im %[[RHS]] : complex<f32>
// CHECK-DAG: %[[REAL_EQUAL:.*]] = arith.cmpf oeq, %[[REAL_LHS]], %[[REAL_RHS]] : f32
// CHECK-DAG: %[[IMAG_EQUAL:.*]] = arith.cmpf oeq, %[[IMAG_LHS]], %[[IMAG_RHS]] : f32
// CHECK: %[[EQUAL:.*]] = arith.andi %[[REAL_EQUAL]], %[[IMAG_EQUAL]] : i1
// CHECK: return %[[EQUAL]] : i1

// CHECK-LABEL: func @complex_exp
// CHECK-SAME: %[[ARG:.*]]: complex<f32>
func @complex_exp(%arg: complex<f32>) -> complex<f32> {
  %exp = complex.exp %arg: complex<f32>
  return %exp : complex<f32>
}
// CHECK: %[[REAL:.*]] = complex.re %[[ARG]] : complex<f32>
// CHECK: %[[IMAG:.*]] = complex.im %[[ARG]] : complex<f32>
// CHECK-DAG: %[[COS_IMAG:.*]] = math.cos %[[IMAG]] : f32
// CHECK-DAG: %[[EXP_REAL:.*]] = math.exp %[[REAL]] : f32
// CHECK-DAG: %[[RESULT_REAL:.]] = arith.mulf %[[EXP_REAL]], %[[COS_IMAG]] : f32
// CHECK-DAG: %[[SIN_IMAG:.*]] = math.sin %[[IMAG]] : f32
// CHECK-DAG: %[[RESULT_IMAG:.*]] = arith.mulf %[[EXP_REAL]], %[[SIN_IMAG]] : f32
// CHECK: %[[RESULT:.*]] = complex.create %[[RESULT_REAL]], %[[RESULT_IMAG]] : complex<f32>
// CHECK: return %[[RESULT]] : complex<f32>

// CHECK-LABEL: func @complex_log
// CHECK-SAME: %[[ARG:.*]]: complex<f32>
func @complex_log(%arg: complex<f32>) -> complex<f32> {
  %log = complex.log %arg: complex<f32>
  return %log : complex<f32>
}
// CHECK: %[[REAL:.*]] = complex.re %[[ARG]] : complex<f32>
// CHECK: %[[IMAG:.*]] = complex.im %[[ARG]] : complex<f32>
// CHECK: %[[SQR_REAL:.*]] = arith.mulf %[[REAL]], %[[REAL]] : f32
// CHECK: %[[SQR_IMAG:.*]] = arith.mulf %[[IMAG]], %[[IMAG]] : f32
// CHECK: %[[SQ_NORM:.*]] = arith.addf %[[SQR_REAL]], %[[SQR_IMAG]] : f32
// CHECK: %[[NORM:.*]] = math.sqrt %[[SQ_NORM]] : f32
// CHECK: %[[RESULT_REAL:.*]] = math.log %[[NORM]] : f32
// CHECK: %[[REAL2:.*]] = complex.re %[[ARG]] : complex<f32>
// CHECK: %[[IMAG2:.*]] = complex.im %[[ARG]] : complex<f32>
// CHECK: %[[RESULT_IMAG:.*]] = math.atan2 %[[IMAG2]], %[[REAL2]] : f32
// CHECK: %[[RESULT:.*]] = complex.create %[[RESULT_REAL]], %[[RESULT_IMAG]] : complex<f32>
// CHECK: return %[[RESULT]] : complex<f32>

// CHECK-LABEL: func @complex_log1p
// CHECK-SAME: %[[ARG:.*]]: complex<f32>
func @complex_log1p(%arg: complex<f32>) -> complex<f32> {
  %log1p = complex.log1p %arg: complex<f32>
  return %log1p : complex<f32>
}
// CHECK: %[[REAL:.*]] = complex.re %[[ARG]] : complex<f32>
// CHECK: %[[IMAG:.*]] = complex.im %[[ARG]] : complex<f32>
// CHECK: %[[ONE:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[REAL_PLUS_ONE:.*]] = arith.addf %[[REAL]], %[[ONE]] : f32
// CHECK: %[[NEW_COMPLEX:.*]] = complex.create %[[REAL_PLUS_ONE]], %[[IMAG]] : complex<f32>
// CHECK: %[[REAL:.*]] = complex.re %[[NEW_COMPLEX]] : complex<f32>
// CHECK: %[[IMAG:.*]] = complex.im %[[NEW_COMPLEX]] : complex<f32>
// CHECK: %[[SQR_REAL:.*]] = arith.mulf %[[REAL]], %[[REAL]] : f32
// CHECK: %[[SQR_IMAG:.*]] = arith.mulf %[[IMAG]], %[[IMAG]] : f32
// CHECK: %[[SQ_NORM:.*]] = arith.addf %[[SQR_REAL]], %[[SQR_IMAG]] : f32
// CHECK: %[[NORM:.*]] = math.sqrt %[[SQ_NORM]] : f32
// CHECK: %[[RESULT_REAL:.*]] = math.log %[[NORM]] : f32
// CHECK: %[[REAL2:.*]] = complex.re %[[NEW_COMPLEX]] : complex<f32>
// CHECK: %[[IMAG2:.*]] = complex.im %[[NEW_COMPLEX]] : complex<f32>
// CHECK: %[[RESULT_IMAG:.*]] = math.atan2 %[[IMAG2]], %[[REAL2]] : f32
// CHECK: %[[RESULT:.*]] = complex.create %[[RESULT_REAL]], %[[RESULT_IMAG]] : complex<f32>
// CHECK: return %[[RESULT]] : complex<f32>

// CHECK-LABEL: func @complex_mul
// CHECK-SAME: (%[[LHS:.*]]: complex<f32>, %[[RHS:.*]]: complex<f32>)
func @complex_mul(%lhs: complex<f32>, %rhs: complex<f32>) -> complex<f32> {
  %mul = complex.mul %lhs, %rhs : complex<f32>
  return %mul : complex<f32>
}
// CHECK: %[[LHS_REAL:.*]] = complex.re %[[LHS]] : complex<f32>
// CHECK: %[[LHS_REAL_ABS:.*]] = math.abs %[[LHS_REAL]] : f32
// CHECK: %[[LHS_IMAG:.*]] = complex.im %[[LHS]] : complex<f32>
// CHECK: %[[LHS_IMAG_ABS:.*]] = math.abs %[[LHS_IMAG]] : f32
// CHECK: %[[RHS_REAL:.*]] = complex.re %[[RHS]] : complex<f32>
// CHECK: %[[RHS_REAL_ABS:.*]] = math.abs %[[RHS_REAL]] : f32
// CHECK: %[[RHS_IMAG:.*]] = complex.im %[[RHS]] : complex<f32>
// CHECK: %[[RHS_IMAG_ABS:.*]] = math.abs %[[RHS_IMAG]] : f32

// CHECK: %[[LHS_REAL_TIMES_RHS_REAL:.*]] = arith.mulf %[[LHS_REAL]], %[[RHS_REAL]] : f32
// CHECK: %[[LHS_REAL_TIMES_RHS_REAL_ABS:.*]] = math.abs %[[LHS_REAL_TIMES_RHS_REAL]] : f32
// CHECK: %[[LHS_IMAG_TIMES_RHS_IMAG:.*]] = arith.mulf %[[LHS_IMAG]], %[[RHS_IMAG]] : f32
// CHECK: %[[LHS_IMAG_TIMES_RHS_IMAG_ABS:.*]] = math.abs %[[LHS_IMAG_TIMES_RHS_IMAG]] : f32
// CHECK: %[[REAL:.*]] = arith.subf %[[LHS_REAL_TIMES_RHS_REAL]], %[[LHS_IMAG_TIMES_RHS_IMAG]] : f32

// CHECK: %[[LHS_IMAG_TIMES_RHS_REAL:.*]] = arith.mulf %[[LHS_IMAG]], %[[RHS_REAL]] : f32
// CHECK: %[[LHS_IMAG_TIMES_RHS_REAL_ABS:.*]] = math.abs %[[LHS_IMAG_TIMES_RHS_REAL]] : f32
// CHECK: %[[LHS_REAL_TIMES_RHS_IMAG:.*]] = arith.mulf %[[LHS_REAL]], %[[RHS_IMAG]] : f32
// CHECK: %[[LHS_REAL_TIMES_RHS_IMAG_ABS:.*]] = math.abs %[[LHS_REAL_TIMES_RHS_IMAG]] : f32
// CHECK: %[[IMAG:.*]] = arith.addf %[[LHS_IMAG_TIMES_RHS_REAL]], %[[LHS_REAL_TIMES_RHS_IMAG]] : f32

// Handle cases where the "naive" calculation results in NaN values.
// CHECK: %[[REAL_IS_NAN:.*]] = arith.cmpf uno, %[[REAL]], %[[REAL]] : f32
// CHECK: %[[IMAG_IS_NAN:.*]] = arith.cmpf uno, %[[IMAG]], %[[IMAG]] : f32
// CHECK: %[[IS_NAN:.*]] = arith.andi %[[REAL_IS_NAN]], %[[IMAG_IS_NAN]] : i1
// CHECK: %[[INF:.*]] = arith.constant 0x7F800000 : f32

// Case 1. LHS_REAL or LHS_IMAG are infinite.
// CHECK: %[[LHS_REAL_IS_INF:.*]] = arith.cmpf oeq, %[[LHS_REAL_ABS]], %[[INF]] : f32
// CHECK: %[[LHS_IMAG_IS_INF:.*]] = arith.cmpf oeq, %[[LHS_IMAG_ABS]], %[[INF]] : f32
// CHECK: %[[LHS_IS_INF:.*]] = arith.ori %[[LHS_REAL_IS_INF]], %[[LHS_IMAG_IS_INF]] : i1
// CHECK:  %[[RHS_REAL_IS_NAN:.*]] = arith.cmpf uno, %[[RHS_REAL]], %[[RHS_REAL]] : f32
// CHECK: %[[RHS_IMAG_IS_NAN:.*]] = arith.cmpf uno, %[[RHS_IMAG]], %[[RHS_IMAG]] : f32
// CHECK: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[ONE:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[LHS_REAL_IS_INF_FLOAT:.*]] = arith.select %[[LHS_REAL_IS_INF]], %[[ONE]], %[[ZERO]] : f32
// CHECK: %[[TMP:.*]] = math.copysign %[[LHS_REAL_IS_INF_FLOAT]], %[[LHS_REAL]] : f32
// CHECK: %[[LHS_REAL1:.*]] = arith.select %[[LHS_IS_INF]], %[[TMP]], %[[LHS_REAL]] : f32
// CHECK: %[[LHS_IMAG_IS_INF_FLOAT:.*]] = arith.select %[[LHS_IMAG_IS_INF]], %[[ONE]], %[[ZERO]] : f32
// CHECK: %[[TMP:.*]] = math.copysign %[[LHS_IMAG_IS_INF_FLOAT]], %[[LHS_IMAG]] : f32
// CHECK: %[[LHS_IMAG1:.*]] = arith.select %[[LHS_IS_INF]], %[[TMP]], %[[LHS_IMAG]] : f32
// CHECK: %[[LHS_IS_INF_AND_RHS_REAL_IS_NAN:.*]] = arith.andi %[[LHS_IS_INF]], %[[RHS_REAL_IS_NAN]] : i1
// CHECK: %[[TMP:.*]] = math.copysign %[[ZERO]], %[[RHS_REAL]] : f32
// CHECK: %[[RHS_REAL1:.*]] = arith.select %[[LHS_IS_INF_AND_RHS_REAL_IS_NAN]], %[[TMP]], %[[RHS_REAL]] : f32
// CHECK: %[[LHS_IS_INF_AND_RHS_IMAG_IS_NAN:.*]] = arith.andi %[[LHS_IS_INF]], %[[RHS_IMAG_IS_NAN]] : i1
// CHECK: %[[TMP:.*]] = math.copysign %[[ZERO]], %[[RHS_IMAG]] : f32
// CHECK: %[[RHS_IMAG1:.*]] = arith.select %[[LHS_IS_INF_AND_RHS_IMAG_IS_NAN]], %[[TMP]], %[[RHS_IMAG]] : f32

// Case 2. RHS_REAL or RHS_IMAG are infinite.
// CHECK: %[[RHS_REAL_IS_INF:.*]] = arith.cmpf oeq, %[[RHS_REAL_ABS]], %[[INF]] : f32
// CHECK: %[[RHS_IMAG_IS_INF:.*]] = arith.cmpf oeq, %[[RHS_IMAG_ABS]], %[[INF]] : f32
// CHECK: %[[RHS_IS_INF:.*]] = arith.ori %[[RHS_REAL_IS_INF]], %[[RHS_IMAG_IS_INF]] : i1
// CHECK: %[[LHS_REAL_IS_NAN:.*]] = arith.cmpf uno, %[[LHS_REAL1]], %[[LHS_REAL1]] : f32
// CHECK: %[[LHS_IMAG_IS_NAN:.*]] = arith.cmpf uno, %[[LHS_IMAG1]], %[[LHS_IMAG1]] : f32
// CHECK: %[[RHS_REAL_IS_INF_FLOAT:.*]] = arith.select %[[RHS_REAL_IS_INF]], %[[ONE]], %[[ZERO]] : f32
// CHECK: %[[TMP:.*]] = math.copysign %[[RHS_REAL_IS_INF_FLOAT]], %[[RHS_REAL1]] : f32
// CHECK: %[[RHS_REAL2:.*]] = arith.select %[[RHS_IS_INF]], %[[TMP]], %[[RHS_REAL1]] : f32
// CHECK: %[[RHS_IMAG_IS_INF_FLOAT:.*]] = arith.select %[[RHS_IMAG_IS_INF]], %[[ONE]], %[[ZERO]] : f32
// CHECK: %[[TMP:.*]] = math.copysign %[[RHS_IMAG_IS_INF_FLOAT]], %[[RHS_IMAG1]] : f32
// CHECK: %[[RHS_IMAG2:.*]] = arith.select %[[RHS_IS_INF]], %[[TMP]], %[[RHS_IMAG1]] : f32
// CHECK: %[[RHS_IS_INF_AND_LHS_REAL_IS_NAN:.*]] = arith.andi %[[RHS_IS_INF]], %[[LHS_REAL_IS_NAN]] : i1
// CHECK: %[[TMP:.*]] = math.copysign %[[ZERO]], %[[LHS_REAL1]] : f32
// CHECK: %[[LHS_REAL2:.*]] = arith.select %[[RHS_IS_INF_AND_LHS_REAL_IS_NAN]], %[[TMP]], %[[LHS_REAL1]] : f32
// CHECK: %[[RHS_IS_INF_AND_LHS_IMAG_IS_NAN:.*]] = arith.andi %[[RHS_IS_INF]], %[[LHS_IMAG_IS_NAN]] : i1
// CHECK: %[[TMP:.*]] = math.copysign %[[ZERO]], %[[LHS_IMAG1]] : f32
// CHECK: %[[LHS_IMAG2:.*]] = arith.select %[[RHS_IS_INF_AND_LHS_IMAG_IS_NAN]], %[[TMP]], %[[LHS_IMAG1]] : f32
// CHECK: %[[RECALC:.*]] = arith.ori %[[LHS_IS_INF]], %[[RHS_IS_INF]] : i1

// Case 3. One of the pairwise products of left hand side with right hand side
// is infinite.
// CHECK: %[[LHS_REAL_TIMES_RHS_REAL_IS_INF:.*]] = arith.cmpf oeq, %[[LHS_REAL_TIMES_RHS_REAL_ABS]], %[[INF]] : f32
// CHECK: %[[LHS_IMAG_TIMES_RHS_IMAG_IS_INF:.*]] = arith.cmpf oeq, %[[LHS_IMAG_TIMES_RHS_IMAG_ABS]], %[[INF]] : f32
// CHECK: %[[IS_SPECIAL_CASE:.*]] = arith.ori %[[LHS_REAL_TIMES_RHS_REAL_IS_INF]], %[[LHS_IMAG_TIMES_RHS_IMAG_IS_INF]] : i1
// CHECK: %[[LHS_REAL_TIMES_RHS_IMAG_IS_INF:.*]] = arith.cmpf oeq, %[[LHS_REAL_TIMES_RHS_IMAG_ABS]], %[[INF]] : f32
// CHECK: %[[IS_SPECIAL_CASE1:.*]] = arith.ori %[[IS_SPECIAL_CASE]], %[[LHS_REAL_TIMES_RHS_IMAG_IS_INF]] : i1
// CHECK: %[[LHS_IMAG_TIMES_RHS_REAL_IS_INF:.*]] = arith.cmpf oeq, %[[LHS_IMAG_TIMES_RHS_REAL_ABS]], %[[INF]] : f32
// CHECK: %[[IS_SPECIAL_CASE2:.*]] = arith.ori %[[IS_SPECIAL_CASE1]], %[[LHS_IMAG_TIMES_RHS_REAL_IS_INF]] : i1
// CHECK: %[[TRUE:.*]] = arith.constant true
// CHECK: %[[NOT_RECALC:.*]] = arith.xori %[[RECALC]], %[[TRUE]] : i1
// CHECK: %[[IS_SPECIAL_CASE3:.*]] = arith.andi %[[IS_SPECIAL_CASE2]], %[[NOT_RECALC]] : i1
// CHECK: %[[IS_SPECIAL_CASE_AND_LHS_REAL_IS_NAN:.*]] = arith.andi %[[IS_SPECIAL_CASE3]], %[[LHS_REAL_IS_NAN]] : i1
// CHECK: %[[TMP:.*]] = math.copysign %[[ZERO]], %[[LHS_REAL2]] : f32
// CHECK: %[[LHS_REAL3:.*]] = arith.select %[[IS_SPECIAL_CASE_AND_LHS_REAL_IS_NAN]], %[[TMP]], %[[LHS_REAL2]] : f32
// CHECK: %[[IS_SPECIAL_CASE_AND_LHS_IMAG_IS_NAN:.*]] = arith.andi %[[IS_SPECIAL_CASE3]], %[[LHS_IMAG_IS_NAN]] : i1
// CHECK: %[[TMP:.*]] = math.copysign %[[ZERO]], %[[LHS_IMAG2]] : f32
// CHECK: %[[LHS_IMAG3:.*]] = arith.select %[[IS_SPECIAL_CASE_AND_LHS_IMAG_IS_NAN]], %[[TMP]], %[[LHS_IMAG2]] : f32
// CHECK: %[[IS_SPECIAL_CASE_AND_RHS_REAL_IS_NAN:.*]] = arith.andi %[[IS_SPECIAL_CASE3]], %[[RHS_REAL_IS_NAN]] : i1
// CHECK: %[[TMP:.*]] = math.copysign %[[ZERO]], %[[RHS_REAL2]] : f32
// CHECK: %[[RHS_REAL3:.*]] = arith.select %[[IS_SPECIAL_CASE_AND_RHS_REAL_IS_NAN]], %[[TMP]], %[[RHS_REAL2]] : f32
// CHECK: %[[IS_SPECIAL_CASE_AND_RHS_IMAG_IS_NAN:.*]] = arith.andi %[[IS_SPECIAL_CASE3]], %[[RHS_IMAG_IS_NAN]] : i1
// CHECK: %[[TMP:.*]] = math.copysign %[[ZERO]], %[[RHS_IMAG2]] : f32
// CHECK: %[[RHS_IMAG3:.*]] = arith.select %[[IS_SPECIAL_CASE_AND_RHS_IMAG_IS_NAN]], %[[TMP]], %[[RHS_IMAG2]] : f32
// CHECK: %[[RECALC2:.*]] = arith.ori %[[RECALC]], %[[IS_SPECIAL_CASE3]] : i1
// CHECK: %[[RECALC3:.*]] = arith.andi %[[IS_NAN]], %[[RECALC2]] : i1

 // Recalculate real part.
// CHECK: %[[LHS_REAL_TIMES_RHS_REAL:.*]] = arith.mulf %[[LHS_REAL3]], %[[RHS_REAL3]] : f32
// CHECK: %[[LHS_IMAG_TIMES_RHS_IMAG:.*]] = arith.mulf %[[LHS_IMAG3]], %[[RHS_IMAG3]] : f32
// CHECK: %[[NEW_REAL:.*]] = arith.subf %[[LHS_REAL_TIMES_RHS_REAL]], %[[LHS_IMAG_TIMES_RHS_IMAG]] : f32
// CHECK: %[[NEW_REAL_TIMES_INF:.*]] = arith.mulf %[[INF]], %[[NEW_REAL]] : f32
// CHECK: %[[FINAL_REAL:.*]] = arith.select %[[RECALC3]], %[[NEW_REAL_TIMES_INF]], %[[REAL]] : f32

// Recalculate imag part.
// CHECK: %[[LHS_IMAG_TIMES_RHS_REAL:.*]] = arith.mulf %[[LHS_IMAG3]], %[[RHS_REAL3]] : f32
// CHECK: %[[LHS_REAL_TIMES_RHS_IMAG:.*]] = arith.mulf %[[LHS_REAL3]], %[[RHS_IMAG3]] : f32
// CHECK: %[[NEW_IMAG:.*]] = arith.addf %[[LHS_IMAG_TIMES_RHS_REAL]], %[[LHS_REAL_TIMES_RHS_IMAG]] : f32
// CHECK: %[[NEW_IMAG_TIMES_INF:.*]] = arith.mulf %[[INF]], %[[NEW_IMAG]] : f32
// CHECK: %[[FINAL_IMAG:.*]] = arith.select %[[RECALC3]], %[[NEW_IMAG_TIMES_INF]], %[[IMAG]] : f32

// CHECK: %[[RESULT:.*]] = complex.create %[[FINAL_REAL]], %[[FINAL_IMAG]] : complex<f32>
// CHECK: return %[[RESULT]] : complex<f32>

// CHECK-LABEL: func @complex_neg
// CHECK-SAME: %[[ARG:.*]]: complex<f32>
func @complex_neg(%arg: complex<f32>) -> complex<f32> {
  %neg = complex.neg %arg: complex<f32>
  return %neg : complex<f32>
}
// CHECK: %[[REAL:.*]] = complex.re %[[ARG]] : complex<f32>
// CHECK: %[[IMAG:.*]] = complex.im %[[ARG]] : complex<f32>
// CHECK-DAG: %[[NEG_REAL:.*]] = arith.negf %[[REAL]] : f32
// CHECK-DAG: %[[NEG_IMAG:.*]] = arith.negf %[[IMAG]] : f32
// CHECK: %[[RESULT:.*]] = complex.create %[[NEG_REAL]], %[[NEG_IMAG]] : complex<f32>
// CHECK: return %[[RESULT]] : complex<f32>

// CHECK-LABEL: func @complex_neq
// CHECK-SAME: %[[LHS:.*]]: complex<f32>, %[[RHS:.*]]: complex<f32>
func @complex_neq(%lhs: complex<f32>, %rhs: complex<f32>) -> i1 {
  %neq = complex.neq %lhs, %rhs: complex<f32>
  return %neq : i1
}
// CHECK: %[[REAL_LHS:.*]] = complex.re %[[LHS]] : complex<f32>
// CHECK: %[[IMAG_LHS:.*]] = complex.im %[[LHS]] : complex<f32>
// CHECK: %[[REAL_RHS:.*]] = complex.re %[[RHS]] : complex<f32>
// CHECK: %[[IMAG_RHS:.*]] = complex.im %[[RHS]] : complex<f32>
// CHECK-DAG: %[[REAL_NOT_EQUAL:.*]] = arith.cmpf une, %[[REAL_LHS]], %[[REAL_RHS]] : f32
// CHECK-DAG: %[[IMAG_NOT_EQUAL:.*]] = arith.cmpf une, %[[IMAG_LHS]], %[[IMAG_RHS]] : f32
// CHECK: %[[NOT_EQUAL:.*]] = arith.ori %[[REAL_NOT_EQUAL]], %[[IMAG_NOT_EQUAL]] : i1
// CHECK: return %[[NOT_EQUAL]] : i1

// CHECK-LABEL: func @complex_sign
// CHECK-SAME: %[[ARG:.*]]: complex<f32>
func @complex_sign(%arg: complex<f32>) -> complex<f32> {
  %sign = complex.sign %arg: complex<f32>
  return %sign : complex<f32>
}
// CHECK: %[[REAL:.*]] = complex.re %[[ARG]] : complex<f32>
// CHECK: %[[IMAG:.*]] = complex.im %[[ARG]] : complex<f32>
// CHECK: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[REAL_IS_ZERO:.*]] = arith.cmpf oeq, %[[REAL]], %[[ZERO]] : f32
// CHECK: %[[IMAG_IS_ZERO:.*]] = arith.cmpf oeq, %1, %cst : f32
// CHECK: %[[IS_ZERO:.*]] = arith.andi %[[REAL_IS_ZERO]], %[[IMAG_IS_ZERO]] : i1
// CHECK: %[[REAL2:.*]] = complex.re %[[ARG]] : complex<f32>
// CHECK: %[[IMAG2:.*]] = complex.im %[[ARG]] : complex<f32>
// CHECK: %[[SQR_REAL:.*]] = arith.mulf %[[REAL2]], %[[REAL2]] : f32
// CHECK: %[[SQR_IMAG:.*]] = arith.mulf %[[IMAG2]], %[[IMAG2]] : f32
// CHECK: %[[SQ_NORM:.*]] = arith.addf %[[SQR_REAL]], %[[SQR_IMAG]] : f32
// CHECK: %[[NORM:.*]] = math.sqrt %[[SQ_NORM]] : f32
// CHECK: %[[REAL_SIGN:.*]] = arith.divf %[[REAL]], %[[NORM]] : f32
// CHECK: %[[IMAG_SIGN:.*]] = arith.divf %[[IMAG]], %[[NORM]] : f32
// CHECK: %[[SIGN:.*]] = complex.create %[[REAL_SIGN]], %[[IMAG_SIGN]] : complex<f32>
// CHECK: %[[RESULT:.*]] = arith.select %[[IS_ZERO]], %[[ARG]], %[[SIGN]] : complex<f32>
// CHECK: return %[[RESULT]] : complex<f32>

// CHECK-LABEL: func @complex_sub
// CHECK-SAME: (%[[LHS:.*]]: complex<f32>, %[[RHS:.*]]: complex<f32>)
func @complex_sub(%lhs: complex<f32>, %rhs: complex<f32>) -> complex<f32> {
  %sub = complex.sub %lhs, %rhs: complex<f32>
  return %sub : complex<f32>
}
// CHECK: %[[REAL_LHS:.*]] = complex.re %[[LHS]] : complex<f32>
// CHECK: %[[REAL_RHS:.*]] = complex.re %[[RHS]] : complex<f32>
// CHECK: %[[RESULT_REAL:.*]] = arith.subf %[[REAL_LHS]], %[[REAL_RHS]] : f32
// CHECK: %[[IMAG_LHS:.*]] = complex.im %[[LHS]] : complex<f32>
// CHECK: %[[IMAG_RHS:.*]] = complex.im %[[RHS]] : complex<f32>
// CHECK: %[[RESULT_IMAG:.*]] = arith.subf %[[IMAG_LHS]], %[[IMAG_RHS]] : f32
// CHECK: %[[RESULT:.*]] = complex.create %[[RESULT_REAL]], %[[RESULT_IMAG]] : complex<f32>
// CHECK: return %[[RESULT]] : complex<f32>
