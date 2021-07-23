// RUN: mlir-opt %s -convert-complex-to-standard | FileCheck %s

// CHECK-LABEL: func @complex_abs
// CHECK-SAME: %[[ARG:.*]]: complex<f32>
func @complex_abs(%arg: complex<f32>) -> f32 {
  %abs = complex.abs %arg: complex<f32>
  return %abs : f32
}
// CHECK: %[[REAL:.*]] = complex.re %[[ARG]] : complex<f32>
// CHECK: %[[IMAG:.*]] = complex.im %[[ARG]] : complex<f32>
// CHECK-DAG: %[[REAL_SQ:.*]] = mulf %[[REAL]], %[[REAL]] : f32
// CHECK-DAG: %[[IMAG_SQ:.*]] = mulf %[[IMAG]], %[[IMAG]] : f32
// CHECK: %[[SQ_NORM:.*]] = addf %[[REAL_SQ]], %[[IMAG_SQ]] : f32
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
// CHECK: %[[RESULT_REAL:.*]] = addf %[[REAL_LHS]], %[[REAL_RHS]] : f32
// CHECK: %[[IMAG_LHS:.*]] = complex.im %[[LHS]] : complex<f32>
// CHECK: %[[IMAG_RHS:.*]] = complex.im %[[RHS]] : complex<f32>
// CHECK: %[[RESULT_IMAG:.*]] = addf %[[IMAG_LHS]], %[[IMAG_RHS]] : f32
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

// CHECK: %[[RHS_REAL_IMAG_RATIO:.*]] = divf %[[RHS_REAL]], %[[RHS_IMAG]] : f32
// CHECK: %[[RHS_REAL_TIMES_RHS_REAL_IMAG_RATIO:.*]] = mulf %[[RHS_REAL_IMAG_RATIO]], %[[RHS_REAL]] : f32
// CHECK: %[[RHS_REAL_IMAG_DENOM:.*]] = addf %[[RHS_IMAG]], %[[RHS_REAL_TIMES_RHS_REAL_IMAG_RATIO]] : f32
// CHECK: %[[LHS_REAL_TIMES_RHS_REAL_IMAG_RATIO:.*]] = mulf %[[LHS_REAL]], %[[RHS_REAL_IMAG_RATIO]] : f32
// CHECK: %[[REAL_NUMERATOR_1:.*]] = addf %[[LHS_REAL_TIMES_RHS_REAL_IMAG_RATIO]], %[[LHS_IMAG]] : f32
// CHECK: %[[RESULT_REAL_1:.*]] = divf %[[REAL_NUMERATOR_1]], %[[RHS_REAL_IMAG_DENOM]] : f32
// CHECK: %[[LHS_IMAG_TIMES_RHS_REAL_IMAG_RATIO:.*]] = mulf %[[LHS_IMAG]], %[[RHS_REAL_IMAG_RATIO]] : f32
// CHECK: %[[IMAG_NUMERATOR_1:.*]] = subf %[[LHS_IMAG_TIMES_RHS_REAL_IMAG_RATIO]], %[[LHS_REAL]] : f32
// CHECK: %[[RESULT_IMAG_1:.*]] = divf %[[IMAG_NUMERATOR_1]], %[[RHS_REAL_IMAG_DENOM]] : f32

// CHECK: %[[RHS_IMAG_REAL_RATIO:.*]] = divf %[[RHS_IMAG]], %[[RHS_REAL]] : f32
// CHECK: %[[RHS_IMAG_TIMES_RHS_IMAG_REAL_RATIO:.*]] = mulf %[[RHS_IMAG_REAL_RATIO]], %[[RHS_IMAG]] : f32
// CHECK: %[[RHS_IMAG_REAL_DENOM:.*]] = addf %[[RHS_REAL]], %[[RHS_IMAG_TIMES_RHS_IMAG_REAL_RATIO]] : f32
// CHECK: %[[LHS_IMAG_TIMES_RHS_IMAG_REAL_RATIO:.*]] = mulf %[[LHS_IMAG]], %[[RHS_IMAG_REAL_RATIO]] : f32
// CHECK: %[[REAL_NUMERATOR_2:.*]] = addf %[[LHS_REAL]], %[[LHS_IMAG_TIMES_RHS_IMAG_REAL_RATIO]] : f32
// CHECK: %[[RESULT_REAL_2:.*]] = divf %[[REAL_NUMERATOR_2]], %[[RHS_IMAG_REAL_DENOM]] : f32
// CHECK: %[[LHS_REAL_TIMES_RHS_IMAG_REAL_RATIO:.*]] = mulf %[[LHS_REAL]], %[[RHS_IMAG_REAL_RATIO]] : f32
// CHECK: %[[IMAG_NUMERATOR_2:.*]] = subf %[[LHS_IMAG]], %[[LHS_REAL_TIMES_RHS_IMAG_REAL_RATIO]] : f32
// CHECK: %[[RESULT_IMAG_2:.*]] = divf %[[IMAG_NUMERATOR_2]], %[[RHS_IMAG_REAL_DENOM]] : f32

// Case 1. Zero denominator, numerator contains at most one NaN value.
// CHECK: %[[ZERO:.*]] = constant 0.000000e+00 : f32
// CHECK: %[[RHS_REAL_ABS:.*]] = absf %[[RHS_REAL]] : f32
// CHECK: %[[RHS_REAL_ABS_IS_ZERO:.*]] = cmpf oeq, %[[RHS_REAL_ABS]], %[[ZERO]] : f32
// CHECK: %[[RHS_IMAG_ABS:.*]] = absf %[[RHS_IMAG]] : f32
// CHECK: %[[RHS_IMAG_ABS_IS_ZERO:.*]] = cmpf oeq, %[[RHS_IMAG_ABS]], %[[ZERO]] : f32
// CHECK: %[[LHS_REAL_IS_NOT_NAN:.*]] = cmpf ord, %[[LHS_REAL]], %[[ZERO]] : f32
// CHECK: %[[LHS_IMAG_IS_NOT_NAN:.*]] = cmpf ord, %[[LHS_IMAG]], %[[ZERO]] : f32
// CHECK: %[[LHS_CONTAINS_NOT_NAN_VALUE:.*]] = or %[[LHS_REAL_IS_NOT_NAN]], %[[LHS_IMAG_IS_NOT_NAN]] : i1
// CHECK: %[[RHS_IS_ZERO:.*]] = and %[[RHS_REAL_ABS_IS_ZERO]], %[[RHS_IMAG_ABS_IS_ZERO]] : i1
// CHECK: %[[RESULT_IS_INFINITY:.*]] = and %[[LHS_CONTAINS_NOT_NAN_VALUE]], %[[RHS_IS_ZERO]] : i1
// CHECK: %[[INF:.*]] = constant 0x7F800000 : f32
// CHECK: %[[INF_WITH_SIGN_OF_RHS_REAL:.*]] = copysign %[[INF]], %[[RHS_REAL]] : f32
// CHECK: %[[INFINITY_RESULT_REAL:.*]] = mulf %[[INF_WITH_SIGN_OF_RHS_REAL]], %[[LHS_REAL]] : f32
// CHECK: %[[INFINITY_RESULT_IMAG:.*]] = mulf %[[INF_WITH_SIGN_OF_RHS_REAL]], %[[LHS_IMAG]] : f32

// Case 2. Infinite numerator, finite denominator.
// CHECK: %[[RHS_REAL_FINITE:.*]] = cmpf one, %[[RHS_REAL_ABS]], %[[INF]] : f32
// CHECK: %[[RHS_IMAG_FINITE:.*]] = cmpf one, %[[RHS_IMAG_ABS]], %[[INF]] : f32
// CHECK: %[[RHS_IS_FINITE:.*]] = and %[[RHS_REAL_FINITE]], %[[RHS_IMAG_FINITE]] : i1
// CHECK: %[[LHS_REAL_ABS:.*]] = absf %[[LHS_REAL]] : f32
// CHECK: %[[LHS_REAL_INFINITE:.*]] = cmpf oeq, %[[LHS_REAL_ABS]], %[[INF]] : f32
// CHECK: %[[LHS_IMAG_ABS:.*]] = absf %[[LHS_IMAG]] : f32
// CHECK: %[[LHS_IMAG_INFINITE:.*]] = cmpf oeq, %[[LHS_IMAG_ABS]], %[[INF]] : f32
// CHECK: %[[LHS_IS_INFINITE:.*]] = or %[[LHS_REAL_INFINITE]], %[[LHS_IMAG_INFINITE]] : i1
// CHECK: %[[INF_NUM_FINITE_DENOM:.*]] = and %[[LHS_IS_INFINITE]], %[[RHS_IS_FINITE]] : i1
// CHECK: %[[ONE:.*]] = constant 1.000000e+00 : f32
// CHECK: %[[LHS_REAL_IS_INF:.*]] = select %[[LHS_REAL_INFINITE]], %[[ONE]], %[[ZERO]] : f32
// CHECK: %[[LHS_REAL_IS_INF_WITH_SIGN:.*]] = copysign %[[LHS_REAL_IS_INF]], %[[LHS_REAL]] : f32
// CHECK: %[[LHS_IMAG_IS_INF:.*]] = select %[[LHS_IMAG_INFINITE]], %[[ONE]], %[[ZERO]] : f32
// CHECK: %[[LHS_IMAG_IS_INF_WITH_SIGN:.*]] = copysign %[[LHS_IMAG_IS_INF]], %[[LHS_IMAG]] : f32
// CHECK: %[[LHS_REAL_IS_INF_WITH_SIGN_TIMES_RHS_REAL:.*]] = mulf %[[LHS_REAL_IS_INF_WITH_SIGN]], %[[RHS_REAL]] : f32
// CHECK: %[[LHS_IMAG_IS_INF_WITH_SIGN_TIMES_RHS_IMAG:.*]] = mulf %[[LHS_IMAG_IS_INF_WITH_SIGN]], %[[RHS_IMAG]] : f32
// CHECK: %[[INF_MULTIPLICATOR_1:.*]] = addf %[[LHS_REAL_IS_INF_WITH_SIGN_TIMES_RHS_REAL]], %[[LHS_IMAG_IS_INF_WITH_SIGN_TIMES_RHS_IMAG]] : f32
// CHECK: %[[RESULT_REAL_3:.*]] = mulf %[[INF]], %[[INF_MULTIPLICATOR_1]] : f32
// CHECK: %[[LHS_REAL_IS_INF_WITH_SIGN_TIMES_RHS_IMAG:.*]] = mulf %[[LHS_REAL_IS_INF_WITH_SIGN]], %[[RHS_IMAG]] : f32
// CHECK: %[[LHS_IMAG_IS_INF_WITH_SIGN_TIMES_RHS_REAL:.*]] = mulf %[[LHS_IMAG_IS_INF_WITH_SIGN]], %[[RHS_REAL]] : f32
// CHECK: %[[INF_MULTIPLICATOR_2:.*]] = subf %[[LHS_IMAG_IS_INF_WITH_SIGN_TIMES_RHS_REAL]], %[[LHS_REAL_IS_INF_WITH_SIGN_TIMES_RHS_IMAG]] : f32
// CHECK: %[[RESULT_IMAG_3:.*]] = mulf %[[INF]], %[[INF_MULTIPLICATOR_2]] : f32

// Case 3. Finite numerator, infinite denominator.
// CHECK: %[[LHS_REAL_FINITE:.*]] = cmpf one, %[[LHS_REAL_ABS]], %[[INF]] : f32
// CHECK: %[[LHS_IMAG_FINITE:.*]] = cmpf one, %[[LHS_IMAG_ABS]], %[[INF]] : f32
// CHECK: %[[LHS_IS_FINITE:.*]] = and %[[LHS_REAL_FINITE]], %[[LHS_IMAG_FINITE]] : i1
// CHECK: %[[RHS_REAL_INFINITE:.*]] = cmpf oeq, %[[RHS_REAL_ABS]], %[[INF]] : f32
// CHECK: %[[RHS_IMAG_INFINITE:.*]] = cmpf oeq, %[[RHS_IMAG_ABS]], %[[INF]] : f32
// CHECK: %[[RHS_IS_INFINITE:.*]] = or %[[RHS_REAL_INFINITE]], %[[RHS_IMAG_INFINITE]] : i1
// CHECK: %[[FINITE_NUM_INFINITE_DENOM:.*]] = and %[[LHS_IS_FINITE]], %[[RHS_IS_INFINITE]] : i1
// CHECK: %[[RHS_REAL_IS_INF:.*]] = select %[[RHS_REAL_INFINITE]], %[[ONE]], %[[ZERO]] : f32
// CHECK: %[[RHS_REAL_IS_INF_WITH_SIGN:.*]] = copysign %[[RHS_REAL_IS_INF]], %[[RHS_REAL]] : f32
// CHECK: %[[RHS_IMAG_IS_INF:.*]] = select %[[RHS_IMAG_INFINITE]], %[[ONE]], %[[ZERO]] : f32
// CHECK: %[[RHS_IMAG_IS_INF_WITH_SIGN:.*]] = copysign %[[RHS_IMAG_IS_INF]], %[[RHS_IMAG]] : f32
// CHECK: %[[RHS_REAL_IS_INF_WITH_SIGN_TIMES_LHS_REAL:.*]] = mulf %[[LHS_REAL]], %[[RHS_REAL_IS_INF_WITH_SIGN]] : f32
// CHECK: %[[RHS_IMAG_IS_INF_WITH_SIGN_TIMES_LHS_IMAG:.*]] = mulf %[[LHS_IMAG]], %[[RHS_IMAG_IS_INF_WITH_SIGN]] : f32
// CHECK: %[[ZERO_MULTIPLICATOR_1:.*]] = addf %[[RHS_REAL_IS_INF_WITH_SIGN_TIMES_LHS_REAL]], %[[RHS_IMAG_IS_INF_WITH_SIGN_TIMES_LHS_IMAG]] : f32
// CHECK: %[[RESULT_REAL_4:.*]] = mulf %[[ZERO]], %[[ZERO_MULTIPLICATOR_1]] : f32
// CHECK: %[[RHS_REAL_IS_INF_WITH_SIGN_TIMES_LHS_IMAG:.*]] = mulf %[[LHS_IMAG]], %[[RHS_REAL_IS_INF_WITH_SIGN]] : f32
// CHECK: %[[RHS_IMAG_IS_INF_WITH_SIGN_TIMES_LHS_REAL:.*]] = mulf %[[LHS_REAL]], %[[RHS_IMAG_IS_INF_WITH_SIGN]] : f32
// CHECK: %[[ZERO_MULTIPLICATOR_2:.*]] = subf %[[RHS_REAL_IS_INF_WITH_SIGN_TIMES_LHS_IMAG]], %[[RHS_IMAG_IS_INF_WITH_SIGN_TIMES_LHS_REAL]] : f32
// CHECK: %[[RESULT_IMAG_4:.*]] = mulf %[[ZERO]], %[[ZERO_MULTIPLICATOR_2]] : f32

// CHECK: %[[REAL_ABS_SMALLER_THAN_IMAG_ABS:.*]] = cmpf olt, %[[RHS_REAL_ABS]], %[[RHS_IMAG_ABS]] : f32
// CHECK: %[[RESULT_REAL:.*]] = select %[[REAL_ABS_SMALLER_THAN_IMAG_ABS]], %[[RESULT_REAL_1]], %[[RESULT_REAL_2]] : f32
// CHECK: %[[RESULT_IMAG:.*]] = select %[[REAL_ABS_SMALLER_THAN_IMAG_ABS]], %[[RESULT_IMAG_1]], %[[RESULT_IMAG_2]] : f32
// CHECK: %[[RESULT_REAL_SPECIAL_CASE_3:.*]] = select %[[FINITE_NUM_INFINITE_DENOM]], %[[RESULT_REAL_4]], %[[RESULT_REAL]] : f32
// CHECK: %[[RESULT_IMAG_SPECIAL_CASE_3:.*]] = select %[[FINITE_NUM_INFINITE_DENOM]], %[[RESULT_IMAG_4]], %[[RESULT_IMAG]] : f32
// CHECK: %[[RESULT_REAL_SPECIAL_CASE_2:.*]] = select %[[INF_NUM_FINITE_DENOM]], %[[RESULT_REAL_3]], %[[RESULT_REAL_SPECIAL_CASE_3]] : f32
// CHECK: %[[RESULT_IMAG_SPECIAL_CASE_2:.*]] = select %[[INF_NUM_FINITE_DENOM]], %[[RESULT_IMAG_3]], %[[RESULT_IMAG_SPECIAL_CASE_3]] : f32
// CHECK: %[[RESULT_REAL_SPECIAL_CASE_1:.*]] = select %[[RESULT_IS_INFINITY]], %[[INFINITY_RESULT_REAL]], %[[RESULT_REAL_SPECIAL_CASE_2]] : f32
// CHECK: %[[RESULT_IMAG_SPECIAL_CASE_1:.*]] = select %[[RESULT_IS_INFINITY]], %[[INFINITY_RESULT_IMAG]], %[[RESULT_IMAG_SPECIAL_CASE_2]] : f32
// CHECK: %[[RESULT_REAL_IS_NAN:.*]] = cmpf uno, %[[RESULT_REAL]], %[[ZERO]] : f32
// CHECK: %[[RESULT_IMAG_IS_NAN:.*]] = cmpf uno, %[[RESULT_IMAG]], %[[ZERO]] : f32
// CHECK: %[[RESULT_IS_NAN:.*]] = and %[[RESULT_REAL_IS_NAN]], %[[RESULT_IMAG_IS_NAN]] : i1
// CHECK: %[[RESULT_REAL_WITH_SPECIAL_CASES:.*]] = select %[[RESULT_IS_NAN]], %[[RESULT_REAL_SPECIAL_CASE_1]], %[[RESULT_REAL]] : f32
// CHECK: %[[RESULT_IMAG_WITH_SPECIAL_CASES:.*]] = select %[[RESULT_IS_NAN]], %[[RESULT_IMAG_SPECIAL_CASE_1]], %[[RESULT_IMAG]] : f32
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
// CHECK-DAG: %[[REAL_EQUAL:.*]] = cmpf oeq, %[[REAL_LHS]], %[[REAL_RHS]] : f32
// CHECK-DAG: %[[IMAG_EQUAL:.*]] = cmpf oeq, %[[IMAG_LHS]], %[[IMAG_RHS]] : f32
// CHECK: %[[EQUAL:.*]] = and %[[REAL_EQUAL]], %[[IMAG_EQUAL]] : i1
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
// CHECK-DAG: %[[RESULT_REAL:.]] = mulf %[[EXP_REAL]], %[[COS_IMAG]] : f32
// CHECK-DAG: %[[SIN_IMAG:.*]] = math.sin %[[IMAG]] : f32
// CHECK-DAG: %[[RESULT_IMAG:.*]] = mulf %[[EXP_REAL]], %[[SIN_IMAG]] : f32
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
// CHECK: %[[SQR_REAL:.*]] = mulf %[[REAL]], %[[REAL]] : f32
// CHECK: %[[SQR_IMAG:.*]] = mulf %[[IMAG]], %[[IMAG]] : f32
// CHECK: %[[SQ_NORM:.*]] = addf %[[SQR_REAL]], %[[SQR_IMAG]] : f32
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
// CHECK: %[[ONE:.*]] = constant 1.000000e+00 : f32
// CHECK: %[[REAL_PLUS_ONE:.*]] = addf %[[REAL]], %[[ONE]] : f32
// CHECK: %[[NEW_COMPLEX:.*]] = complex.create %[[REAL_PLUS_ONE]], %[[IMAG]] : complex<f32>
// CHECK: %[[REAL:.*]] = complex.re %[[NEW_COMPLEX]] : complex<f32>
// CHECK: %[[IMAG:.*]] = complex.im %[[NEW_COMPLEX]] : complex<f32>
// CHECK: %[[SQR_REAL:.*]] = mulf %[[REAL]], %[[REAL]] : f32
// CHECK: %[[SQR_IMAG:.*]] = mulf %[[IMAG]], %[[IMAG]] : f32
// CHECK: %[[SQ_NORM:.*]] = addf %[[SQR_REAL]], %[[SQR_IMAG]] : f32
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
// CHECK: %[[LHS_REAL_ABS:.*]] = absf %[[LHS_REAL]] : f32
// CHECK: %[[LHS_IMAG:.*]] = complex.im %[[LHS]] : complex<f32>
// CHECK: %[[LHS_IMAG_ABS:.*]] = absf %[[LHS_IMAG]] : f32
// CHECK: %[[RHS_REAL:.*]] = complex.re %[[RHS]] : complex<f32>
// CHECK: %[[RHS_REAL_ABS:.*]] = absf %[[RHS_REAL]] : f32
// CHECK: %[[RHS_IMAG:.*]] = complex.im %[[RHS]] : complex<f32>
// CHECK: %[[RHS_IMAG_ABS:.*]] = absf %[[RHS_IMAG]] : f32

// CHECK: %[[LHS_REAL_TIMES_RHS_REAL:.*]] = mulf %[[LHS_REAL]], %[[RHS_REAL]] : f32
// CHECK: %[[LHS_REAL_TIMES_RHS_REAL_ABS:.*]] = absf %[[LHS_REAL_TIMES_RHS_REAL]] : f32
// CHECK: %[[LHS_IMAG_TIMES_RHS_IMAG:.*]] = mulf %[[LHS_IMAG]], %[[RHS_IMAG]] : f32
// CHECK: %[[LHS_IMAG_TIMES_RHS_IMAG_ABS:.*]] = absf %[[LHS_IMAG_TIMES_RHS_IMAG]] : f32
// CHECK: %[[REAL:.*]] = subf %[[LHS_REAL_TIMES_RHS_REAL]], %[[LHS_IMAG_TIMES_RHS_IMAG]] : f32

// CHECK: %[[LHS_IMAG_TIMES_RHS_REAL:.*]] = mulf %[[LHS_IMAG]], %[[RHS_REAL]] : f32
// CHECK: %[[LHS_IMAG_TIMES_RHS_REAL_ABS:.*]] = absf %[[LHS_IMAG_TIMES_RHS_REAL]] : f32
// CHECK: %[[LHS_REAL_TIMES_RHS_IMAG:.*]] = mulf %[[LHS_REAL]], %[[RHS_IMAG]] : f32
// CHECK: %[[LHS_REAL_TIMES_RHS_IMAG_ABS:.*]] = absf %[[LHS_REAL_TIMES_RHS_IMAG]] : f32
// CHECK: %[[IMAG:.*]] = addf %[[LHS_IMAG_TIMES_RHS_REAL]], %[[LHS_REAL_TIMES_RHS_IMAG]] : f32

// Handle cases where the "naive" calculation results in NaN values.
// CHECK: %[[REAL_IS_NAN:.*]] = cmpf uno, %[[REAL]], %[[REAL]] : f32
// CHECK: %[[IMAG_IS_NAN:.*]] = cmpf uno, %[[IMAG]], %[[IMAG]] : f32
// CHECK: %[[IS_NAN:.*]] = and %[[REAL_IS_NAN]], %[[IMAG_IS_NAN]] : i1
// CHECK: %[[INF:.*]] = constant 0x7F800000 : f32

// Case 1. LHS_REAL or LHS_IMAG are infinite.
// CHECK: %[[LHS_REAL_IS_INF:.*]] = cmpf oeq, %[[LHS_REAL_ABS]], %[[INF]] : f32
// CHECK: %[[LHS_IMAG_IS_INF:.*]] = cmpf oeq, %[[LHS_IMAG_ABS]], %[[INF]] : f32
// CHECK: %[[LHS_IS_INF:.*]] = or %[[LHS_REAL_IS_INF]], %[[LHS_IMAG_IS_INF]] : i1
// CHECK:  %[[RHS_REAL_IS_NAN:.*]] = cmpf uno, %[[RHS_REAL]], %[[RHS_REAL]] : f32
// CHECK: %[[RHS_IMAG_IS_NAN:.*]] = cmpf uno, %[[RHS_IMAG]], %[[RHS_IMAG]] : f32
// CHECK: %[[ZERO:.*]] = constant 0.000000e+00 : f32
// CHECK: %[[ONE:.*]] = constant 1.000000e+00 : f32
// CHECK: %[[LHS_REAL_IS_INF_FLOAT:.*]] = select %[[LHS_REAL_IS_INF]], %[[ONE]], %[[ZERO]] : f32
// CHECK: %[[TMP:.*]] = copysign %[[LHS_REAL_IS_INF_FLOAT]], %[[LHS_REAL]] : f32
// CHECK: %[[LHS_REAL1:.*]] = select %[[LHS_IS_INF]], %[[TMP]], %[[LHS_REAL]] : f32
// CHECK: %[[LHS_IMAG_IS_INF_FLOAT:.*]] = select %[[LHS_IMAG_IS_INF]], %[[ONE]], %[[ZERO]] : f32
// CHECK: %[[TMP:.*]] = copysign %[[LHS_IMAG_IS_INF_FLOAT]], %[[LHS_IMAG]] : f32
// CHECK: %[[LHS_IMAG1:.*]] = select %[[LHS_IS_INF]], %[[TMP]], %[[LHS_IMAG]] : f32
// CHECK: %[[LHS_IS_INF_AND_RHS_REAL_IS_NAN:.*]] = and %[[LHS_IS_INF]], %[[RHS_REAL_IS_NAN]] : i1
// CHECK: %[[TMP:.*]] = copysign %[[ZERO]], %[[RHS_REAL]] : f32
// CHECK: %[[RHS_REAL1:.*]] = select %[[LHS_IS_INF_AND_RHS_REAL_IS_NAN]], %[[TMP]], %[[RHS_REAL]] : f32
// CHECK: %[[LHS_IS_INF_AND_RHS_IMAG_IS_NAN:.*]] = and %[[LHS_IS_INF]], %[[RHS_IMAG_IS_NAN]] : i1
// CHECK: %[[TMP:.*]] = copysign %[[ZERO]], %[[RHS_IMAG]] : f32
// CHECK: %[[RHS_IMAG1:.*]] = select %[[LHS_IS_INF_AND_RHS_IMAG_IS_NAN]], %[[TMP]], %[[RHS_IMAG]] : f32

// Case 2. RHS_REAL or RHS_IMAG are infinite.
// CHECK: %[[RHS_REAL_IS_INF:.*]] = cmpf oeq, %[[RHS_REAL_ABS]], %[[INF]] : f32
// CHECK: %[[RHS_IMAG_IS_INF:.*]] = cmpf oeq, %[[RHS_IMAG_ABS]], %[[INF]] : f32
// CHECK: %[[RHS_IS_INF:.*]] = or %[[RHS_REAL_IS_INF]], %[[RHS_IMAG_IS_INF]] : i1
// CHECK: %[[LHS_REAL_IS_NAN:.*]] = cmpf uno, %[[LHS_REAL1]], %[[LHS_REAL1]] : f32
// CHECK: %[[LHS_IMAG_IS_NAN:.*]] = cmpf uno, %[[LHS_IMAG1]], %[[LHS_IMAG1]] : f32
// CHECK: %[[RHS_REAL_IS_INF_FLOAT:.*]] = select %[[RHS_REAL_IS_INF]], %[[ONE]], %[[ZERO]] : f32
// CHECK: %[[TMP:.*]] = copysign %[[RHS_REAL_IS_INF_FLOAT]], %[[RHS_REAL1]] : f32
// CHECK: %[[RHS_REAL2:.*]] = select %[[RHS_IS_INF]], %[[TMP]], %[[RHS_REAL1]] : f32
// CHECK: %[[RHS_IMAG_IS_INF_FLOAT:.*]] = select %[[RHS_IMAG_IS_INF]], %[[ONE]], %[[ZERO]] : f32
// CHECK: %[[TMP:.*]] = copysign %[[RHS_IMAG_IS_INF_FLOAT]], %[[RHS_IMAG1]] : f32
// CHECK: %[[RHS_IMAG2:.*]] = select %[[RHS_IS_INF]], %[[TMP]], %[[RHS_IMAG1]] : f32
// CHECK: %[[RHS_IS_INF_AND_LHS_REAL_IS_NAN:.*]] = and %[[RHS_IS_INF]], %[[LHS_REAL_IS_NAN]] : i1
// CHECK: %[[TMP:.*]] = copysign %[[ZERO]], %[[LHS_REAL1]] : f32
// CHECK: %[[LHS_REAL2:.*]] = select %[[RHS_IS_INF_AND_LHS_REAL_IS_NAN]], %[[TMP]], %[[LHS_REAL1]] : f32
// CHECK: %[[RHS_IS_INF_AND_LHS_IMAG_IS_NAN:.*]] = and %[[RHS_IS_INF]], %[[LHS_IMAG_IS_NAN]] : i1
// CHECK: %[[TMP:.*]] = copysign %[[ZERO]], %[[LHS_IMAG1]] : f32
// CHECK: %[[LHS_IMAG2:.*]] = select %[[RHS_IS_INF_AND_LHS_IMAG_IS_NAN]], %[[TMP]], %[[LHS_IMAG1]] : f32
// CHECK: %[[RECALC:.*]] = or %[[LHS_IS_INF]], %[[RHS_IS_INF]] : i1

// Case 3. One of the pairwise products of left hand side with right hand side
// is infinite.
// CHECK: %[[LHS_REAL_TIMES_RHS_REAL_IS_INF:.*]] = cmpf oeq, %[[LHS_REAL_TIMES_RHS_REAL_ABS]], %[[INF]] : f32
// CHECK: %[[LHS_IMAG_TIMES_RHS_IMAG_IS_INF:.*]] = cmpf oeq, %[[LHS_IMAG_TIMES_RHS_IMAG_ABS]], %[[INF]] : f32
// CHECK: %[[IS_SPECIAL_CASE:.*]] = or %[[LHS_REAL_TIMES_RHS_REAL_IS_INF]], %[[LHS_IMAG_TIMES_RHS_IMAG_IS_INF]] : i1
// CHECK: %[[LHS_REAL_TIMES_RHS_IMAG_IS_INF:.*]] = cmpf oeq, %[[LHS_REAL_TIMES_RHS_IMAG_ABS]], %[[INF]] : f32
// CHECK: %[[IS_SPECIAL_CASE1:.*]] = or %[[IS_SPECIAL_CASE]], %[[LHS_REAL_TIMES_RHS_IMAG_IS_INF]] : i1
// CHECK: %[[LHS_IMAG_TIMES_RHS_REAL_IS_INF:.*]] = cmpf oeq, %[[LHS_IMAG_TIMES_RHS_REAL_ABS]], %[[INF]] : f32
// CHECK: %[[IS_SPECIAL_CASE2:.*]] = or %[[IS_SPECIAL_CASE1]], %[[LHS_IMAG_TIMES_RHS_REAL_IS_INF]] : i1
// CHECK: %[[TRUE:.*]] = constant true
// CHECK: %[[NOT_RECALC:.*]] = xor %[[RECALC]], %[[TRUE]] : i1
// CHECK: %[[IS_SPECIAL_CASE3:.*]] = and %[[IS_SPECIAL_CASE2]], %[[NOT_RECALC]] : i1
// CHECK: %[[IS_SPECIAL_CASE_AND_LHS_REAL_IS_NAN:.*]] = and %[[IS_SPECIAL_CASE3]], %[[LHS_REAL_IS_NAN]] : i1
// CHECK: %[[TMP:.*]] = copysign %[[ZERO]], %[[LHS_REAL2]] : f32
// CHECK: %[[LHS_REAL3:.*]] = select %[[IS_SPECIAL_CASE_AND_LHS_REAL_IS_NAN]], %[[TMP]], %[[LHS_REAL2]] : f32
// CHECK: %[[IS_SPECIAL_CASE_AND_LHS_IMAG_IS_NAN:.*]] = and %[[IS_SPECIAL_CASE3]], %[[LHS_IMAG_IS_NAN]] : i1
// CHECK: %[[TMP:.*]] = copysign %[[ZERO]], %[[LHS_IMAG2]] : f32
// CHECK: %[[LHS_IMAG3:.*]] = select %[[IS_SPECIAL_CASE_AND_LHS_IMAG_IS_NAN]], %[[TMP]], %[[LHS_IMAG2]] : f32
// CHECK: %[[IS_SPECIAL_CASE_AND_RHS_REAL_IS_NAN:.*]] = and %[[IS_SPECIAL_CASE3]], %[[RHS_REAL_IS_NAN]] : i1
// CHECK: %[[TMP:.*]] = copysign %[[ZERO]], %[[RHS_REAL2]] : f32
// CHECK: %[[RHS_REAL3:.*]] = select %[[IS_SPECIAL_CASE_AND_RHS_REAL_IS_NAN]], %[[TMP]], %[[RHS_REAL2]] : f32
// CHECK: %[[IS_SPECIAL_CASE_AND_RHS_IMAG_IS_NAN:.*]] = and %[[IS_SPECIAL_CASE3]], %[[RHS_IMAG_IS_NAN]] : i1
// CHECK: %[[TMP:.*]] = copysign %[[ZERO]], %[[RHS_IMAG2]] : f32
// CHECK: %[[RHS_IMAG3:.*]] = select %[[IS_SPECIAL_CASE_AND_RHS_IMAG_IS_NAN]], %[[TMP]], %[[RHS_IMAG2]] : f32
// CHECK: %[[RECALC2:.*]] = or %[[RECALC]], %[[IS_SPECIAL_CASE3]] : i1
// CHECK: %[[RECALC3:.*]] = and %[[IS_NAN]], %[[RECALC2]] : i1

 // Recalculate real part.
// CHECK: %[[LHS_REAL_TIMES_RHS_REAL:.*]] = mulf %[[LHS_REAL3]], %[[RHS_REAL3]] : f32
// CHECK: %[[LHS_IMAG_TIMES_RHS_IMAG:.*]] = mulf %[[LHS_IMAG3]], %[[RHS_IMAG3]] : f32
// CHECK: %[[NEW_REAL:.*]] = subf %[[LHS_REAL_TIMES_RHS_REAL]], %[[LHS_IMAG_TIMES_RHS_IMAG]] : f32
// CHECK: %[[NEW_REAL_TIMES_INF:.*]] = mulf %[[INF]], %[[NEW_REAL]] : f32
// CHECK: %[[FINAL_REAL:.*]] = select %[[RECALC3]], %[[NEW_REAL_TIMES_INF]], %[[REAL]] : f32

// Recalculate imag part.
// CHECK: %[[LHS_IMAG_TIMES_RHS_REAL:.*]] = mulf %[[LHS_IMAG3]], %[[RHS_REAL3]] : f32
// CHECK: %[[LHS_REAL_TIMES_RHS_IMAG:.*]] = mulf %[[LHS_REAL3]], %[[RHS_IMAG3]] : f32
// CHECK: %[[NEW_IMAG:.*]] = addf %[[LHS_IMAG_TIMES_RHS_REAL]], %[[LHS_REAL_TIMES_RHS_IMAG]] : f32
// CHECK: %[[NEW_IMAG_TIMES_INF:.*]] = mulf %[[INF]], %[[NEW_IMAG]] : f32
// CHECK: %[[FINAL_IMAG:.*]] = select %[[RECALC3]], %[[NEW_IMAG_TIMES_INF]], %[[IMAG]] : f32

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
// CHECK-DAG: %[[NEG_REAL:.*]] = negf %[[REAL]] : f32
// CHECK-DAG: %[[NEG_IMAG:.*]] = negf %[[IMAG]] : f32
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
// CHECK-DAG: %[[REAL_NOT_EQUAL:.*]] = cmpf une, %[[REAL_LHS]], %[[REAL_RHS]] : f32
// CHECK-DAG: %[[IMAG_NOT_EQUAL:.*]] = cmpf une, %[[IMAG_LHS]], %[[IMAG_RHS]] : f32
// CHECK: %[[NOT_EQUAL:.*]] = or %[[REAL_NOT_EQUAL]], %[[IMAG_NOT_EQUAL]] : i1
// CHECK: return %[[NOT_EQUAL]] : i1

// CHECK-LABEL: func @complex_sign
// CHECK-SAME: %[[ARG:.*]]: complex<f32>
func @complex_sign(%arg: complex<f32>) -> complex<f32> {
  %sign = complex.sign %arg: complex<f32>
  return %sign : complex<f32>
}
// CHECK: %[[REAL:.*]] = complex.re %[[ARG]] : complex<f32>
// CHECK: %[[IMAG:.*]] = complex.im %[[ARG]] : complex<f32>
// CHECK: %[[ZERO:.*]] = constant 0.000000e+00 : f32
// CHECK: %[[REAL_IS_ZERO:.*]] = cmpf oeq, %[[REAL]], %[[ZERO]] : f32
// CHECK: %[[IMAG_IS_ZERO:.*]] = cmpf oeq, %1, %cst : f32
// CHECK: %[[IS_ZERO:.*]] = and %[[REAL_IS_ZERO]], %[[IMAG_IS_ZERO]] : i1
// CHECK: %[[REAL2:.*]] = complex.re %[[ARG]] : complex<f32>
// CHECK: %[[IMAG2:.*]] = complex.im %[[ARG]] : complex<f32>
// CHECK: %[[SQR_REAL:.*]] = mulf %[[REAL2]], %[[REAL2]] : f32
// CHECK: %[[SQR_IMAG:.*]] = mulf %[[IMAG2]], %[[IMAG2]] : f32
// CHECK: %[[SQ_NORM:.*]] = addf %[[SQR_REAL]], %[[SQR_IMAG]] : f32
// CHECK: %[[NORM:.*]] = math.sqrt %[[SQ_NORM]] : f32
// CHECK: %[[REAL_SIGN:.*]] = divf %[[REAL]], %[[NORM]] : f32
// CHECK: %[[IMAG_SIGN:.*]] = divf %[[IMAG]], %[[NORM]] : f32
// CHECK: %[[SIGN:.*]] = complex.create %[[REAL_SIGN]], %[[IMAG_SIGN]] : complex<f32>
// CHECK: %[[RESULT:.*]] = select %[[IS_ZERO]], %[[ARG]], %[[SIGN]] : complex<f32>
// CHECK: return %[[RESULT]] : complex<f32>

// CHECK-LABEL: func @complex_sub
// CHECK-SAME: (%[[LHS:.*]]: complex<f32>, %[[RHS:.*]]: complex<f32>)
func @complex_sub(%lhs: complex<f32>, %rhs: complex<f32>) -> complex<f32> {
  %sub = complex.sub %lhs, %rhs: complex<f32>
  return %sub : complex<f32>
}
// CHECK: %[[REAL_LHS:.*]] = complex.re %[[LHS]] : complex<f32>
// CHECK: %[[REAL_RHS:.*]] = complex.re %[[RHS]] : complex<f32>
// CHECK: %[[RESULT_REAL:.*]] = subf %[[REAL_LHS]], %[[REAL_RHS]] : f32
// CHECK: %[[IMAG_LHS:.*]] = complex.im %[[LHS]] : complex<f32>
// CHECK: %[[IMAG_RHS:.*]] = complex.im %[[RHS]] : complex<f32>
// CHECK: %[[RESULT_IMAG:.*]] = subf %[[IMAG_LHS]], %[[IMAG_RHS]] : f32
// CHECK: %[[RESULT:.*]] = complex.create %[[RESULT_REAL]], %[[RESULT_IMAG]] : complex<f32>
// CHECK: return %[[RESULT]] : complex<f32>
