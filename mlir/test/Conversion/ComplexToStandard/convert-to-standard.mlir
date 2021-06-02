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
