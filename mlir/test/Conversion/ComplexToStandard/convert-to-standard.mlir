// RUN: mlir-opt %s --convert-complex-to-standard --split-input-file |\
// RUN: FileCheck %s --dump-input=always

// CHECK-LABEL: func @complex_abs
// CHECK-SAME: %[[ARG:.*]]: complex<f32>
func.func @complex_abs(%arg: complex<f32>) -> f32 {
  %abs = complex.abs %arg: complex<f32>
  return %abs : f32
}

// CHECK: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[ONE:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[REAL:.*]] = complex.re %[[ARG]] : complex<f32>
// CHECK: %[[IMAG:.*]] = complex.im %[[ARG]] : complex<f32>
// CHECK: %[[IS_REAL_ZERO:.*]] = arith.cmpf oeq, %[[REAL]], %[[ZERO]] : f32
// CHECK: %[[IS_IMAG_ZERO:.*]] = arith.cmpf oeq, %[[IMAG]], %[[ZERO]] : f32
// CHECK: %[[IMAG_DIV_REAL:.*]] = arith.divf %[[IMAG]], %[[REAL]] : f32
// CHECK: %[[IMAG_SQ:.*]] = arith.mulf %[[IMAG_DIV_REAL]], %[[IMAG_DIV_REAL]] : f32
// CHECK: %[[IMAG_SQ_PLUS_ONE:.*]] = arith.addf %[[IMAG_SQ]], %[[ONE]] : f32
// CHECK: %[[IMAG_SQRT:.*]] = math.sqrt %[[IMAG_SQ_PLUS_ONE]] : f32
// CHECK: %[[REAL_ABS:.*]] = math.absf %[[REAL]] : f32
// CHECK: %[[ABS_IMAG:.*]] = arith.mulf %[[IMAG_SQRT]], %[[REAL_ABS]] : f32
// CHECK: %[[REAL_DIV_IMAG:.*]] = arith.divf %[[REAL]], %[[IMAG]] : f32
// CHECK: %[[REAL_SQ:.*]] = arith.mulf %[[REAL_DIV_IMAG]], %[[REAL_DIV_IMAG]] : f32
// CHECK: %[[REAL_SQ_PLUS_ONE:.*]] = arith.addf %[[REAL_SQ]], %[[ONE]] : f32
// CHECK: %[[REAL_SQRT:.*]] = math.sqrt %[[REAL_SQ_PLUS_ONE]] : f32
// CHECK: %[[IMAG_ABS:.*]] = math.absf %[[IMAG]] : f32
// CHECK: %[[ABS_REAL:.*]] = arith.mulf %[[REAL_SQRT]], %[[IMAG_ABS]] : f32
// CHECK: %[[REAL_GT_IMAG:.*]] = arith.cmpf ogt, %[[REAL]], %[[IMAG]] : f32
// CHECK: %[[ABS1:.*]] = arith.select %[[REAL_GT_IMAG]], %[[ABS_IMAG]], %[[ABS_REAL]] : f32
// CHECK: %[[ABS2:.*]] = arith.select %[[IS_IMAG_ZERO]], %[[REAL_ABS]], %[[ABS1]] : f32
// CHECK: %[[ABS3:.*]] = arith.select %[[IS_REAL_ZERO]], %[[IMAG_ABS]], %[[ABS2]] : f32
// CHECK: return %[[ABS3]] : f32

// -----

// CHECK-LABEL: func @complex_atan2
func.func @complex_atan2(%lhs: complex<f32>,
                         %rhs: complex<f32>) -> complex<f32> {
  %atan2 = complex.atan2 %lhs, %rhs : complex<f32>
  return %atan2 : complex<f32>
}

// -----

// CHECK-LABEL: func @complex_add
// CHECK-SAME: (%[[LHS:.*]]: complex<f32>, %[[RHS:.*]]: complex<f32>)
func.func @complex_add(%lhs: complex<f32>, %rhs: complex<f32>) -> complex<f32> {
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

// -----

// CHECK-LABEL: func @complex_cos
// CHECK-SAME: %[[ARG:.*]]: complex<f32>
func.func @complex_cos(%arg: complex<f32>) -> complex<f32> {
  %cos = complex.cos %arg : complex<f32>
  return %cos : complex<f32>
}
// CHECK-DAG: %[[REAL:.*]] = complex.re %[[ARG]]
// CHECK-DAG: %[[IMAG:.*]] = complex.im %[[ARG]]
// CHECK-DAG: %[[HALF:.*]] = arith.constant 5.000000e-01 : f32
// CHECK-DAG: %[[EXP:.*]] = math.exp %[[IMAG]] : f32
// CHECK-DAG: %[[HALF_EXP:.*]] = arith.mulf %[[HALF]], %[[EXP]]
// CHECK-DAG: %[[HALF_REXP:.*]] = arith.divf %[[HALF]], %[[EXP]]
// CHECK-DAG: %[[SIN:.*]] = math.sin %[[REAL]] : f32
// CHECK-DAG: %[[COS:.*]] = math.cos %[[REAL]] : f32
// CHECK-DAG: %[[EXP_SUM:.*]] = arith.addf %[[HALF_REXP]], %[[HALF_EXP]]
// CHECK-DAG: %[[RESULT_REAL:.*]] = arith.mulf %[[EXP_SUM]], %[[COS]]
// CHECK-DAG: %[[EXP_DIFF:.*]] = arith.subf %[[HALF_REXP]], %[[HALF_EXP]]
// CHECK-DAG: %[[RESULT_IMAG:.*]] = arith.mulf %[[EXP_DIFF]], %[[SIN]]
// CHECK-DAG: %[[RESULT:.*]] = complex.create %[[RESULT_REAL]], %[[RESULT_IMAG]] : complex<f32>
// CHECK:     return %[[RESULT]]

// -----

// CHECK-LABEL: func @complex_div
// CHECK-SAME: (%[[LHS:.*]]: complex<f32>, %[[RHS:.*]]: complex<f32>)
func.func @complex_div(%lhs: complex<f32>, %rhs: complex<f32>) -> complex<f32> {
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
// CHECK: %[[RHS_REAL_ABS:.*]] = math.absf %[[RHS_REAL]] : f32
// CHECK: %[[RHS_REAL_ABS_IS_ZERO:.*]] = arith.cmpf oeq, %[[RHS_REAL_ABS]], %[[ZERO]] : f32
// CHECK: %[[RHS_IMAG_ABS:.*]] = math.absf %[[RHS_IMAG]] : f32
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
// CHECK: %[[LHS_REAL_ABS:.*]] = math.absf %[[LHS_REAL]] : f32
// CHECK: %[[LHS_REAL_INFINITE:.*]] = arith.cmpf oeq, %[[LHS_REAL_ABS]], %[[INF]] : f32
// CHECK: %[[LHS_IMAG_ABS:.*]] = math.absf %[[LHS_IMAG]] : f32
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

// -----

// CHECK-LABEL: func @complex_eq
// CHECK-SAME: %[[LHS:.*]]: complex<f32>, %[[RHS:.*]]: complex<f32>
func.func @complex_eq(%lhs: complex<f32>, %rhs: complex<f32>) -> i1 {
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

// -----

// CHECK-LABEL: func @complex_exp
// CHECK-SAME: %[[ARG:.*]]: complex<f32>
func.func @complex_exp(%arg: complex<f32>) -> complex<f32> {
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

// -----

// CHECK-LABEL:   func.func @complex_expm1(
// CHECK-SAME:                             %[[ARG:.*]]: complex<f32>) -> complex<f32> {
func.func @complex_expm1(%arg: complex<f32>) -> complex<f32> {
  %expm1 = complex.expm1 %arg: complex<f32>
  return %expm1 : complex<f32>
}
// CHECK: %[[REAL_I:.*]] = complex.re %[[ARG]] : complex<f32>
// CHECK: %[[IMAG_I:.*]] = complex.im %[[ARG]] : complex<f32>
// CHECK: %[[EXP:.*]] = math.exp %[[REAL_I]] : f32
// CHECK: %[[COS:.*]] = math.cos %[[IMAG_I]] : f32
// CHECK: %[[RES_REAL:.*]] = arith.mulf %[[EXP]], %[[COS]] : f32
// CHECK: %[[SIN:.*]] = math.sin %[[IMAG_I]] : f32
// CHECK: %[[RES_IMAG:.*]] = arith.mulf %[[EXP]], %[[SIN]] : f32
// CHECK: %[[RES_EXP:.*]] = complex.create %[[RES_REAL]], %[[RES_IMAG]] : complex<f32>
// CHECK: %[[REAL:.*]] = complex.re %[[RES_EXP]] : complex<f32>
// CHECK: %[[ONE:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[REAL_M1:.*]] = arith.subf %[[REAL]], %[[ONE]] : f32
// CHECK: %[[IMAG:.*]] = complex.im %[[RES_EXP]] : complex<f32>
// CHECK: %[[RES:.*]] = complex.create %[[REAL_M1]], %[[IMAG]] : complex<f32>
// CHECK: return %[[RES]] : complex<f32>

// -----

// CHECK-LABEL: func @complex_log
// CHECK-SAME: %[[ARG:.*]]: complex<f32>
func.func @complex_log(%arg: complex<f32>) -> complex<f32> {
  %log = complex.log %arg: complex<f32>
  return %log : complex<f32>
}
// CHECK: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[ONE:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[REAL:.*]] = complex.re %[[ARG]] : complex<f32>
// CHECK: %[[IMAG:.*]] = complex.im %[[ARG]] : complex<f32>
// CHECK: %[[IS_REAL_ZERO:.*]] = arith.cmpf oeq, %[[REAL]], %[[ZERO]] : f32
// CHECK: %[[IS_IMAG_ZERO:.*]] = arith.cmpf oeq, %[[IMAG]], %[[ZERO]] : f32
// CHECK: %[[IMAG_DIV_REAL:.*]] = arith.divf %[[IMAG]], %[[REAL]] : f32
// CHECK: %[[IMAG_SQ:.*]] = arith.mulf %[[IMAG_DIV_REAL]], %[[IMAG_DIV_REAL]] : f32
// CHECK: %[[IMAG_SQ_PLUS_ONE:.*]] = arith.addf %[[IMAG_SQ]], %[[ONE]] : f32
// CHECK: %[[IMAG_SQRT:.*]] = math.sqrt %[[IMAG_SQ_PLUS_ONE]] : f32
// CHECK: %[[REAL_ABS:.*]] = math.absf %[[REAL]] : f32
// CHECK: %[[ABS_IMAG:.*]] = arith.mulf %[[IMAG_SQRT]], %[[REAL_ABS]] : f32
// CHECK: %[[REAL_DIV_IMAG:.*]] = arith.divf %[[REAL]], %[[IMAG]] : f32
// CHECK: %[[REAL_SQ:.*]] = arith.mulf %[[REAL_DIV_IMAG]], %[[REAL_DIV_IMAG]] : f32
// CHECK: %[[REAL_SQ_PLUS_ONE:.*]] = arith.addf %[[REAL_SQ]], %[[ONE]] : f32
// CHECK: %[[REAL_SQRT:.*]] = math.sqrt %[[REAL_SQ_PLUS_ONE]] : f32
// CHECK: %[[IMAG_ABS:.*]] = math.absf %[[IMAG]] : f32
// CHECK: %[[ABS_REAL:.*]] = arith.mulf %[[REAL_SQRT]], %[[IMAG_ABS]] : f32
// CHECK: %[[REAL_GT_IMAG:.*]] = arith.cmpf ogt, %[[REAL]], %[[IMAG]] : f32
// CHECK: %[[ABS1:.*]] = arith.select %[[REAL_GT_IMAG]], %[[ABS_IMAG]], %[[ABS_REAL]] : f32
// CHECK: %[[ABS2:.*]] = arith.select %[[IS_IMAG_ZERO]], %[[REAL_ABS]], %[[ABS1]] : f32
// CHECK: %[[NORM:.*]] = arith.select %[[IS_REAL_ZERO]], %[[IMAG_ABS]], %[[ABS2]] : f32
// CHECK: %[[RESULT_REAL:.*]] = math.log %[[NORM]] : f32
// CHECK: %[[REAL2:.*]] = complex.re %[[ARG]] : complex<f32>
// CHECK: %[[IMAG2:.*]] = complex.im %[[ARG]] : complex<f32>
// CHECK: %[[RESULT_IMAG:.*]] = math.atan2 %[[IMAG2]], %[[REAL2]] : f32
// CHECK: %[[RESULT:.*]] = complex.create %[[RESULT_REAL]], %[[RESULT_IMAG]] : complex<f32>
// CHECK: return %[[RESULT]] : complex<f32>

// -----

// CHECK-LABEL: func @complex_log1p
// CHECK-SAME: %[[ARG:.*]]: complex<f32>
func.func @complex_log1p(%arg: complex<f32>) -> complex<f32> {
  %log1p = complex.log1p %arg: complex<f32>
  return %log1p : complex<f32>
}

// CHECK: %[[REAL:.*]] = complex.re %[[ARG]] : complex<f32>
// CHECK: %[[IMAG:.*]] = complex.im %[[ARG]] : complex<f32>
// CHECK: %[[ONE_HALF:.*]] = arith.constant 5.000000e-01 : f32
// CHECK: %[[ONE:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[TWO:.*]] = arith.constant 2.000000e+00 : f32
// CHECK: %[[SQ_SUM_0:.*]] = arith.mulf %[[REAL]], %[[REAL]] : f32
// CHECK: %[[TWO_REAL:.*]] = arith.mulf %[[REAL]], %[[TWO]] : f32
// CHECK: %[[SQ_SUM_1:.*]] = arith.addf %[[SQ_SUM_0]], %[[TWO_REAL]] : f32
// CHECK: %[[SQ_IMAG:.*]] = arith.mulf %[[IMAG]], %[[IMAG]] : f32
// CHECK: %[[SQ_SUM_2:.*]] = arith.addf %[[SQ_SUM_1]], %[[SQ_IMAG]] : f32
// CHECK: %[[LOG_SQ_SUM:.*]] = math.log1p %[[SQ_SUM_2]] : f32
// CHECK: %[[RESULT_REAL:.*]] = arith.mulf %[[LOG_SQ_SUM]], %[[ONE_HALF]] : f32
// CHECK: %[[REAL_PLUS_ONE:.*]] = arith.addf %[[REAL]], %[[ONE]] : f32
// CHECK: %[[RESULT_IMAG:.*]] = math.atan2 %[[IMAG]], %[[REAL_PLUS_ONE]] : f32
// CHECK: %[[RESULT:.*]] = complex.create %[[RESULT_REAL]], %[[RESULT_IMAG]] : complex<f32>
// CHECK: return %[[RESULT]] : complex<f32>

// -----

// CHECK-LABEL: func @complex_mul
// CHECK-SAME: (%[[LHS:.*]]: complex<f32>, %[[RHS:.*]]: complex<f32>)
func.func @complex_mul(%lhs: complex<f32>, %rhs: complex<f32>) -> complex<f32> {
  %mul = complex.mul %lhs, %rhs : complex<f32>
  return %mul : complex<f32>
}
// CHECK: %[[LHS_REAL:.*]] = complex.re %[[LHS]] : complex<f32>
// CHECK: %[[LHS_REAL_ABS:.*]] = math.absf %[[LHS_REAL]] : f32
// CHECK: %[[LHS_IMAG:.*]] = complex.im %[[LHS]] : complex<f32>
// CHECK: %[[LHS_IMAG_ABS:.*]] = math.absf %[[LHS_IMAG]] : f32
// CHECK: %[[RHS_REAL:.*]] = complex.re %[[RHS]] : complex<f32>
// CHECK: %[[RHS_REAL_ABS:.*]] = math.absf %[[RHS_REAL]] : f32
// CHECK: %[[RHS_IMAG:.*]] = complex.im %[[RHS]] : complex<f32>
// CHECK: %[[RHS_IMAG_ABS:.*]] = math.absf %[[RHS_IMAG]] : f32

// CHECK: %[[LHS_REAL_TIMES_RHS_REAL:.*]] = arith.mulf %[[LHS_REAL]], %[[RHS_REAL]] : f32
// CHECK: %[[LHS_REAL_TIMES_RHS_REAL_ABS:.*]] = math.absf %[[LHS_REAL_TIMES_RHS_REAL]] : f32
// CHECK: %[[LHS_IMAG_TIMES_RHS_IMAG:.*]] = arith.mulf %[[LHS_IMAG]], %[[RHS_IMAG]] : f32
// CHECK: %[[LHS_IMAG_TIMES_RHS_IMAG_ABS:.*]] = math.absf %[[LHS_IMAG_TIMES_RHS_IMAG]] : f32
// CHECK: %[[REAL:.*]] = arith.subf %[[LHS_REAL_TIMES_RHS_REAL]], %[[LHS_IMAG_TIMES_RHS_IMAG]] : f32

// CHECK: %[[LHS_IMAG_TIMES_RHS_REAL:.*]] = arith.mulf %[[LHS_IMAG]], %[[RHS_REAL]] : f32
// CHECK: %[[LHS_IMAG_TIMES_RHS_REAL_ABS:.*]] = math.absf %[[LHS_IMAG_TIMES_RHS_REAL]] : f32
// CHECK: %[[LHS_REAL_TIMES_RHS_IMAG:.*]] = arith.mulf %[[LHS_REAL]], %[[RHS_IMAG]] : f32
// CHECK: %[[LHS_REAL_TIMES_RHS_IMAG_ABS:.*]] = math.absf %[[LHS_REAL_TIMES_RHS_IMAG]] : f32
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

// -----

// CHECK-LABEL: func @complex_neg
// CHECK-SAME: %[[ARG:.*]]: complex<f32>
func.func @complex_neg(%arg: complex<f32>) -> complex<f32> {
  %neg = complex.neg %arg: complex<f32>
  return %neg : complex<f32>
}
// CHECK: %[[REAL:.*]] = complex.re %[[ARG]] : complex<f32>
// CHECK: %[[IMAG:.*]] = complex.im %[[ARG]] : complex<f32>
// CHECK-DAG: %[[NEG_REAL:.*]] = arith.negf %[[REAL]] : f32
// CHECK-DAG: %[[NEG_IMAG:.*]] = arith.negf %[[IMAG]] : f32
// CHECK: %[[RESULT:.*]] = complex.create %[[NEG_REAL]], %[[NEG_IMAG]] : complex<f32>
// CHECK: return %[[RESULT]] : complex<f32>

// -----

// CHECK-LABEL: func @complex_neq
// CHECK-SAME: %[[LHS:.*]]: complex<f32>, %[[RHS:.*]]: complex<f32>
func.func @complex_neq(%lhs: complex<f32>, %rhs: complex<f32>) -> i1 {
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

// -----

// CHECK-LABEL: func @complex_sin
// CHECK-SAME: %[[ARG:.*]]: complex<f32>
func.func @complex_sin(%arg: complex<f32>) -> complex<f32> {
  %sin = complex.sin %arg : complex<f32>
  return %sin : complex<f32>
}
// CHECK-DAG: %[[REAL:.*]] = complex.re %[[ARG]]
// CHECK-DAG: %[[IMAG:.*]] = complex.im %[[ARG]]
// CHECK-DAG: %[[HALF:.*]] = arith.constant 5.000000e-01 : f32
// CHECK-DAG: %[[EXP:.*]] = math.exp %[[IMAG]] : f32
// CHECK-DAG: %[[HALF_EXP:.*]] = arith.mulf %[[HALF]], %[[EXP]]
// CHECK-DAG: %[[HALF_REXP:.*]] = arith.divf %[[HALF]], %[[EXP]]
// CHECK-DAG: %[[SIN:.*]] = math.sin %[[REAL]] : f32
// CHECK-DAG: %[[COS:.*]] = math.cos %[[REAL]] : f32
// CHECK-DAG: %[[EXP_SUM:.*]] = arith.addf %[[HALF_EXP]], %[[HALF_REXP]]
// CHECK-DAG: %[[RESULT_REAL:.*]] = arith.mulf %[[EXP_SUM]], %[[SIN]]
// CHECK-DAG: %[[EXP_DIFF:.*]] = arith.subf %[[HALF_EXP]], %[[HALF_REXP]]
// CHECK-DAG: %[[RESULT_IMAG:.*]] = arith.mulf %[[EXP_DIFF]], %[[COS]]
// CHECK-DAG: %[[RESULT:.*]] = complex.create %[[RESULT_REAL]], %[[RESULT_IMAG]] : complex<f32>
// CHECK:     return %[[RESULT]]

// -----

// CHECK-LABEL: func @complex_sign
// CHECK-SAME: %[[ARG:.*]]: complex<f32>
func.func @complex_sign(%arg: complex<f32>) -> complex<f32> {
  %sign = complex.sign %arg: complex<f32>
  return %sign : complex<f32>
}
// CHECK: %[[REAL:.*]] = complex.re %[[ARG]] : complex<f32>
// CHECK: %[[IMAG:.*]] = complex.im %[[ARG]] : complex<f32>
// CHECK: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[REAL_IS_ZERO:.*]] = arith.cmpf oeq, %[[REAL]], %[[ZERO]] : f32
// CHECK: %[[IMAG_IS_ZERO:.*]] = arith.cmpf oeq, %[[IMAG]], %[[ZERO]] : f32
// CHECK: %[[IS_ZERO:.*]] = arith.andi %[[REAL_IS_ZERO]], %[[IMAG_IS_ZERO]] : i1
// CHECK: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[ONE:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[REAL2:.*]] = complex.re %[[ARG]] : complex<f32>
// CHECK: %[[IMAG2:.*]] = complex.im %[[ARG]] : complex<f32>
// CHECK: %[[IS_REAL_ZERO:.*]] = arith.cmpf oeq, %[[REAL2]], %[[ZERO]] : f32
// CHECK: %[[IS_IMAG_ZERO:.*]] = arith.cmpf oeq, %[[IMAG2]], %[[ZERO]] : f32
// CHECK: %[[IMAG_DIV_REAL:.*]] = arith.divf %[[IMAG2]], %[[REAL2]] : f32
// CHECK: %[[IMAG_SQ:.*]] = arith.mulf %[[IMAG_DIV_REAL]], %[[IMAG_DIV_REAL]] : f32
// CHECK: %[[IMAG_SQ_PLUS_ONE:.*]] = arith.addf %[[IMAG_SQ]], %[[ONE]] : f32
// CHECK: %[[IMAG_SQRT:.*]] = math.sqrt %[[IMAG_SQ_PLUS_ONE]] : f32
// CHECK: %[[REAL_ABS:.*]] = math.absf %[[REAL2]] : f32
// CHECK: %[[ABS_IMAG:.*]] = arith.mulf %[[IMAG_SQRT]], %[[REAL_ABS]] : f32
// CHECK: %[[REAL_DIV_IMAG:.*]] = arith.divf %[[REAL2]], %[[IMAG2]] : f32
// CHECK: %[[REAL_SQ:.*]] = arith.mulf %[[REAL_DIV_IMAG]], %[[REAL_DIV_IMAG]] : f32
// CHECK: %[[REAL_SQ_PLUS_ONE:.*]] = arith.addf %[[REAL_SQ]], %[[ONE]] : f32
// CHECK: %[[REAL_SQRT:.*]] = math.sqrt %[[REAL_SQ_PLUS_ONE]] : f32
// CHECK: %[[IMAG_ABS:.*]] = math.absf %[[IMAG2]] : f32
// CHECK: %[[ABS_REAL:.*]] = arith.mulf %[[REAL_SQRT]], %[[IMAG_ABS]] : f32
// CHECK: %[[REAL_GT_IMAG:.*]] = arith.cmpf ogt, %[[REAL2]], %[[IMAG2]] : f32
// CHECK: %[[ABS1:.*]] = arith.select %[[REAL_GT_IMAG]], %[[ABS_IMAG]], %[[ABS_REAL]] : f32
// CHECK: %[[ABS2:.*]] = arith.select %[[IS_IMAG_ZERO]], %[[REAL_ABS]], %[[ABS1]] : f32
// CHECK: %[[NORM:.*]] = arith.select %[[IS_REAL_ZERO]], %[[IMAG_ABS]], %[[ABS2]] : f32
// CHECK: %[[REAL_SIGN:.*]] = arith.divf %[[REAL]], %[[NORM]] : f32
// CHECK: %[[IMAG_SIGN:.*]] = arith.divf %[[IMAG]], %[[NORM]] : f32
// CHECK: %[[SIGN:.*]] = complex.create %[[REAL_SIGN]], %[[IMAG_SIGN]] : complex<f32>
// CHECK: %[[RESULT:.*]] = arith.select %[[IS_ZERO]], %[[ARG]], %[[SIGN]] : complex<f32>
// CHECK: return %[[RESULT]] : complex<f32>

// -----

// CHECK-LABEL: func @complex_sub
// CHECK-SAME: (%[[LHS:.*]]: complex<f32>, %[[RHS:.*]]: complex<f32>)
func.func @complex_sub(%lhs: complex<f32>, %rhs: complex<f32>) -> complex<f32> {
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

// -----

// CHECK-LABEL: func @complex_tan
// CHECK-SAME: %[[ARG:.*]]: complex<f32>
func.func @complex_tan(%arg: complex<f32>) -> complex<f32> {
  %tan = complex.tan %arg: complex<f32>
  return %tan : complex<f32>
}
// CHECK-DAG: %[[REAL:.*]] = complex.re %[[ARG]]
// CHECK-DAG: %[[IMAG:.*]] = complex.im %[[ARG]]
// CHECK-DAG: %[[HALF:.*]] = arith.constant 5.000000e-01 : f32
// CHECK-DAG: %[[EXP:.*]] = math.exp %[[IMAG]] : f32
// CHECK-DAG: %[[HALF_EXP:.*]] = arith.mulf %[[HALF]], %[[EXP]]
// CHECK-DAG: %[[HALF_REXP:.*]] = arith.divf %[[HALF]], %[[EXP]]
// CHECK-DAG: %[[SIN:.*]] = math.sin %[[REAL]] : f32
// CHECK-DAG: %[[COS:.*]] = math.cos %[[REAL]] : f32
// CHECK-DAG: %[[EXP_SUM:.*]] = arith.addf %[[HALF_REXP]], %[[HALF_EXP]]
// CHECK-DAG: %[[COS_REAL:.*]] = arith.mulf %[[EXP_SUM]], %[[COS]]
// CHECK-DAG: %[[EXP_DIFF:.*]] = arith.subf %[[HALF_REXP]], %[[HALF_EXP]]
// CHECK-DAG: %[[COS_IMAG:.*]] = arith.mulf %[[EXP_DIFF]], %[[SIN]]
// CHECK-DAG: %[[COS_COMP:.*]] = complex.create %[[COS_REAL]], %[[COS_IMAG]] : complex<f32>

// CHECK-DAG: %[[REAL:.*]] = complex.re %[[ARG]]
// CHECK-DAG: %[[IMAG:.*]] = complex.im %[[ARG]]
// CHECK-DAG: %[[HALF:.*]] = arith.constant 5.000000e-01 : f32
// CHECK-DAG: %[[EXP:.*]] = math.exp %[[IMAG]] : f32
// CHECK-DAG: %[[HALF_EXP:.*]] = arith.mulf %[[HALF]], %[[EXP]]
// CHECK-DAG: %[[HALF_REXP:.*]] = arith.divf %[[HALF]], %[[EXP]]
// CHECK-DAG: %[[SIN:.*]] = math.sin %[[REAL]] : f32
// CHECK-DAG: %[[COS:.*]] = math.cos %[[REAL]] : f32
// CHECK-DAG: %[[EXP_SUM:.*]] = arith.addf %[[HALF_EXP]], %[[HALF_REXP]]
// CHECK-DAG: %[[SIN_REAL:.*]] = arith.mulf %[[EXP_SUM]], %[[SIN]]
// CHECK-DAG: %[[EXP_DIFF:.*]] = arith.subf %[[HALF_EXP]], %[[HALF_REXP]]
// CHECK-DAG: %[[SIN_IMAG:.*]] = arith.mulf %[[EXP_DIFF]], %[[COS]]
// CHECK-DAG: %[[SIN_COMP:.*]] = complex.create %[[SIN_REAL]], %[[SIN_IMAG]] : complex<f32>

// CHECK: %[[LHS_REAL:.*]] = complex.re %[[SIN_COMP]] : complex<f32>
// CHECK: %[[LHS_IMAG:.*]] = complex.im %[[SIN_COMP]] : complex<f32>
// CHECK: %[[RHS_REAL:.*]] = complex.re %[[COS_COMP]] : complex<f32>
// CHECK: %[[RHS_IMAG:.*]] = complex.im %[[COS_COMP]] : complex<f32>

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
// CHECK: %[[RHS_REAL_ABS:.*]] = math.absf %[[RHS_REAL]] : f32
// CHECK: %[[RHS_REAL_ABS_IS_ZERO:.*]] = arith.cmpf oeq, %[[RHS_REAL_ABS]], %[[ZERO]] : f32
// CHECK: %[[RHS_IMAG_ABS:.*]] = math.absf %[[RHS_IMAG]] : f32
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
// CHECK: %[[LHS_REAL_ABS:.*]] = math.absf %[[LHS_REAL]] : f32
// CHECK: %[[LHS_REAL_INFINITE:.*]] = arith.cmpf oeq, %[[LHS_REAL_ABS]], %[[INF]] : f32
// CHECK: %[[LHS_IMAG_ABS:.*]] = math.absf %[[LHS_IMAG]] : f32
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

// -----

// CHECK-LABEL: func @complex_tanh
// CHECK-SAME: %[[ARG:.*]]: complex<f32>
func.func @complex_tanh(%arg: complex<f32>) -> complex<f32> {
  %tanh = complex.tanh %arg: complex<f32>
  return %tanh : complex<f32>
}
// CHECK: %[[REAL:.*]] = complex.re %[[ARG]] : complex<f32>
// CHECK: %[[IMAG:.*]] = complex.im %[[ARG]] : complex<f32>
// CHECK: %[[TANH_A:.*]] = math.tanh %[[REAL]] : f32
// CHECK: %[[COS_B:.*]] = math.cos %[[IMAG]] : f32
// CHECK: %[[SIN_B:.*]] = math.sin %[[IMAG]] : f32
// CHECK: %[[TAN_B:.*]] = arith.divf %[[SIN_B]], %[[COS_B]] : f32
// CHECK: %[[NUM:.*]] = complex.create %[[TANH_A]], %[[TAN_B]] : complex<f32>
// CHECK: %[[ONE:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[MUL:.*]] = arith.mulf %[[TANH_A]], %[[TAN_B]] : f32
// CHECK: %[[DENOM:.*]] = complex.create %[[ONE]], %[[MUL]] : complex<f32>

// -----

// CHECK-LABEL: func @complex_sqrt
func.func @complex_sqrt(%arg: complex<f32>) -> complex<f32> {
  %sqrt = complex.sqrt %arg : complex<f32>
  return %sqrt : complex<f32>
}

// -----

// CHECK-LABEL: func @complex_conj
// CHECK-SAME: %[[ARG:.*]]: complex<f32>
func.func @complex_conj(%arg: complex<f32>) -> complex<f32> {
  %conj = complex.conj %arg: complex<f32>
  return %conj : complex<f32>
}
// CHECK: %[[REAL:.*]] = complex.re %[[ARG]] : complex<f32>
// CHECK: %[[IMAG:.*]] = complex.im %[[ARG]] : complex<f32>
// CHECK: %[[NEG_IMAG:.*]] = arith.negf %[[IMAG]] : f32
// CHECK: %[[RESULT:.*]] = complex.create %[[REAL]], %[[NEG_IMAG]] : complex<f32>
// CHECK: return %[[RESULT]] : complex<f32>

// -----

// CHECK-LABEL:   func.func @complex_pow
func.func @complex_pow(%lhs: complex<f32>,
                       %rhs: complex<f32>) -> complex<f32> {
  %pow = complex.pow %lhs, %rhs : complex<f32>
  return %pow : complex<f32>
}

// -----

// CHECK-LABEL:   func.func @complex_rsqrt
func.func @complex_rsqrt(%arg: complex<f32>) -> complex<f32> {
  %rsqrt = complex.rsqrt %arg : complex<f32>
  return %rsqrt : complex<f32>
}

// -----

// CHECK-LABEL:   func.func @complex_angle
// CHECK-SAME: %[[ARG:.*]]: complex<f32>
func.func @complex_angle(%arg: complex<f32>) -> f32 {
  %angle = complex.angle %arg : complex<f32>
  return %angle : f32
}
// CHECK: %[[REAL:.*]] = complex.re %[[ARG]] : complex<f32>
// CHECK: %[[IMAG:.*]] = complex.im %[[ARG]] : complex<f32>
// CHECK: %[[RESULT:.*]] = math.atan2 %[[IMAG]], %[[REAL]] : f32
// CHECK: return %[[RESULT]] : f32

// -----

// CHECK-LABEL: func @complex_abs_with_fmf
// CHECK-SAME: %[[ARG:.*]]: complex<f32>
func.func @complex_abs_with_fmf(%arg: complex<f32>) -> f32 {
  %abs = complex.abs %arg fastmath<nnan,contract> : complex<f32>
  return %abs : f32
}
// CHECK: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[ONE:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[REAL:.*]] = complex.re %[[ARG]] : complex<f32>
// CHECK: %[[IMAG:.*]] = complex.im %[[ARG]] : complex<f32>
// CHECK: %[[IS_REAL_ZERO:.*]] = arith.cmpf oeq, %[[REAL]], %[[ZERO]] : f32
// CHECK: %[[IS_IMAG_ZERO:.*]] = arith.cmpf oeq, %[[IMAG]], %[[ZERO]] : f32
// CHECK: %[[IMAG_DIV_REAL:.*]] = arith.divf %[[IMAG]], %[[REAL]] fastmath<nnan,contract> : f32
// CHECK: %[[IMAG_SQ:.*]] = arith.mulf %[[IMAG_DIV_REAL]], %[[IMAG_DIV_REAL]] fastmath<nnan,contract> : f32
// CHECK: %[[IMAG_SQ_PLUS_ONE:.*]] = arith.addf %[[IMAG_SQ]], %[[ONE]] fastmath<nnan,contract> : f32
// CHECK: %[[IMAG_SQRT:.*]] = math.sqrt %[[IMAG_SQ_PLUS_ONE]] fastmath<nnan,contract> : f32
// CHECK: %[[REAL_ABS:.*]] = math.absf %[[REAL]] fastmath<nnan,contract> : f32
// CHECK: %[[ABS_IMAG:.*]] = arith.mulf %[[IMAG_SQRT]], %[[REAL_ABS]] fastmath<nnan,contract> : f32
// CHECK: %[[REAL_DIV_IMAG:.*]] = arith.divf %[[REAL]], %[[IMAG]] fastmath<nnan,contract> : f32
// CHECK: %[[REAL_SQ:.*]] = arith.mulf %[[REAL_DIV_IMAG]], %[[REAL_DIV_IMAG]] fastmath<nnan,contract> : f32
// CHECK: %[[REAL_SQ_PLUS_ONE:.*]] = arith.addf %[[REAL_SQ]], %[[ONE]] fastmath<nnan,contract> : f32
// CHECK: %[[REAL_SQRT:.*]] = math.sqrt %[[REAL_SQ_PLUS_ONE]] fastmath<nnan,contract> : f32
// CHECK: %[[IMAG_ABS:.*]] = math.absf %[[IMAG]] fastmath<nnan,contract> : f32
// CHECK: %[[ABS_REAL:.*]] = arith.mulf %[[REAL_SQRT]], %[[IMAG_ABS]] fastmath<nnan,contract> : f32
// CHECK: %[[REAL_GT_IMAG:.*]] = arith.cmpf ogt, %[[REAL]], %[[IMAG]] : f32
// CHECK: %[[ABS1:.*]] = arith.select %[[REAL_GT_IMAG]], %[[ABS_IMAG]], %[[ABS_REAL]] : f32
// CHECK: %[[ABS2:.*]] = arith.select %[[IS_IMAG_ZERO]], %[[REAL_ABS]], %[[ABS1]] : f32
// CHECK: %[[ABS3:.*]] = arith.select %[[IS_REAL_ZERO]], %[[IMAG_ABS]], %[[ABS2]] : f32
// CHECK: return %[[ABS3]] : f32

// -----

// CHECK-LABEL: func @complex_add_with_fmf
// CHECK-SAME: (%[[LHS:.*]]: complex<f32>, %[[RHS:.*]]: complex<f32>)
func.func @complex_add_with_fmf(%lhs: complex<f32>, %rhs: complex<f32>) -> complex<f32> {
  %add = complex.add %lhs, %rhs fastmath<nnan,contract> : complex<f32>
  return %add : complex<f32>
}
// CHECK: %[[REAL_LHS:.*]] = complex.re %[[LHS]] : complex<f32>
// CHECK: %[[REAL_RHS:.*]] = complex.re %[[RHS]] : complex<f32>
// CHECK: %[[RESULT_REAL:.*]] = arith.addf %[[REAL_LHS]], %[[REAL_RHS]] fastmath<nnan,contract> : f32
// CHECK: %[[IMAG_LHS:.*]] = complex.im %[[LHS]] : complex<f32>
// CHECK: %[[IMAG_RHS:.*]] = complex.im %[[RHS]] : complex<f32>
// CHECK: %[[RESULT_IMAG:.*]] = arith.addf %[[IMAG_LHS]], %[[IMAG_RHS]] fastmath<nnan,contract> : f32
// CHECK: %[[RESULT:.*]] = complex.create %[[RESULT_REAL]], %[[RESULT_IMAG]] : complex<f32>
// CHECK: return %[[RESULT]] : complex<f32>

// -----

// CHECK-LABEL: func @complex_sub_with_fmf
// CHECK-SAME: (%[[LHS:.*]]: complex<f32>, %[[RHS:.*]]: complex<f32>)
func.func @complex_sub_with_fmf(%lhs: complex<f32>, %rhs: complex<f32>) -> complex<f32> {
  %sub = complex.sub %lhs, %rhs fastmath<nnan,contract> : complex<f32>
  return %sub : complex<f32>
}
// CHECK: %[[REAL_LHS:.*]] = complex.re %[[LHS]] : complex<f32>
// CHECK: %[[REAL_RHS:.*]] = complex.re %[[RHS]] : complex<f32>
// CHECK: %[[RESULT_REAL:.*]] = arith.subf %[[REAL_LHS]], %[[REAL_RHS]] fastmath<nnan,contract> : f32
// CHECK: %[[IMAG_LHS:.*]] = complex.im %[[LHS]] : complex<f32>
// CHECK: %[[IMAG_RHS:.*]] = complex.im %[[RHS]] : complex<f32>
// CHECK: %[[RESULT_IMAG:.*]] = arith.subf %[[IMAG_LHS]], %[[IMAG_RHS]] fastmath<nnan,contract> : f32
// CHECK: %[[RESULT:.*]] = complex.create %[[RESULT_REAL]], %[[RESULT_IMAG]] : complex<f32>
// CHECK: return %[[RESULT]] : complex<f32>

// -----

// CHECK-LABEL: func @complex_exp_with_fmf
// CHECK-SAME: %[[ARG:.*]]: complex<f32>
func.func @complex_exp_with_fmf(%arg: complex<f32>) -> complex<f32> {
  %exp = complex.exp %arg fastmath<nnan,contract> : complex<f32>
  return %exp : complex<f32>
}
// CHECK: %[[REAL:.*]] = complex.re %[[ARG]] : complex<f32>
// CHECK: %[[IMAG:.*]] = complex.im %[[ARG]] : complex<f32>
// CHECK-DAG: %[[COS_IMAG:.*]] = math.cos %[[IMAG]] fastmath<nnan,contract> : f32
// CHECK-DAG: %[[EXP_REAL:.*]] = math.exp %[[REAL]] fastmath<nnan,contract> : f32
// CHECK-DAG: %[[RESULT_REAL:.]] = arith.mulf %[[EXP_REAL]], %[[COS_IMAG]] fastmath<nnan,contract> : f32
// CHECK-DAG: %[[SIN_IMAG:.*]] = math.sin %[[IMAG]] fastmath<nnan,contract> : f32
// CHECK-DAG: %[[RESULT_IMAG:.*]] = arith.mulf %[[EXP_REAL]], %[[SIN_IMAG]] fastmath<nnan,contract> : f32
// CHECK: %[[RESULT:.*]] = complex.create %[[RESULT_REAL]], %[[RESULT_IMAG]] : complex<f32>
// CHECK: return %[[RESULT]] : complex<f32>

// -----

// CHECK-LABEL:   func.func @complex_expm1_with_fmf(
// CHECK-SAME:                             %[[ARG:.*]]: complex<f32>) -> complex<f32> {
func.func @complex_expm1_with_fmf(%arg: complex<f32>) -> complex<f32> {
  %expm1 = complex.expm1 %arg fastmath<nnan,contract> : complex<f32>
  return %expm1 : complex<f32>
}
// CHECK: %[[REAL_I:.*]] = complex.re %[[ARG]] : complex<f32>
// CHECK: %[[IMAG_I:.*]] = complex.im %[[ARG]] : complex<f32>
// CHECK: %[[EXP:.*]] = math.exp %[[REAL_I]] fastmath<nnan,contract> : f32
// CHECK: %[[COS:.*]] = math.cos %[[IMAG_I]] fastmath<nnan,contract> : f32
// CHECK: %[[RES_REAL:.*]] = arith.mulf %[[EXP]], %[[COS]] fastmath<nnan,contract> : f32
// CHECK: %[[SIN:.*]] = math.sin %[[IMAG_I]] fastmath<nnan,contract> : f32
// CHECK: %[[RES_IMAG:.*]] = arith.mulf %[[EXP]], %[[SIN]] fastmath<nnan,contract> : f32
// CHECK: %[[RES_EXP:.*]] = complex.create %[[RES_REAL]], %[[RES_IMAG]] : complex<f32>
// CHECK: %[[REAL:.*]] = complex.re %[[RES_EXP]] : complex<f32>
// CHECK: %[[ONE:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[REAL_M1:.*]] = arith.subf %[[REAL]], %[[ONE]] fastmath<nnan,contract> : f32
// CHECK: %[[IMAG:.*]] = complex.im %[[RES_EXP]] : complex<f32>
// CHECK: %[[RES:.*]] = complex.create %[[REAL_M1]], %[[IMAG]] : complex<f32>
// CHECK: return %[[RES]] : complex<f32>

// -----

// CHECK-LABEL: func @complex_log_with_fmf
// CHECK-SAME: %[[ARG:.*]]: complex<f32>
func.func @complex_log_with_fmf(%arg: complex<f32>) -> complex<f32> {
  %log = complex.log %arg fastmath<nnan,contract> : complex<f32>
  return %log : complex<f32>
}
// CHECK: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[ONE:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[REAL:.*]] = complex.re %[[ARG]] : complex<f32>
// CHECK: %[[IMAG:.*]] = complex.im %[[ARG]] : complex<f32>
// CHECK: %[[IS_REAL_ZERO:.*]] = arith.cmpf oeq, %[[REAL]], %[[ZERO]] : f32
// CHECK: %[[IS_IMAG_ZERO:.*]] = arith.cmpf oeq, %[[IMAG]], %[[ZERO]] : f32
// CHECK: %[[IMAG_DIV_REAL:.*]] = arith.divf %[[IMAG]], %[[REAL]] fastmath<nnan,contract> : f32
// CHECK: %[[IMAG_SQ:.*]] = arith.mulf %[[IMAG_DIV_REAL]], %[[IMAG_DIV_REAL]] fastmath<nnan,contract> : f32
// CHECK: %[[IMAG_SQ_PLUS_ONE:.*]] = arith.addf %[[IMAG_SQ]], %[[ONE]] fastmath<nnan,contract> : f32
// CHECK: %[[IMAG_SQRT:.*]] = math.sqrt %[[IMAG_SQ_PLUS_ONE]] fastmath<nnan,contract> : f32
// CHECK: %[[REAL_ABS:.*]] = math.absf %[[REAL]] fastmath<nnan,contract> : f32
// CHECK: %[[ABS_IMAG:.*]] = arith.mulf %[[IMAG_SQRT]], %[[REAL_ABS]] fastmath<nnan,contract> : f32
// CHECK: %[[REAL_DIV_IMAG:.*]] = arith.divf %[[REAL]], %[[IMAG]] fastmath<nnan,contract> : f32
// CHECK: %[[REAL_SQ:.*]] = arith.mulf %[[REAL_DIV_IMAG]], %[[REAL_DIV_IMAG]] fastmath<nnan,contract> : f32
// CHECK: %[[REAL_SQ_PLUS_ONE:.*]] = arith.addf %[[REAL_SQ]], %[[ONE]] fastmath<nnan,contract> : f32
// CHECK: %[[REAL_SQRT:.*]] = math.sqrt %[[REAL_SQ_PLUS_ONE]] fastmath<nnan,contract> : f32
// CHECK: %[[IMAG_ABS:.*]] = math.absf %[[IMAG]] fastmath<nnan,contract> : f32
// CHECK: %[[ABS_REAL:.*]] = arith.mulf %[[REAL_SQRT]], %[[IMAG_ABS]] fastmath<nnan,contract> : f32
// CHECK: %[[REAL_GT_IMAG:.*]] = arith.cmpf ogt, %[[REAL]], %[[IMAG]] : f32
// CHECK: %[[ABS1:.*]] = arith.select %[[REAL_GT_IMAG]], %[[ABS_IMAG]], %[[ABS_REAL]] : f32
// CHECK: %[[ABS2:.*]] = arith.select %[[IS_IMAG_ZERO]], %[[REAL_ABS]], %[[ABS1]] : f32
// CHECK: %[[NORM:.*]] = arith.select %[[IS_REAL_ZERO]], %[[IMAG_ABS]], %[[ABS2]] : f32
// CHECK: %[[RESULT_REAL:.*]] = math.log %[[NORM]] fastmath<nnan,contract> : f32
// CHECK: %[[REAL2:.*]] = complex.re %[[ARG]] : complex<f32>
// CHECK: %[[IMAG2:.*]] = complex.im %[[ARG]] : complex<f32>
// CHECK: %[[RESULT_IMAG:.*]] = math.atan2 %[[IMAG2]], %[[REAL2]] fastmath<nnan,contract> : f32
// CHECK: %[[RESULT:.*]] = complex.create %[[RESULT_REAL]], %[[RESULT_IMAG]] : complex<f32>
// CHECK: return %[[RESULT]] : complex<f32>

// -----

// CHECK-LABEL: func @complex_log1p_with_fmf
// CHECK-SAME: %[[ARG:.*]]: complex<f32>
func.func @complex_log1p_with_fmf(%arg: complex<f32>) -> complex<f32> {
  %log1p = complex.log1p %arg fastmath<nnan,contract> : complex<f32>
  return %log1p : complex<f32>
}

// CHECK: %[[REAL:.*]] = complex.re %[[ARG]] : complex<f32>
// CHECK: %[[IMAG:.*]] = complex.im %[[ARG]] : complex<f32>
// CHECK: %[[ONE_HALF:.*]] = arith.constant 5.000000e-01 : f32
// CHECK: %[[ONE:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[TWO:.*]] = arith.constant 2.000000e+00 : f32
// CHECK: %[[SQ_SUM_0:.*]] = arith.mulf %[[REAL]], %[[REAL]] fastmath<nnan,contract> : f32
// CHECK: %[[TWO_REAL:.*]] = arith.mulf %[[REAL]], %[[TWO]] fastmath<nnan,contract> : f32
// CHECK: %[[SQ_SUM_1:.*]] = arith.addf %[[SQ_SUM_0]], %[[TWO_REAL]] fastmath<nnan,contract> : f32
// CHECK: %[[SQ_IMAG:.*]] = arith.mulf %[[IMAG]], %[[IMAG]] fastmath<nnan,contract> : f32
// CHECK: %[[SQ_SUM_2:.*]] = arith.addf %[[SQ_SUM_1]], %[[SQ_IMAG]] fastmath<nnan,contract> : f32
// CHECK: %[[LOG_SQ_SUM:.*]] = math.log1p %[[SQ_SUM_2]] fastmath<nnan,contract> : f32
// CHECK: %[[RESULT_REAL:.*]] = arith.mulf %[[LOG_SQ_SUM]], %[[ONE_HALF]] fastmath<nnan,contract> : f32
// CHECK: %[[REAL_PLUS_ONE:.*]] = arith.addf %[[REAL]], %[[ONE]] fastmath<nnan,contract> : f32
// CHECK: %[[RESULT_IMAG:.*]] = math.atan2 %[[IMAG]], %[[REAL_PLUS_ONE]] fastmath<nnan,contract> : f32
// CHECK: %[[RESULT:.*]] = complex.create %[[RESULT_REAL]], %[[RESULT_IMAG]] : complex<f32>
// CHECK: return %[[RESULT]] : complex<f32>

// -----

// CHECK-LABEL: func @complex_mul_with_fmf
// CHECK-SAME: (%[[LHS:.*]]: complex<f32>, %[[RHS:.*]]: complex<f32>)
func.func @complex_mul_with_fmf(%lhs: complex<f32>, %rhs: complex<f32>) -> complex<f32> {
  %mul = complex.mul %lhs, %rhs fastmath<nnan,contract> : complex<f32>
  return %mul : complex<f32>
}
// CHECK: %[[LHS_REAL:.*]] = complex.re %[[LHS]] : complex<f32>
// CHECK: %[[LHS_REAL_ABS:.*]] = math.absf %[[LHS_REAL]] fastmath<nnan,contract> : f32
// CHECK: %[[LHS_IMAG:.*]] = complex.im %[[LHS]] : complex<f32>
// CHECK: %[[LHS_IMAG_ABS:.*]] = math.absf %[[LHS_IMAG]] fastmath<nnan,contract> : f32
// CHECK: %[[RHS_REAL:.*]] = complex.re %[[RHS]] : complex<f32>
// CHECK: %[[RHS_REAL_ABS:.*]] = math.absf %[[RHS_REAL]] fastmath<nnan,contract> : f32
// CHECK: %[[RHS_IMAG:.*]] = complex.im %[[RHS]] : complex<f32>
// CHECK: %[[RHS_IMAG_ABS:.*]] = math.absf %[[RHS_IMAG]] fastmath<nnan,contract> : f32

// CHECK: %[[LHS_REAL_TIMES_RHS_REAL:.*]] = arith.mulf %[[LHS_REAL]], %[[RHS_REAL]] fastmath<nnan,contract> : f32
// CHECK: %[[LHS_REAL_TIMES_RHS_REAL_ABS:.*]] = math.absf %[[LHS_REAL_TIMES_RHS_REAL]] fastmath<nnan,contract> : f32
// CHECK: %[[LHS_IMAG_TIMES_RHS_IMAG:.*]] = arith.mulf %[[LHS_IMAG]], %[[RHS_IMAG]] fastmath<nnan,contract> : f32
// CHECK: %[[LHS_IMAG_TIMES_RHS_IMAG_ABS:.*]] = math.absf %[[LHS_IMAG_TIMES_RHS_IMAG]] fastmath<nnan,contract> : f32
// CHECK: %[[REAL:.*]] = arith.subf %[[LHS_REAL_TIMES_RHS_REAL]], %[[LHS_IMAG_TIMES_RHS_IMAG]] fastmath<nnan,contract> : f32

// CHECK: %[[LHS_IMAG_TIMES_RHS_REAL:.*]] = arith.mulf %[[LHS_IMAG]], %[[RHS_REAL]] fastmath<nnan,contract> : f32
// CHECK: %[[LHS_IMAG_TIMES_RHS_REAL_ABS:.*]] = math.absf %[[LHS_IMAG_TIMES_RHS_REAL]] fastmath<nnan,contract> : f32
// CHECK: %[[LHS_REAL_TIMES_RHS_IMAG:.*]] = arith.mulf %[[LHS_REAL]], %[[RHS_IMAG]] fastmath<nnan,contract> : f32
// CHECK: %[[LHS_REAL_TIMES_RHS_IMAG_ABS:.*]] = math.absf %[[LHS_REAL_TIMES_RHS_IMAG]] fastmath<nnan,contract> : f32
// CHECK: %[[IMAG:.*]] = arith.addf %[[LHS_IMAG_TIMES_RHS_REAL]], %[[LHS_REAL_TIMES_RHS_IMAG]] fastmath<nnan,contract> : f32

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
// CHECK: %[[LHS_REAL_TIMES_RHS_REAL:.*]] = arith.mulf %[[LHS_REAL3]], %[[RHS_REAL3]] fastmath<nnan,contract> : f32
// CHECK: %[[LHS_IMAG_TIMES_RHS_IMAG:.*]] = arith.mulf %[[LHS_IMAG3]], %[[RHS_IMAG3]] fastmath<nnan,contract> : f32
// CHECK: %[[NEW_REAL:.*]] = arith.subf %[[LHS_REAL_TIMES_RHS_REAL]], %[[LHS_IMAG_TIMES_RHS_IMAG]] fastmath<nnan,contract> : f32
// CHECK: %[[NEW_REAL_TIMES_INF:.*]] = arith.mulf %[[INF]], %[[NEW_REAL]] fastmath<nnan,contract> : f32
// CHECK: %[[FINAL_REAL:.*]] = arith.select %[[RECALC3]], %[[NEW_REAL_TIMES_INF]], %[[REAL]] : f32

// Recalculate imag part.
// CHECK: %[[LHS_IMAG_TIMES_RHS_REAL:.*]] = arith.mulf %[[LHS_IMAG3]], %[[RHS_REAL3]] fastmath<nnan,contract> : f32
// CHECK: %[[LHS_REAL_TIMES_RHS_IMAG:.*]] = arith.mulf %[[LHS_REAL3]], %[[RHS_IMAG3]] fastmath<nnan,contract> : f32
// CHECK: %[[NEW_IMAG:.*]] = arith.addf %[[LHS_IMAG_TIMES_RHS_REAL]], %[[LHS_REAL_TIMES_RHS_IMAG]] fastmath<nnan,contract> : f32
// CHECK: %[[NEW_IMAG_TIMES_INF:.*]] = arith.mulf %[[INF]], %[[NEW_IMAG]] fastmath<nnan,contract> : f32
// CHECK: %[[FINAL_IMAG:.*]] = arith.select %[[RECALC3]], %[[NEW_IMAG_TIMES_INF]], %[[IMAG]] : f32

// CHECK: %[[RESULT:.*]] = complex.create %[[FINAL_REAL]], %[[FINAL_IMAG]] : complex<f32>
// CHECK: return %[[RESULT]] : complex<f32>

// -----

// CHECK-LABEL: func @complex_atan2_with_fmf
func.func @complex_atan2_with_fmf(%lhs: complex<f32>,
                         %rhs: complex<f32>) -> complex<f32> {
  %atan2 = complex.atan2 %lhs, %rhs fastmath<nnan,contract> : complex<f32>
  return %atan2 : complex<f32>
}

// CHECK: %[[VAR0:.*]] = complex.re %arg1 : complex<f32>
// CHECK: %[[VAR1:.*]] = math.absf %[[VAR0]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR2:.*]] = complex.im %arg1 : complex<f32>
// CHECK: %[[VAR3:.*]] = math.absf %[[VAR2]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR4:.*]] = complex.re %arg1 : complex<f32>
// CHECK: %[[VAR5:.*]] = math.absf %[[VAR4]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR6:.*]] = complex.im %arg1 : complex<f32>
// CHECK: %[[VAR7:.*]] = math.absf %[[VAR6]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR8:.*]] = arith.mulf %[[VAR0]], %[[VAR4]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR9:.*]] = math.absf %[[VAR8]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR10:.*]] = arith.mulf %[[VAR2]], %[[VAR6]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR11:.*]] = math.absf %[[VAR10]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR12:.*]] = arith.subf %[[VAR8]], %[[VAR10]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR13:.*]] = arith.mulf %[[VAR2]], %[[VAR4]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR14:.*]] = math.absf %[[VAR13]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR15:.*]] = arith.mulf %[[VAR0]], %[[VAR6]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR16:.*]] = math.absf %[[VAR15]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR17:.*]] = arith.addf %[[VAR13]], %[[VAR15]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR18:.*]] = arith.cmpf uno, %[[VAR12]], %[[VAR12]] : f32
// CHECK: %[[VAR19:.*]] = arith.cmpf uno, %[[VAR17]], %[[VAR17]] : f32
// CHECK: %[[VAR20:.*]] = arith.andi %[[VAR18]], %[[VAR19]] : i1
// CHECK: %[[CST:.*]] = arith.constant 0x7F800000 : f32
// CHECK: %[[VAR21:.*]] = arith.cmpf oeq, %[[VAR1]], %[[CST]] : f32
// CHECK: %[[VAR22:.*]] = arith.cmpf oeq, %[[VAR3]], %[[CST]] : f32
// CHECK: %[[VAR23:.*]] = arith.ori %[[VAR21]], %[[VAR22]] : i1
// CHECK: %[[VAR24:.*]] = arith.cmpf uno, %[[VAR4]], %[[VAR4]] : f32
// CHECK: %[[VAR25:.*]] = arith.cmpf uno, %[[VAR6]], %[[VAR6]] : f32
// CHECK: %[[CST_0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[CST_1:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[VAR26:.*]] = arith.select %[[VAR21]], %[[CST_1]], %[[CST_0]] : f32
// CHECK: %[[VAR27:.*]] = math.copysign %[[VAR26]], %[[VAR0]] : f32
// CHECK: %[[VAR28:.*]] = arith.select %[[VAR23]], %[[VAR27]], %[[VAR0]] : f32
// CHECK: %[[VAR29:.*]] = arith.select %[[VAR22]], %[[CST_1]], %[[CST_0]] : f32
// CHECK: %[[VAR30:.*]] = math.copysign %[[VAR29]], %[[VAR2]] : f32
// CHECK: %[[VAR31:.*]] = arith.select %[[VAR23]], %[[VAR30]], %[[VAR2]] : f32
// CHECK: %[[VAR32:.*]] = arith.andi %[[VAR23]], %[[VAR24]] : i1
// CHECK: %[[VAR33:.*]] = math.copysign %[[CST_0]], %[[VAR4]] : f32
// CHECK: %[[VAR34:.*]] = arith.select %[[VAR32]], %[[VAR33]], %[[VAR4]] : f32
// CHECK: %[[VAR35:.*]] = arith.andi %[[VAR23]], %[[VAR25]] : i1
// CHECK: %[[VAR36:.*]] = math.copysign %[[CST_0]], %[[VAR6]] : f32
// CHECK: %[[VAR37:.*]] = arith.select %[[VAR35]], %[[VAR36]], %[[VAR6]] : f32
// CHECK: %[[VAR38:.*]] = arith.cmpf oeq, %[[VAR5]], %cst : f32
// CHECK: %[[VAR39:.*]] = arith.cmpf oeq, %[[VAR7]], %cst : f32
// CHECK: %[[VAR40:.*]] = arith.ori %[[VAR38]], %[[VAR39]] : i1
// CHECK: %[[VAR41:.*]] = arith.cmpf uno, %[[VAR28]], %[[VAR28]] : f32
// CHECK: %[[VAR42:.*]] = arith.cmpf uno, %[[VAR31]], %[[VAR31]] : f32
// CHECK: %[[VAR43:.*]] = arith.select %[[VAR38]], %[[CST_1]], %[[CST_0]] : f32
// CHECK: %[[VAR44:.*]] = math.copysign %[[VAR43]], %[[VAR34]] : f32
// CHECK: %[[VAR45:.*]] = arith.select %[[VAR40]], %[[VAR44]], %[[VAR34]] : f32
// CHECK: %[[VAR46:.*]] = arith.select %[[VAR39]], %[[CST_1]], %[[CST_0]] : f32
// CHECK: %[[VAR47:.*]] = math.copysign %[[VAR46]], %[[VAR37]] : f32
// CHECK: %[[VAR48:.*]] = arith.select %[[VAR40]], %[[VAR47]], %[[VAR37]] : f32
// CHECK: %[[VAR49:.*]] = arith.andi %[[VAR40]], %[[VAR41]] : i1
// CHECK: %[[VAR50:.*]] = math.copysign %[[CST_0]], %[[VAR28]] : f32
// CHECK: %[[VAR51:.*]] = arith.select %[[VAR49]], %[[VAR50]], %[[VAR28]] : f32
// CHECK: %[[VAR52:.*]] = arith.andi %[[VAR40]], %[[VAR42]] : i1
// CHECK: %[[VAR53:.*]] = math.copysign %[[CST_0]], %[[VAR31]] : f32
// CHECK: %[[VAR54:.*]] = arith.select %[[VAR52]], %[[VAR53]], %[[VAR31]] : f32
// CHECK: %[[VAR55:.*]] = arith.ori %[[VAR23]], %[[VAR40]] : i1
// CHECK: %[[VAR56:.*]] = arith.cmpf oeq, %[[VAR9]], %[[CST]] : f32
// CHECK: %[[VAR57:.*]] = arith.cmpf oeq, %[[VAR11]], %[[CST]] : f32
// CHECK: %[[VAR58:.*]] = arith.ori %[[VAR56]], %[[VAR57]] : i1
// CHECK: %[[VAR59:.*]] = arith.cmpf oeq, %[[VAR16]], %[[CST]] : f32
// CHECK: %[[VAR60:.*]] = arith.ori %[[VAR58]], %[[VAR59]] : i1
// CHECK: %[[VAR61:.*]] = arith.cmpf oeq, %[[VAR14]], %[[CST]] : f32
// CHECK: %[[VAR62:.*]] = arith.ori %[[VAR60]], %[[VAR61]] : i1
// CHECK: %[[TRUE:.*]] = arith.constant true
// CHECK: %[[VAR63:.*]] = arith.xori %[[VAR55]], %[[TRUE]] : i1
// CHECK: %[[VAR64:.*]] = arith.andi %[[VAR62]], %[[VAR63]] : i1
// CHECK: %[[VAR65:.*]] = arith.andi %[[VAR64]], %[[VAR41]] : i1
// CHECK: %[[VAR66:.*]] = math.copysign %[[CST_0]], %[[VAR51]] : f32
// CHECK: %[[VAR67:.*]] = arith.select %[[VAR65]], %[[VAR66]], %[[VAR51]] : f32
// CHECK: %[[VAR68:.*]] = arith.andi %[[VAR64]], %[[VAR42]] : i1
// CHECK: %[[VAR69:.*]] = math.copysign %[[CST_0]], %[[VAR54]] : f32
// CHECK: %[[VAR70:.*]] = arith.select %[[VAR68]], %[[VAR69]], %[[VAR54]] : f32
// CHECK: %[[VAR71:.*]] = arith.andi %[[VAR64]], %[[VAR24]] : i1
// CHECK: %[[VAR72:.*]] = math.copysign %[[CST_0]], %[[VAR45]] : f32
// CHECK: %[[VAR73:.*]] = arith.select %[[VAR71]], %[[VAR72]], %[[VAR45]] : f32
// CHECK: %[[VAR74:.*]] = arith.andi %[[VAR64]], %[[VAR25]] : i1
// CHECK: %[[VAR75:.*]] = math.copysign %[[CST_0]], %[[VAR48]] : f32
// CHECK: %[[VAR76:.*]] = arith.select %[[VAR74]], %[[VAR75]], %[[VAR48]] : f32
// CHECK: %[[VAR77:.*]] = arith.ori %[[VAR55]], %[[VAR64]] : i1
// CHECK: %[[VAR78:.*]] = arith.andi %[[VAR20]], %[[VAR77]] : i1
// CHECK: %[[VAR79:.*]] = arith.mulf %[[VAR67]], %[[VAR73]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR80:.*]] = arith.mulf %[[VAR70]], %[[VAR76]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR81:.*]] = arith.subf %[[VAR79]], %[[VAR80]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR82:.*]] = arith.mulf %[[CST]], %[[VAR81]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR83:.*]] = arith.select %[[VAR78]], %[[VAR82]], %[[VAR12]] : f32
// CHECK: %[[VAR84:.*]] = arith.mulf %[[VAR70]], %[[VAR73]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR85:.*]] = arith.mulf %[[VAR67]], %[[VAR76]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR86:.*]] = arith.addf %[[VAR84]], %[[VAR85]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR87:.*]] = arith.mulf %[[CST]], %[[VAR86]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR88:.*]] = arith.select %[[VAR78]], %[[VAR87]], %[[VAR17]] : f32
// CHECK: %[[VAR89:.*]] = complex.create %[[VAR83]], %[[VAR88]] : complex<f32>
// CHECK: %[[VAR90:.*]] = complex.re %arg0 : complex<f32>
// CHECK: %[[VAR91:.*]] = math.absf %[[VAR90]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR92:.*]] = complex.im %arg0 : complex<f32>
// CHECK: %[[VAR93:.*]] = math.absf %[[VAR92]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR94:.*]] = complex.re %arg0 : complex<f32>
// CHECK: %[[VAR95:.*]] = math.absf %[[VAR94]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR96:.*]] = complex.im %arg0 : complex<f32>
// CHECK: %[[VAR97:.*]] = math.absf %[[VAR96]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR98:.*]] = arith.mulf %[[VAR90]], %[[VAR94]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR99:.*]] = math.absf %[[VAR98]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR100:.*]] = arith.mulf %[[VAR92]], %[[VAR96]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR101:.*]] = math.absf %[[VAR100]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR102:.*]] = arith.subf %[[VAR98]], %[[VAR100]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR103:.*]] = arith.mulf %[[VAR92]], %[[VAR94]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR104:.*]] = math.absf %[[VAR103]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR105:.*]] = arith.mulf %[[VAR90]], %[[VAR96]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR106:.*]] = math.absf %[[VAR105]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR107:.*]] = arith.addf %[[VAR103]], %[[VAR105]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR108:.*]] = arith.cmpf uno, %[[VAR102]], %[[VAR102]] : f32
// CHECK: %[[VAR109:.*]] = arith.cmpf uno, %[[VAR107]], %[[VAR107]] : f32
// CHECK: %[[VAR110:.*]] = arith.andi %[[VAR108]], %[[VAR109]] : i1
// CHECK: %[[CST_2:.*]] = arith.constant 0x7F800000 : f32
// CHECK: %[[VAR111:.*]] = arith.cmpf oeq, %[[VAR91]], %[[CST_2]] : f32
// CHECK: %[[VAR112:.*]] = arith.cmpf oeq, %[[VAR93]], %[[CST_2]] : f32
// CHECK: %[[VAR113:.*]] = arith.ori %[[VAR111]], %[[VAR112]] : i1
// CHECK: %[[VAR114:.*]] = arith.cmpf uno, %[[VAR94]], %[[VAR94]] : f32
// CHECK: %[[VAR115:.*]] = arith.cmpf uno, %[[VAR96]], %[[VAR96]] : f32
// CHECK: %[[CST_3:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[CST_4:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[VAR116:.*]] = arith.select %[[VAR111]], %[[CST_4]], %[[CST_3]] : f32
// CHECK: %[[VAR117:.*]] = math.copysign %[[VAR116]], %[[VAR90]] : f32
// CHECK: %[[VAR118:.*]] = arith.select %[[VAR113]], %[[VAR117]], %[[VAR90]] : f32
// CHECK: %[[VAR119:.*]] = arith.select %[[VAR112]], %[[CST_4]], %[[CST_3]] : f32
// CHECK: %[[VAR120:.*]] = math.copysign %[[VAR119]], %[[VAR92]] : f32
// CHECK: %[[VAR121:.*]] = arith.select %[[VAR113]], %[[VAR120]], %[[VAR92]] : f32
// CHECK: %[[VAR122:.*]] = arith.andi %[[VAR113]], %[[VAR114]] : i1
// CHECK: %[[VAR123:.*]] = math.copysign %[[CST_3]], %[[VAR94]] : f32
// CHECK: %[[VAR124:.*]] = arith.select %[[VAR122]], %[[VAR123]], %[[VAR94]] : f32
// CHECK: %[[VAR125:.*]] = arith.andi %[[VAR113]], %[[VAR115]] : i1
// CHECK: %[[VAR126:.*]] = math.copysign %[[CST_3]], %[[VAR96]] : f32
// CHECK: %[[VAR127:.*]] = arith.select %[[VAR125]], %[[VAR126]], %[[VAR96]] : f32
// CHECK: %[[VAR128:.*]] = arith.cmpf oeq, %[[VAR95]], %[[CST_2]] : f32
// CHECK: %[[VAR129:.*]] = arith.cmpf oeq, %[[VAR97]], %[[CST_2]] : f32
// CHECK: %[[VAR130:.*]] = arith.ori %[[VAR128]], %[[VAR129]] : i1
// CHECK: %[[VAR131:.*]] = arith.cmpf uno, %[[VAR118]], %[[VAR118]] : f32
// CHECK: %[[VAR132:.*]] = arith.cmpf uno, %[[VAR121]], %[[VAR121]] : f32
// CHECK: %[[VAR133:.*]] = arith.select %[[VAR128]], %[[CST_4]], %[[CST_3]] : f32
// CHECK: %[[VAR134:.*]] = math.copysign %[[VAR133]], %[[VAR124]] : f32
// CHECK: %[[VAR135:.*]] = arith.select %[[VAR130]], %[[VAR134]], %[[VAR124]] : f32
// CHECK: %[[VAR136:.*]] = arith.select %[[VAR129]], %[[CST_4]], %[[CST_3]] : f32
// CHECK: %[[VAR137:.*]] = math.copysign %[[VAR136]], %[[VAR127]] : f32
// CHECK: %[[VAR138:.*]] = arith.select %[[VAR130]], %[[VAR137]], %[[VAR127]] : f32
// CHECK: %[[VAR139:.*]] = arith.andi %[[VAR130]], %[[VAR131]] : i1
// CHECK: %[[VAR140:.*]] = math.copysign %[[CST_3]], %[[VAR118]] : f32
// CHECK: %[[VAR141:.*]] = arith.select %[[VAR139]], %[[VAR140]], %[[VAR118]] : f32
// CHECK: %[[VAR142:.*]] = arith.andi %[[VAR130]], %[[VAR132]] : i1
// CHECK: %[[VAR143:.*]] = math.copysign %[[CST_3]], %[[VAR121]] : f32
// CHECK: %[[VAR144:.*]] = arith.select %[[VAR142]], %[[VAR143]], %[[VAR121]] : f32
// CHECK: %[[VAR145:.*]] = arith.ori %[[VAR113]], %[[VAR130]] : i1
// CHECK: %[[VAR146:.*]] = arith.cmpf oeq, %[[VAR99]], %[[CST_2]] : f32
// CHECK: %[[VAR147:.*]] = arith.cmpf oeq, %[[VAR101]], %[[CST_2]] : f32
// CHECK: %[[VAR148:.*]] = arith.ori %[[VAR146]], %[[VAR147]] : i1
// CHECK: %[[VAR149:.*]] = arith.cmpf oeq, %[[VAR106]], %[[CST_2]] : f32
// CHECK: %[[VAR150:.*]] = arith.ori %[[VAR148]], %[[VAR149]] : i1
// CHECK: %[[VAR151:.*]] = arith.cmpf oeq, %[[VAR104]], %[[CST_2]] : f32
// CHECK: %[[VAR152:.*]] = arith.ori %[[VAR150]], %[[VAR151]] : i1
// CHECK: %[[TRUE_5:.*]] = arith.constant true
// CHECK: %[[VAR153:.*]] = arith.xori %[[VAR145]], %[[TRUE_5]] : i1
// CHECK: %[[VAR154:.*]] = arith.andi %[[VAR152]], %[[VAR153]] : i1
// CHECK: %[[VAR155:.*]] = arith.andi %[[VAR154]], %[[VAR131]] : i1
// CHECK: %[[VAR156:.*]] = math.copysign %[[CST_3]], %[[VAR141]] : f32
// CHECK: %[[VAR157:.*]] = arith.select %[[VAR155]], %[[VAR156]], %[[VAR141]] : f32
// CHECK: %[[VAR158:.*]] = arith.andi %[[VAR154]], %[[VAR132]] : i1
// CHECK: %[[VAR159:.*]] = math.copysign %[[CST_3]], %[[VAR144]] : f32
// CHECK: %[[VAR160:.*]] = arith.select %[[VAR158]], %[[VAR159]], %[[VAR144]] : f32
// CHECK: %[[VAR161:.*]] = arith.andi %[[VAR154]], %[[VAR114]] : i1
// CHECK: %[[VAR162:.*]] = math.copysign %[[CST_3]], %[[VAR135]] : f32
// CHECK: %[[VAR163:.*]] = arith.select %[[VAR161]], %[[VAR162]], %[[VAR135]] : f32
// CHECK: %[[VAR164:.*]] = arith.andi %[[VAR154]], %[[VAR115]] : i1
// CHECK: %[[VAR165:.*]] = math.copysign %[[CST_3]], %[[VAR138]] : f32
// CHECK: %[[VAR166:.*]] = arith.select %[[VAR164]], %[[VAR165]], %[[VAR138]] : f32
// CHECK: %[[VAR167:.*]] = arith.ori %[[VAR145]], %[[VAR154]] : i1
// CHECK: %[[VAR168:.*]] = arith.andi %[[VAR110]], %[[VAR167]] : i1
// CHECK: %[[VAR169:.*]] = arith.mulf %[[VAR157]], %[[VAR163]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR170:.*]] = arith.mulf %[[VAR160]], %[[VAR166]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR171:.*]] = arith.subf %[[VAR169]], %[[VAR170]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR172:.*]] = arith.mulf %[[CST_2]], %[[VAR171]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR173:.*]] = arith.select %[[VAR168]], %[[VAR172]], %[[VAR102]] : f32
// CHECK: %[[VAR174:.*]] = arith.mulf %[[VAR160]], %[[VAR163]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR175:.*]] = arith.mulf %[[VAR157]], %[[VAR166]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR176:.*]] = arith.addf %[[VAR174]], %[[VAR175]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR177:.*]] = arith.mulf %[[CST_2]], %[[VAR176]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR178:.*]] = arith.select %[[VAR168]], %[[VAR177]], %[[VAR107]] : f32
// CHECK: %[[VAR179:.*]] = complex.create %[[VAR173]], %[[VAR178]] : complex<f32>
// CHECK: %[[VAR180:.*]] = complex.re %[[VAR89]] : complex<f32>
// CHECK: %[[VAR181:.*]] = complex.re %[[VAR179]] : complex<f32>
// CHECK: %[[VAR182:.*]] = arith.addf %[[VAR180]], %[[VAR181]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR183:.*]] = complex.im %[[VAR89]] : complex<f32>
// CHECK: %[[VAR184:.*]] = complex.im %[[VAR179]] : complex<f32>
// CHECK: %[[VAR185:.*]] = arith.addf %[[VAR183]], %[[VAR184]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR186:.*]] = complex.create %[[VAR182]], %[[VAR185]] : complex<f32>
// CHECK: %[[CST_6:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[VAR187:.*]] = complex.re %[[VAR186]] : complex<f32>
// CHECK: %[[VAR188:.*]] = complex.im %[[VAR186]] : complex<f32>
// CHECK: %[[VAR189:.*]] = math.absf %[[VAR187]] : f32
// CHECK: %[[CST_7:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[CST_8:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[VAR190:.*]] = complex.re %[[VAR186]] : complex<f32>
// CHECK: %[[VAR191:.*]] = complex.im %[[VAR186]] : complex<f32>
// CHECK: %[[VAR192:.*]] = arith.cmpf oeq, %[[VAR190]], %[[CST_7]] : f32
// CHECK: %[[VAR193:.*]] = arith.cmpf oeq, %[[VAR191]], %[[CST_7]] : f32
// CHECK: %[[VAR194:.*]] = arith.divf %[[VAR191]], %[[VAR190]] : f32
// CHECK: %[[VAR195:.*]] = arith.mulf %[[VAR194]], %[[VAR194]] : f32
// CHECK: %[[VAR196:.*]] = arith.addf %[[VAR195]], %[[CST_8]] : f32
// CHECK: %[[VAR197:.*]] = math.sqrt %[[VAR196]] : f32
// CHECK: %[[VAR198:.*]] = math.absf %[[VAR190]] : f32
// CHECK: %[[VAR199:.*]] = arith.mulf %[[VAR197]], %[[VAR198]] : f32
// CHECK: %[[VAR200:.*]] = arith.divf %[[VAR190]], %[[VAR191]] : f32
// CHECK: %[[VAR201:.*]] = arith.mulf %[[VAR200]], %[[VAR200]] : f32
// CHECK: %[[VAR202:.*]] = arith.addf %[[VAR201]], %[[CST_8]] : f32
// CHECK: %[[VAR203:.*]] = math.sqrt %[[VAR202]] : f32
// CHECK: %[[VAR204:.*]] = math.absf %[[VAR191]] : f32
// CHECK: %[[VAR205:.*]] = arith.mulf %[[VAR203]], %[[VAR204]] : f32
// CHECK: %[[VAR206:.*]] = arith.cmpf ogt, %[[VAR190]], %[[VAR191]] : f32
// CHECK: %[[VAR207:.*]] = arith.select %[[VAR206]], %[[VAR199]], %[[VAR205]] : f32
// CHECK: %[[VAR208:.*]] = arith.select %[[VAR193]], %[[VAR198]], %[[VAR207]] : f32
// CHECK: %[[VAR209:.*]] = arith.select %[[VAR192]], %[[VAR204]], %[[VAR208]] : f32
// CHECK: %[[VAR210:.*]] = arith.addf %[[VAR189]], %[[VAR209]] : f32
// CHECK: %[[CST_9:.*]] = arith.constant 5.000000e-01 : f32
// CHECK: %[[VAR211:.*]] = arith.mulf %[[VAR210]], %[[CST_9]] : f32
// CHECK: %[[VAR212:.*]] = math.sqrt %[[VAR211]] : f32
// CHECK: %[[VAR213:.*]] = arith.cmpf olt, %[[VAR187]], %[[CST_6]] : f32
// CHECK: %[[VAR214:.*]] = arith.cmpf olt, %[[VAR188]], %[[CST_6]] : f32
// CHECK: %[[VAR215:.*]] = arith.addf %[[VAR212]], %[[VAR212]] : f32
// CHECK: %[[VAR216:.*]] = arith.divf %[[VAR188]], %[[VAR215]] : f32
// CHECK: %[[VAR217:.*]] = arith.negf %[[VAR212]] : f32
// CHECK: %[[VAR218:.*]] = arith.select %[[VAR214]], %[[VAR217]], %[[VAR212]] : f32
// CHECK: %[[VAR219:.*]] = arith.select %[[VAR213]], %[[VAR218]], %[[VAR216]] : f32
// CHECK: %[[VAR220:.*]] = arith.addf %[[VAR219]], %[[VAR219]] : f32
// CHECK: %[[VAR221:.*]] = arith.divf %[[VAR188]], %[[VAR220]] : f32
// CHECK: %[[VAR222:.*]] = arith.select %[[VAR213]], %[[VAR221]], %[[VAR212]] : f32
// CHECK: %[[VAR223:.*]] = arith.cmpf oeq, %[[VAR187]], %[[CST_6]] : f32
// CHECK: %[[VAR224:.*]] = arith.cmpf oeq, %[[VAR188]], %[[CST_6]] : f32
// CHECK: %[[VAR225:.*]] = arith.andi %[[VAR223]], %[[VAR224]] : i1
// CHECK: %[[VAR226:.*]] = arith.select %[[VAR225]], %[[CST_6]], %[[VAR222]] : f32
// CHECK: %[[VAR227:.*]] = arith.select %[[VAR225]], %[[CST_6]], %[[VAR219]] : f32
// CHECK: %[[VAR228:.*]] = complex.create %[[VAR226]], %[[VAR227]] : complex<f32>
// CHECK: %[[CST_10:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[CST_11:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[VAR229:.*]] = complex.create %[[CST_10]], %[[CST_11]] : complex<f32>
// CHECK: %[[VAR230:.*]] = complex.re %[[VAR229]] : complex<f32>
// CHECK: %[[VAR231:.*]] = math.absf %[[VAR230]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR232:.*]] = complex.im %[[VAR229]] : complex<f32>
// CHECK: %[[VAR233:.*]] = math.absf %[[VAR232]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR234:.*]] = complex.re %arg0 : complex<f32>
// CHECK: %[[VAR235:.*]] = math.absf %[[VAR234]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR236:.*]] = complex.im %arg0 : complex<f32>
// CHECK: %[[VAR237:.*]] = math.absf %[[VAR236]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR238:.*]] = arith.mulf %[[VAR230]], %[[VAR234]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR239:.*]] = math.absf %[[VAR238]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR240:.*]] = arith.mulf %[[VAR232]], %[[VAR236]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR241:.*]] = math.absf %[[VAR240]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR242:.*]] = arith.subf %[[VAR238]], %[[VAR240]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR243:.*]] = arith.mulf %[[VAR232]], %[[VAR234]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR244:.*]] = math.absf %[[VAR243]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR245:.*]] = arith.mulf %[[VAR230]], %[[VAR236]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR246:.*]] = math.absf %[[VAR245]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR247:.*]] = arith.addf %[[VAR243]], %[[VAR245]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR248:.*]] = arith.cmpf uno, %[[VAR242]], %[[VAR242]] : f32
// CHECK: %[[VAR249:.*]] = arith.cmpf uno, %[[VAR247]], %[[VAR247]] : f32
// CHECK: %[[VAR250:.*]] = arith.andi %[[VAR248]], %[[VAR249]] : i1
// CHECK: %[[CST_12:.*]] = arith.constant 0x7F800000 : f32
// CHECK: %[[VAR251:.*]] = arith.cmpf oeq, %[[VAR231]], %[[CST_12]] : f32
// CHECK: %[[VAR252:.*]] = arith.cmpf oeq, %[[VAR233]], %[[CST_12]] : f32
// CHECK: %[[VAR253:.*]] = arith.ori %[[VAR251]], %[[VAR252]] : i1
// CHECK: %[[VAR254:.*]] = arith.cmpf uno, %[[VAR234]], %[[VAR234]] : f32
// CHECK: %[[VAR255:.*]] = arith.cmpf uno, %[[VAR236]], %[[VAR236]] : f32
// CHECK: %[[CST_13:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[CST_14:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[VAR256:.*]] = arith.select %[[VAR251]], %[[CST_14]], %[[CST_13]] : f32
// CHECK: %[[VAR257:.*]] = math.copysign %[[VAR256]], %[[VAR230]] : f32
// CHECK: %[[VAR258:.*]] = arith.select %[[VAR253]], %[[VAR257]], %[[VAR230]] : f32
// CHECK: %[[VAR259:.*]] = arith.select %[[VAR252]], %[[CST_14]], %[[CST_13]] : f32
// CHECK: %[[VAR260:.*]] = math.copysign %[[VAR259]], %[[VAR232]] : f32
// CHECK: %[[VAR261:.*]] = arith.select %[[VAR253]], %[[VAR260]], %[[VAR232]] : f32
// CHECK: %[[VAR262:.*]] = arith.andi %[[VAR253]], %[[VAR254]] : i1
// CHECK: %[[VAR263:.*]] = math.copysign %[[CST_13]], %[[VAR234]] : f32
// CHECK: %[[VAR264:.*]] = arith.select %[[VAR262]], %[[VAR263]], %[[VAR234]] : f32
// CHECK: %[[VAR265:.*]] = arith.andi %[[VAR253]], %[[VAR255]] : i1
// CHECK: %[[VAR266:.*]] = math.copysign %[[CST_13]], %[[VAR236]] : f32
// CHECK: %[[VAR267:.*]] = arith.select %[[VAR265]], %[[VAR266]], %[[VAR236]] : f32
// CHECK: %[[VAR268:.*]] = arith.cmpf oeq, %[[VAR235]], %[[CST_12]] : f32
// CHECK: %[[VAR269:.*]] = arith.cmpf oeq, %[[VAR237]], %[[CST_12]] : f32
// CHECK: %[[VAR270:.*]] = arith.ori %[[VAR268]], %[[VAR269]] : i1
// CHECK: %[[VAR271:.*]] = arith.cmpf uno, %[[VAR258]], %[[VAR258]] : f32
// CHECK: %[[VAR272:.*]] = arith.cmpf uno, %[[VAR261]], %[[VAR261]] : f32
// CHECK: %[[VAR273:.*]] = arith.select %[[VAR268]], %[[CST_14]], %[[CST_13]] : f32
// CHECK: %[[VAR274:.*]] = math.copysign %[[VAR273]], %[[VAR264]] : f32
// CHECK: %[[VAR275:.*]] = arith.select %[[VAR270]], %[[VAR274]], %[[VAR264]] : f32
// CHECK: %[[VAR276:.*]] = arith.select %[[VAR269]], %[[CST_14]], %[[CST_13]] : f32
// CHECK: %[[VAR277:.*]] = math.copysign %[[VAR276]], %[[VAR267]] : f32
// CHECK: %[[VAR278:.*]] = arith.select %[[VAR270]], %[[VAR277]], %[[VAR267]] : f32
// CHECK: %[[VAR279:.*]] = arith.andi %[[VAR270]], %[[VAR271]] : i1
// CHECK: %[[VAR280:.*]] = math.copysign %[[CST_13]], %[[VAR258]] : f32
// CHECK: %[[VAR281:.*]] = arith.select %[[VAR279]], %[[VAR280]], %[[VAR258]] : f32
// CHECK: %[[VAR282:.*]] = arith.andi %[[VAR270]], %[[VAR272]] : i1
// CHECK: %[[VAR283:.*]] = math.copysign %[[CST_13]], %[[VAR261]] : f32
// CHECK: %[[VAR284:.*]] = arith.select %[[VAR282]], %[[VAR283]], %[[VAR261]] : f32
// CHECK: %[[VAR285:.*]] = arith.ori %[[VAR253]], %[[VAR270]] : i1
// CHECK: %[[VAR286:.*]] = arith.cmpf oeq, %[[VAR239]], %[[CST_12]] : f32
// CHECK: %[[VAR287:.*]] = arith.cmpf oeq, %[[VAR241]], %[[CST_12]] : f32
// CHECK: %[[VAR288:.*]] = arith.ori %[[VAR286]], %[[VAR287]] : i1
// CHECK: %[[VAR289:.*]] = arith.cmpf oeq, %[[VAR246]], %[[CST_12]] : f32
// CHECK: %[[VAR290:.*]] = arith.ori %[[VAR288]], %[[VAR289]] : i1
// CHECK: %[[VAR291:.*]] = arith.cmpf oeq, %[[VAR244]], %[[CST_12]] : f32
// CHECK: %[[VAR292:.*]] = arith.ori %[[VAR290]], %[[VAR291]] : i1
// CHECK: %[[TRUE_15:.*]] = arith.constant true
// CHECK: %[[VAR293:.*]] = arith.xori %[[VAR285]], %[[TRUE_15]] : i1
// CHECK: %[[VAR294:.*]] = arith.andi %[[VAR292]], %[[VAR293]] : i1
// CHECK: %[[VAR295:.*]] = arith.andi %[[VAR294]], %[[VAR271]] : i1
// CHECK: %[[VAR296:.*]] = math.copysign %[[CST_13]], %[[VAR281]] : f32
// CHECK: %[[VAR297:.*]] = arith.select %[[VAR295]], %[[VAR296]], %[[VAR281]] : f32
// CHECK: %[[VAR298:.*]] = arith.andi %[[VAR294]], %[[VAR272]] : i1
// CHECK: %[[VAR299:.*]] = math.copysign %[[CST_13]], %[[VAR284]] : f32
// CHECK: %[[VAR300:.*]] = arith.select %[[VAR298]], %[[VAR299]], %[[VAR284]] : f32
// CHECK: %[[VAR301:.*]] = arith.andi %[[VAR294]], %[[VAR254]] : i1
// CHECK: %[[VAR302:.*]] = math.copysign %[[CST_13]], %[[VAR275]] : f32
// CHECK: %[[VAR303:.*]] = arith.select %[[VAR301]], %[[VAR302]], %[[VAR275]] : f32
// CHECK: %[[VAR304:.*]] = arith.andi %[[VAR294]], %[[VAR255]] : i1
// CHECK: %[[VAR305:.*]] = math.copysign %[[CST_13]], %[[VAR278]] : f32
// CHECK: %[[VAR306:.*]] = arith.select %[[VAR304]], %[[VAR305]], %[[VAR278]] : f32
// CHECK: %[[VAR307:.*]] = arith.ori %[[VAR285]], %[[VAR294]] : i1
// CHECK: %[[VAR308:.*]] = arith.andi %[[VAR250]], %[[VAR307]] : i1
// CHECK: %[[VAR309:.*]] = arith.mulf %[[VAR297]], %[[VAR303]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR310:.*]] = arith.mulf %[[VAR300]], %[[VAR306]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR311:.*]] = arith.subf %[[VAR309]], %[[VAR310]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR312:.*]] = arith.mulf %[[CST_12]], %[[VAR311]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR313:.*]] = arith.select %[[VAR308]], %[[VAR312]], %[[VAR242]] : f32
// CHECK: %[[VAR314:.*]] = arith.mulf %[[VAR300]], %[[VAR303]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR315:.*]] = arith.mulf %[[VAR297]], %[[VAR306]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR316:.*]] = arith.addf %[[VAR314]], %[[VAR315]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR317:.*]] = arith.mulf %[[CST_12]], %[[VAR316]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR318:.*]] = arith.select %[[VAR308]], %[[VAR317]], %[[VAR247]] : f32
// CHECK: %[[VAR319:.*]] = complex.create %[[VAR313]], %[[VAR318]] : complex<f32>
// CHECK: %[[VAR320:.*]] = complex.re %arg1 : complex<f32>
// CHECK: %[[VAR321:.*]] = complex.re %[[VAR319]] : complex<f32>
// CHECK: %[[VAR322:.*]] = arith.addf %[[VAR320]], %[[VAR321]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR323:.*]] = complex.im %arg1 : complex<f32>
// CHECK: %[[VAR324:.*]] = complex.im %[[VAR319]] : complex<f32>
// CHECK: %[[VAR325:.*]] = arith.addf %[[VAR323]], %[[VAR324]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR326:.*]] = complex.create %[[VAR322]], %[[VAR325]] : complex<f32>
// CHECK: %[[VAR327:.*]] = complex.re %[[VAR326]] : complex<f32>
// CHECK: %[[VAR328:.*]] = complex.im %[[VAR326]] : complex<f32>
// CHECK: %[[VAR329:.*]] = complex.re %[[VAR228]] : complex<f32>
// CHECK: %[[VAR330:.*]] = complex.im %[[VAR228]] : complex<f32>
// CHECK: %[[VAR331:.*]] = arith.divf %[[VAR329]], %[[VAR330]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR332:.*]] = arith.mulf %[[VAR331]], %[[VAR329]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR333:.*]] = arith.addf %[[VAR330]], %[[VAR332]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR334:.*]] = arith.mulf %[[VAR327]], %[[VAR331]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR335:.*]] = arith.addf %[[VAR334]], %[[VAR328]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR336:.*]] = arith.divf %[[VAR335]], %[[VAR333]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR337:.*]] = arith.mulf %[[VAR328]], %[[VAR331]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR338:.*]] = arith.subf %[[VAR337]], %[[VAR327]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR339:.*]] = arith.divf %[[VAR338]], %[[VAR333]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR340:.*]] = arith.divf %[[VAR330]], %[[VAR329]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR341:.*]] = arith.mulf %[[VAR340]], %[[VAR330]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR342:.*]] = arith.addf %[[VAR329]], %[[VAR341]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR343:.*]] = arith.mulf %[[VAR328]], %[[VAR340]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR344:.*]] = arith.addf %[[VAR327]], %[[VAR343]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR345:.*]] = arith.divf %[[VAR344]], %[[VAR342]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR346:.*]] = arith.mulf %[[VAR327]], %[[VAR340]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR347:.*]] = arith.subf %[[VAR328]], %[[VAR346]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR348:.*]] = arith.divf %[[VAR347]], %[[VAR342]] fastmath<nnan,contract> : f32
// CHECK: %[[CST_16:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[VAR349:.*]] = math.absf %[[VAR329]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR350:.*]] = arith.cmpf oeq, %[[VAR349]], %[[CST_16]] : f32
// CHECK: %[[VAR351:.*]] = math.absf %[[VAR330]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR352:.*]] = arith.cmpf oeq, %[[VAR351]], %[[CST_16]] : f32
// CHECK: %[[VAR353:.*]] = arith.cmpf ord, %[[VAR327]], %[[CST_16]] : f32
// CHECK: %[[VAR354:.*]] = arith.cmpf ord, %[[VAR328]], %[[CST_16]] : f32
// CHECK: %[[VAR355:.*]] = arith.ori %[[VAR353]], %[[VAR354]] : i1
// CHECK: %[[VAR356:.*]] = arith.andi %[[VAR350]], %[[VAR352]] : i1
// CHECK: %[[VAR357:.*]] = arith.andi %[[VAR355]], %[[VAR356]] : i1
// CHECK: %[[CST_17:.*]] = arith.constant 0x7F800000 : f32
// CHECK: %[[VAR358:.*]] = math.copysign %[[CST_17]], %[[VAR329]] : f32
// CHECK: %[[VAR359:.*]] = arith.mulf %[[VAR358]], %[[VAR327]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR360:.*]] = arith.mulf %[[VAR358]], %[[VAR328]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR361:.*]] = arith.cmpf one, %[[VAR349]], %[[CST_17]] : f32
// CHECK: %[[VAR362:.*]] = arith.cmpf one, %[[VAR351]], %[[CST_17]] : f32
// CHECK: %[[VAR363:.*]] = arith.andi %[[VAR361]], %[[VAR362]] : i1
// CHECK: %[[VAR364:.*]] = math.absf %[[VAR327]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR365:.*]] = arith.cmpf oeq, %[[VAR364]], %[[CST_17]] : f32
// CHECK: %[[VAR366:.*]] = math.absf %[[VAR328]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR367:.*]] = arith.cmpf oeq, %[[VAR366]], %[[CST_17]] : f32
// CHECK: %[[VAR368:.*]] = arith.ori %[[VAR365]], %[[VAR367]] : i1
// CHECK: %[[VAR369:.*]] = arith.andi %[[VAR368]], %[[VAR363]] : i1
// CHECK: %[[CST_18:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[VAR370:.*]] = arith.select %[[VAR365]], %[[CST_18]], %[[CST_16]] : f32
// CHECK: %[[VAR371:.*]] = math.copysign %[[VAR370]], %[[VAR327]] : f32
// CHECK: %[[VAR372:.*]] = arith.select %[[VAR367]], %[[CST_18]], %[[CST_16]] : f32
// CHECK: %[[VAR373:.*]] = math.copysign %[[VAR372]], %[[VAR328]] : f32
// CHECK: %[[VAR374:.*]] = arith.mulf %[[VAR371]], %[[VAR329]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR375:.*]] = arith.mulf %[[VAR373]], %[[VAR330]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR376:.*]] = arith.addf %[[VAR374]], %[[VAR375]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR377:.*]] = arith.mulf %[[CST_17]], %[[VAR376]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR378:.*]] = arith.mulf %[[VAR371]], %[[VAR330]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR379:.*]] = arith.mulf %[[VAR373]], %[[VAR329]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR380:.*]] = arith.subf %[[VAR379]], %[[VAR378]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR381:.*]] = arith.mulf %[[CST_17]], %[[VAR380]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR382:.*]] = arith.cmpf one, %[[VAR364]], %[[CST_17]] : f32
// CHECK: %[[VAR383:.*]] = arith.cmpf one, %[[VAR366]], %[[CST_17]] : f32
// CHECK: %[[VAR384:.*]] = arith.andi %[[VAR382]], %[[VAR383]] : i1
// CHECK: %[[VAR385:.*]] = arith.cmpf oeq, %[[VAR349]], %[[CST_17]] : f32
// CHECK: %[[VAR386:.*]] = arith.cmpf oeq, %[[VAR351]], %[[CST_17]] : f32
// CHECK: %[[VAR387:.*]] = arith.ori %[[VAR385]], %[[VAR386]] : i1
// CHECK: %[[VAR388:.*]] = arith.andi %[[VAR384]], %[[VAR387]] : i1
// CHECK: %[[VAR389:.*]] = arith.select %[[VAR385]], %[[CST_18]], %[[CST_16]] : f32
// CHECK: %[[VAR390:.*]] = math.copysign %[[VAR389]], %[[VAR329]] : f32
// CHECK: %[[VAR391:.*]] = arith.select %[[VAR386]], %[[CST_18]], %[[CST_16]] : f32
// CHECK: %[[VAR392:.*]] = math.copysign %[[VAR391]], %[[VAR330]] : f32
// CHECK: %[[VAR393:.*]] = arith.mulf %[[VAR327]], %[[VAR390]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR394:.*]] = arith.mulf %[[VAR328]], %[[VAR392]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR395:.*]] = arith.addf %[[VAR393]], %[[VAR394]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR396:.*]] = arith.mulf %[[CST_16]], %[[VAR395]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR397:.*]] = arith.mulf %[[VAR328]], %[[VAR390]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR398:.*]] = arith.mulf %[[VAR327]], %[[VAR392]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR399:.*]] = arith.subf %[[VAR397]], %[[VAR398]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR400:.*]] = arith.mulf %[[CST_16]], %[[VAR399]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR401:.*]] = arith.cmpf olt, %[[VAR349]], %[[VAR351]] : f32
// CHECK: %[[VAR402:.*]] = arith.select %[[VAR401]], %[[VAR336]], %[[VAR345]] : f32
// CHECK: %[[VAR403:.*]] = arith.select %[[VAR401]], %[[VAR339]], %[[VAR348]] : f32
// CHECK: %[[VAR404:.*]] = arith.select %[[VAR388]], %[[VAR396]], %[[VAR402]] : f32
// CHECK: %[[VAR405:.*]] = arith.select %[[VAR388]], %[[VAR400]], %[[VAR403]] : f32
// CHECK: %[[VAR406:.*]] = arith.select %[[VAR369]], %[[VAR377]], %[[VAR404]] : f32
// CHECK: %[[VAR407:.*]] = arith.select %[[VAR369]], %[[VAR381]], %[[VAR405]] : f32
// CHECK: %[[VAR408:.*]] = arith.select %[[VAR357]], %[[VAR359]], %[[VAR406]] : f32
// CHECK: %[[VAR409:.*]] = arith.select %[[VAR357]], %[[VAR360]], %[[VAR407]] : f32
// CHECK: %[[VAR410:.*]] = arith.cmpf uno, %[[VAR402]], %[[CST_16]] : f32
// CHECK: %[[VAR411:.*]] = arith.cmpf uno, %[[VAR403]], %[[CST_16]] : f32
// CHECK: %[[VAR412:.*]] = arith.andi %[[VAR410]], %[[VAR411]] : i1
// CHECK: %[[VAR413:.*]] = arith.select %[[VAR412]], %[[VAR408]], %[[VAR402]] : f32
// CHECK: %[[VAR414:.*]] = arith.select %[[VAR412]], %[[VAR409]], %[[VAR403]] : f32
// CHECK: %[[VAR415:.*]] = complex.create %[[VAR413]], %[[VAR414]] : complex<f32>
// CHECK: %[[CST_19:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[CST_20:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[VAR416:.*]] = complex.re %[[VAR415]] : complex<f32>
// CHECK: %[[VAR417:.*]] = complex.im %[[VAR415]] : complex<f32>
// CHECK: %[[VAR418:.*]] = arith.cmpf oeq, %[[VAR416]], %[[CST_19]] : f32
// CHECK: %[[VAR419:.*]] = arith.cmpf oeq, %[[VAR417]], %[[CST_19]] : f32
// CHECK: %[[VAR420:.*]] = arith.divf %[[VAR417]], %[[VAR416]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR421:.*]] = arith.mulf %[[VAR420]], %[[VAR420]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR422:.*]] = arith.addf %[[VAR421]], %[[CST_20]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR423:.*]] = math.sqrt %[[VAR422]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR424:.*]] = math.absf %[[VAR416]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR425:.*]] = arith.mulf %[[VAR423]], %[[VAR424]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR426:.*]] = arith.divf %[[VAR416]], %[[VAR417]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR427:.*]] = arith.mulf %[[VAR426]], %[[VAR426]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR428:.*]] = arith.addf %[[VAR427]], %[[CST_20]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR429:.*]] = math.sqrt %[[VAR428]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR430:.*]] = math.absf %[[VAR417]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR431:.*]] = arith.mulf %[[VAR429]], %[[VAR430]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR432:.*]] = arith.cmpf ogt, %[[VAR416]], %[[VAR417]] : f32
// CHECK: %[[VAR433:.*]] = arith.select %[[VAR432]], %[[VAR425]], %[[VAR431]] : f32
// CHECK: %[[VAR434:.*]] = arith.select %[[VAR419]], %[[VAR424]], %[[VAR433]] : f32
// CHECK: %[[VAR435:.*]] = arith.select %[[VAR418]], %[[VAR430]], %[[VAR434]] : f32
// CHECK: %[[VAR436:.*]] = math.log %[[VAR435]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR437:.*]] = complex.re %[[VAR415]] : complex<f32>
// CHECK: %[[VAR438:.*]] = complex.im %[[VAR415]] : complex<f32>
// CHECK: %[[VAR439:.*]] = math.atan2 %[[VAR438]], %[[VAR437]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR440:.*]] = complex.create %[[VAR436]], %[[VAR439]] : complex<f32>
// CHECK: %[[CST_21:.*]] = arith.constant -1.000000e+00 : f32
// CHECK: %[[VAR441:.*]] = complex.create %[[CST_10]], %[[CST_21]] : complex<f32>
// CHECK: %[[VAR442:.*]] = complex.re %[[VAR441]] : complex<f32>
// CHECK: %[[VAR443:.*]] = math.absf %[[VAR442]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR444:.*]] = complex.im %[[VAR441]] : complex<f32>
// CHECK: %[[VAR445:.*]] = math.absf %[[VAR444]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR446:.*]] = complex.re %[[VAR440]] : complex<f32>
// CHECK: %[[VAR447:.*]] = math.absf %[[VAR446]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR448:.*]] = complex.im %[[VAR440]] : complex<f32>
// CHECK: %[[VAR449:.*]] = math.absf %[[VAR448]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR450:.*]] = arith.mulf %[[VAR442]], %[[VAR446]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR451:.*]] = math.absf %[[VAR450]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR452:.*]] = arith.mulf %[[VAR444]], %[[VAR448]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR453:.*]] = math.absf %[[VAR452]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR454:.*]] = arith.subf %[[VAR450]], %[[VAR452]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR455:.*]] = arith.mulf %[[VAR444]], %[[VAR446]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR456:.*]] = math.absf %[[VAR455]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR457:.*]] = arith.mulf %[[VAR442]], %[[VAR448]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR458:.*]] = math.absf %[[VAR457]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR459:.*]] = arith.addf %[[VAR455]], %[[VAR457]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR460:.*]] = arith.cmpf uno, %[[VAR454]], %[[VAR454]] : f32
// CHECK: %[[VAR461:.*]] = arith.cmpf uno, %[[VAR459]], %[[VAR459]] : f32
// CHECK: %[[VAR462:.*]] = arith.andi %[[VAR460]], %[[VAR461]] : i1
// CHECK: %[[CST_22:.*]] = arith.constant 0x7F800000 : f32
// CHECK: %[[VAR463:.*]] = arith.cmpf oeq, %[[VAR443]], %[[CST_22]] : f32
// CHECK: %[[VAR464:.*]] = arith.cmpf oeq, %[[VAR445]], %[[CST_22]] : f32
// CHECK: %[[VAR465:.*]] = arith.ori %[[VAR463]], %[[VAR464]] : i1
// CHECK: %[[VAR466:.*]] = arith.cmpf uno, %[[VAR446]], %[[VAR446]] : f32
// CHECK: %[[VAR467:.*]] = arith.cmpf uno, %[[VAR448]], %[[VAR448]] : f32
// CHECK: %[[CST_23:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[CST_24:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[VAR468:.*]] = arith.select %[[VAR463]], %[[CST_24]], %[[CST_23]] : f32
// CHECK: %[[VAR469:.*]] = math.copysign %[[VAR468]], %[[VAR442]] : f32
// CHECK: %[[VAR470:.*]] = arith.select %[[VAR465]], %[[VAR469]], %[[VAR442]] : f32
// CHECK: %[[VAR471:.*]] = arith.select %[[VAR464]], %[[CST_24]], %[[CST_23]] : f32
// CHECK: %[[VAR472:.*]] = math.copysign %[[VAR471]], %[[VAR444]] : f32
// CHECK: %[[VAR473:.*]] = arith.select %[[VAR465]], %[[VAR472]], %[[VAR444]] : f32
// CHECK: %[[VAR474:.*]] = arith.andi %[[VAR465]], %[[VAR466]] : i1
// CHECK: %[[VAR475:.*]] = math.copysign %[[CST_23]], %[[VAR446]] : f32
// CHECK: %[[VAR476:.*]] = arith.select %[[VAR474]], %[[VAR475]], %[[VAR446]] : f32
// CHECK: %[[VAR477:.*]] = arith.andi %[[VAR465]], %[[VAR467]] : i1
// CHECK: %[[VAR478:.*]] = math.copysign %[[CST_23]], %[[VAR448]] : f32
// CHECK: %[[VAR479:.*]] = arith.select %[[VAR477]], %[[VAR478]], %[[VAR448]] : f32
// CHECK: %[[VAR480:.*]] = arith.cmpf oeq, %[[VAR447]], %[[CST_22]] : f32
// CHECK: %[[VAR481:.*]] = arith.cmpf oeq, %[[VAR449]], %[[CST_22]] : f32
// CHECK: %[[VAR482:.*]] = arith.ori %[[VAR480]], %[[VAR481]] : i1
// CHECK: %[[VAR483:.*]] = arith.cmpf uno, %[[VAR470]], %[[VAR470]] : f32
// CHECK: %[[VAR484:.*]] = arith.cmpf uno, %[[VAR473]], %[[VAR473]] : f32
// CHECK: %[[VAR485:.*]] = arith.select %[[VAR480]], %[[CST_24]], %[[CST_23]] : f32
// CHECK: %[[VAR486:.*]] = math.copysign %[[VAR485]], %[[VAR476]] : f32
// CHECK: %[[VAR487:.*]] = arith.select %[[VAR482]], %[[VAR486]], %[[VAR476]] : f32
// CHECK: %[[VAR488:.*]] = arith.select %[[VAR481]], %[[CST_24]], %[[CST_23]] : f32
// CHECK: %[[VAR489:.*]] = math.copysign %[[VAR488]], %[[VAR479]] : f32
// CHECK: %[[VAR490:.*]] = arith.select %[[VAR482]], %[[VAR489]], %[[VAR479]] : f32
// CHECK: %[[VAR491:.*]] = arith.andi %[[VAR482]], %[[VAR483]] : i1
// CHECK: %[[VAR492:.*]] = math.copysign %[[CST_23]], %[[VAR470]] : f32
// CHECK: %[[VAR493:.*]] = arith.select %[[VAR491]], %[[VAR492]], %[[VAR470]] : f32
// CHECK: %[[VAR494:.*]] = arith.andi %[[VAR482]], %[[VAR484]] : i1
// CHECK: %[[VAR495:.*]] = math.copysign %[[CST_23]], %[[VAR473]] : f32
// CHECK: %[[VAR496:.*]] = arith.select %[[VAR494]], %[[VAR495]], %[[VAR473]] : f32
// CHECK: %[[VAR497:.*]] = arith.ori %[[VAR465]], %[[VAR482]] : i1
// CHECK: %[[VAR498:.*]] = arith.cmpf oeq, %[[VAR451]], %[[CST_22]] : f32
// CHECK: %[[VAR499:.*]] = arith.cmpf oeq, %[[VAR453]], %[[CST_22]] : f32
// CHECK: %[[VAR500:.*]] = arith.ori %[[VAR498]], %[[VAR499]] : i1
// CHECK: %[[VAR501:.*]] = arith.cmpf oeq, %[[VAR458]], %[[CST_22]] : f32
// CHECK: %[[VAR502:.*]] = arith.ori %[[VAR500]], %[[VAR501]] : i1
// CHECK: %[[VAR503:.*]] = arith.cmpf oeq, %[[VAR456]], %[[CST_22]] : f32
// CHECK: %[[VAR504:.*]] = arith.ori %[[VAR502]], %[[VAR503]] : i1
// CHECK: %[[TRUE_25:.*]] = arith.constant true
// CHECK: %[[VAR505:.*]] = arith.xori %[[VAR497]], %[[TRUE_25]] : i1
// CHECK: %[[VAR506:.*]] = arith.andi %[[VAR504]], %[[VAR505]] : i1
// CHECK: %[[VAR507:.*]] = arith.andi %[[VAR506]], %[[VAR483]] : i1
// CHECK: %[[VAR508:.*]] = math.copysign %[[CST_23]], %[[VAR493]] : f32
// CHECK: %[[VAR509:.*]] = arith.select %[[VAR507]], %[[VAR508]], %[[VAR493]] : f32
// CHECK: %[[VAR510:.*]] = arith.andi %[[VAR506]], %[[VAR484]] : i1
// CHECK: %[[VAR511:.*]] = math.copysign %[[CST_23]], %[[VAR496]] : f32
// CHECK: %[[VAR512:.*]] = arith.select %[[VAR510]], %[[VAR511]], %[[VAR496]] : f32
// CHECK: %[[VAR513:.*]] = arith.andi %[[VAR506]], %[[VAR466]] : i1
// CHECK: %[[VAR514:.*]] = math.copysign %[[CST_23]], %[[VAR487]] : f32
// CHECK: %[[VAR515:.*]] = arith.select %[[VAR513]], %[[VAR514]], %[[VAR487]] : f32
// CHECK: %[[VAR516:.*]] = arith.andi %[[VAR506]], %[[VAR467]] : i1
// CHECK: %[[VAR517:.*]] = math.copysign %[[CST_23]], %[[VAR490]] : f32
// CHECK: %[[VAR518:.*]] = arith.select %[[VAR516]], %[[VAR517]], %[[VAR490]] : f32
// CHECK: %[[VAR519:.*]] = arith.ori %[[VAR497]], %[[VAR506]] : i1
// CHECK: %[[VAR520:.*]] = arith.andi %[[VAR462]], %[[VAR519]] : i1
// CHECK: %[[VAR521:.*]] = arith.mulf %[[VAR509]], %[[VAR515]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR522:.*]] = arith.mulf %[[VAR512]], %[[VAR518]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR523:.*]] = arith.subf %[[VAR521]], %[[VAR522]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR524:.*]] = arith.mulf %[[CST_22]], %[[VAR523]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR525:.*]] = arith.select %[[VAR520]], %[[VAR524]], %[[VAR454]] : f32
// CHECK: %[[VAR526:.*]] = arith.mulf %[[VAR512]], %[[VAR515]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR527:.*]] = arith.mulf %[[VAR509]], %[[VAR518]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR528:.*]] = arith.addf %[[VAR526]], %[[VAR527]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR529:.*]] = arith.mulf %[[CST_22]], %[[VAR528]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR530:.*]] = arith.select %[[VAR520]], %[[VAR529]], %[[VAR459]] : f32
// CHECK: %[[VAR531:.*]] = complex.create %[[VAR525]], %[[VAR530]] : complex<f32>
// CHECK: return %[[VAR531]] : complex<f32>

// -----

// CHECK-LABEL: func @complex_div_with_fmf
// CHECK-SAME: (%[[LHS:.*]]: complex<f32>, %[[RHS:.*]]: complex<f32>)
func.func @complex_div_with_fmf(%lhs: complex<f32>, %rhs: complex<f32>) -> complex<f32> {
  %div = complex.div %lhs, %rhs fastmath<nnan,contract> : complex<f32>
  return %div : complex<f32>
}
// CHECK: %[[LHS_REAL:.*]] = complex.re %[[LHS]] : complex<f32>
// CHECK: %[[LHS_IMAG:.*]] = complex.im %[[LHS]] : complex<f32>
// CHECK: %[[RHS_REAL:.*]] = complex.re %[[RHS]] : complex<f32>
// CHECK: %[[RHS_IMAG:.*]] = complex.im %[[RHS]] : complex<f32>

// CHECK: %[[RHS_REAL_IMAG_RATIO:.*]] = arith.divf %[[RHS_REAL]], %[[RHS_IMAG]] fastmath<nnan,contract> : f32
// CHECK: %[[RHS_REAL_TIMES_RHS_REAL_IMAG_RATIO:.*]] = arith.mulf %[[RHS_REAL_IMAG_RATIO]], %[[RHS_REAL]] fastmath<nnan,contract> : f32
// CHECK: %[[RHS_REAL_IMAG_DENOM:.*]] = arith.addf %[[RHS_IMAG]], %[[RHS_REAL_TIMES_RHS_REAL_IMAG_RATIO]] fastmath<nnan,contract> : f32
// CHECK: %[[LHS_REAL_TIMES_RHS_REAL_IMAG_RATIO:.*]] = arith.mulf %[[LHS_REAL]], %[[RHS_REAL_IMAG_RATIO]] fastmath<nnan,contract> : f32
// CHECK: %[[REAL_NUMERATOR_1:.*]] = arith.addf %[[LHS_REAL_TIMES_RHS_REAL_IMAG_RATIO]], %[[LHS_IMAG]] fastmath<nnan,contract> : f32
// CHECK: %[[RESULT_REAL_1:.*]] = arith.divf %[[REAL_NUMERATOR_1]], %[[RHS_REAL_IMAG_DENOM]] fastmath<nnan,contract> : f32
// CHECK: %[[LHS_IMAG_TIMES_RHS_REAL_IMAG_RATIO:.*]] = arith.mulf %[[LHS_IMAG]], %[[RHS_REAL_IMAG_RATIO]] fastmath<nnan,contract> : f32
// CHECK: %[[IMAG_NUMERATOR_1:.*]] = arith.subf %[[LHS_IMAG_TIMES_RHS_REAL_IMAG_RATIO]], %[[LHS_REAL]] fastmath<nnan,contract> : f32
// CHECK: %[[RESULT_IMAG_1:.*]] = arith.divf %[[IMAG_NUMERATOR_1]], %[[RHS_REAL_IMAG_DENOM]] fastmath<nnan,contract> : f32

// CHECK: %[[RHS_IMAG_REAL_RATIO:.*]] = arith.divf %[[RHS_IMAG]], %[[RHS_REAL]] fastmath<nnan,contract> : f32
// CHECK: %[[RHS_IMAG_TIMES_RHS_IMAG_REAL_RATIO:.*]] = arith.mulf %[[RHS_IMAG_REAL_RATIO]], %[[RHS_IMAG]] fastmath<nnan,contract> : f32
// CHECK: %[[RHS_IMAG_REAL_DENOM:.*]] = arith.addf %[[RHS_REAL]], %[[RHS_IMAG_TIMES_RHS_IMAG_REAL_RATIO]] fastmath<nnan,contract> : f32
// CHECK: %[[LHS_IMAG_TIMES_RHS_IMAG_REAL_RATIO:.*]] = arith.mulf %[[LHS_IMAG]], %[[RHS_IMAG_REAL_RATIO]] fastmath<nnan,contract> : f32
// CHECK: %[[REAL_NUMERATOR_2:.*]] = arith.addf %[[LHS_REAL]], %[[LHS_IMAG_TIMES_RHS_IMAG_REAL_RATIO]] fastmath<nnan,contract> : f32
// CHECK: %[[RESULT_REAL_2:.*]] = arith.divf %[[REAL_NUMERATOR_2]], %[[RHS_IMAG_REAL_DENOM]] fastmath<nnan,contract> : f32
// CHECK: %[[LHS_REAL_TIMES_RHS_IMAG_REAL_RATIO:.*]] = arith.mulf %[[LHS_REAL]], %[[RHS_IMAG_REAL_RATIO]] fastmath<nnan,contract> : f32
// CHECK: %[[IMAG_NUMERATOR_2:.*]] = arith.subf %[[LHS_IMAG]], %[[LHS_REAL_TIMES_RHS_IMAG_REAL_RATIO]] fastmath<nnan,contract> : f32
// CHECK: %[[RESULT_IMAG_2:.*]] = arith.divf %[[IMAG_NUMERATOR_2]], %[[RHS_IMAG_REAL_DENOM]] fastmath<nnan,contract> : f32

// Case 1. Zero denominator, numerator contains at most one NaN value.
// CHECK: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[RHS_REAL_ABS:.*]] = math.absf %[[RHS_REAL]] fastmath<nnan,contract> : f32
// CHECK: %[[RHS_REAL_ABS_IS_ZERO:.*]] = arith.cmpf oeq, %[[RHS_REAL_ABS]], %[[ZERO]] : f32
// CHECK: %[[RHS_IMAG_ABS:.*]] = math.absf %[[RHS_IMAG]] fastmath<nnan,contract> : f32
// CHECK: %[[RHS_IMAG_ABS_IS_ZERO:.*]] = arith.cmpf oeq, %[[RHS_IMAG_ABS]], %[[ZERO]] : f32
// CHECK: %[[LHS_REAL_IS_NOT_NAN:.*]] = arith.cmpf ord, %[[LHS_REAL]], %[[ZERO]] : f32
// CHECK: %[[LHS_IMAG_IS_NOT_NAN:.*]] = arith.cmpf ord, %[[LHS_IMAG]], %[[ZERO]] : f32
// CHECK: %[[LHS_CONTAINS_NOT_NAN_VALUE:.*]] = arith.ori %[[LHS_REAL_IS_NOT_NAN]], %[[LHS_IMAG_IS_NOT_NAN]] : i1
// CHECK: %[[RHS_IS_ZERO:.*]] = arith.andi %[[RHS_REAL_ABS_IS_ZERO]], %[[RHS_IMAG_ABS_IS_ZERO]] : i1
// CHECK: %[[RESULT_IS_INFINITY:.*]] = arith.andi %[[LHS_CONTAINS_NOT_NAN_VALUE]], %[[RHS_IS_ZERO]] : i1
// CHECK: %[[INF:.*]] = arith.constant 0x7F800000 : f32
// CHECK: %[[INF_WITH_SIGN_OF_RHS_REAL:.*]] = math.copysign %[[INF]], %[[RHS_REAL]] : f32
// CHECK: %[[INFINITY_RESULT_REAL:.*]] = arith.mulf %[[INF_WITH_SIGN_OF_RHS_REAL]], %[[LHS_REAL]] fastmath<nnan,contract> : f32
// CHECK: %[[INFINITY_RESULT_IMAG:.*]] = arith.mulf %[[INF_WITH_SIGN_OF_RHS_REAL]], %[[LHS_IMAG]] fastmath<nnan,contract> : f32

// Case 2. Infinite numerator, finite denominator.
// CHECK: %[[RHS_REAL_FINITE:.*]] = arith.cmpf one, %[[RHS_REAL_ABS]], %[[INF]] : f32
// CHECK: %[[RHS_IMAG_FINITE:.*]] = arith.cmpf one, %[[RHS_IMAG_ABS]], %[[INF]] : f32
// CHECK: %[[RHS_IS_FINITE:.*]] = arith.andi %[[RHS_REAL_FINITE]], %[[RHS_IMAG_FINITE]] : i1
// CHECK: %[[LHS_REAL_ABS:.*]] = math.absf %[[LHS_REAL]] fastmath<nnan,contract> : f32
// CHECK: %[[LHS_REAL_INFINITE:.*]] = arith.cmpf oeq, %[[LHS_REAL_ABS]], %[[INF]] : f32
// CHECK: %[[LHS_IMAG_ABS:.*]] = math.absf %[[LHS_IMAG]] fastmath<nnan,contract> : f32
// CHECK: %[[LHS_IMAG_INFINITE:.*]] = arith.cmpf oeq, %[[LHS_IMAG_ABS]], %[[INF]] : f32
// CHECK: %[[LHS_IS_INFINITE:.*]] = arith.ori %[[LHS_REAL_INFINITE]], %[[LHS_IMAG_INFINITE]] : i1
// CHECK: %[[INF_NUM_FINITE_DENOM:.*]] = arith.andi %[[LHS_IS_INFINITE]], %[[RHS_IS_FINITE]] : i1
// CHECK: %[[ONE:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[LHS_REAL_IS_INF:.*]] = arith.select %[[LHS_REAL_INFINITE]], %[[ONE]], %[[ZERO]] : f32
// CHECK: %[[LHS_REAL_IS_INF_WITH_SIGN:.*]] = math.copysign %[[LHS_REAL_IS_INF]], %[[LHS_REAL]] : f32
// CHECK: %[[LHS_IMAG_IS_INF:.*]] = arith.select %[[LHS_IMAG_INFINITE]], %[[ONE]], %[[ZERO]] : f32
// CHECK: %[[LHS_IMAG_IS_INF_WITH_SIGN:.*]] = math.copysign %[[LHS_IMAG_IS_INF]], %[[LHS_IMAG]] : f32
// CHECK: %[[LHS_REAL_IS_INF_WITH_SIGN_TIMES_RHS_REAL:.*]] = arith.mulf %[[LHS_REAL_IS_INF_WITH_SIGN]], %[[RHS_REAL]] fastmath<nnan,contract> : f32
// CHECK: %[[LHS_IMAG_IS_INF_WITH_SIGN_TIMES_RHS_IMAG:.*]] = arith.mulf %[[LHS_IMAG_IS_INF_WITH_SIGN]], %[[RHS_IMAG]] fastmath<nnan,contract> : f32
// CHECK: %[[INF_MULTIPLICATOR_1:.*]] = arith.addf %[[LHS_REAL_IS_INF_WITH_SIGN_TIMES_RHS_REAL]], %[[LHS_IMAG_IS_INF_WITH_SIGN_TIMES_RHS_IMAG]] fastmath<nnan,contract> : f32
// CHECK: %[[RESULT_REAL_3:.*]] = arith.mulf %[[INF]], %[[INF_MULTIPLICATOR_1]] fastmath<nnan,contract> : f32
// CHECK: %[[LHS_REAL_IS_INF_WITH_SIGN_TIMES_RHS_IMAG:.*]] = arith.mulf %[[LHS_REAL_IS_INF_WITH_SIGN]], %[[RHS_IMAG]] fastmath<nnan,contract> : f32
// CHECK: %[[LHS_IMAG_IS_INF_WITH_SIGN_TIMES_RHS_REAL:.*]] = arith.mulf %[[LHS_IMAG_IS_INF_WITH_SIGN]], %[[RHS_REAL]] fastmath<nnan,contract> : f32
// CHECK: %[[INF_MULTIPLICATOR_2:.*]] = arith.subf %[[LHS_IMAG_IS_INF_WITH_SIGN_TIMES_RHS_REAL]], %[[LHS_REAL_IS_INF_WITH_SIGN_TIMES_RHS_IMAG]] fastmath<nnan,contract> : f32
// CHECK: %[[RESULT_IMAG_3:.*]] = arith.mulf %[[INF]], %[[INF_MULTIPLICATOR_2]] fastmath<nnan,contract> : f32

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
// CHECK: %[[RHS_REAL_IS_INF_WITH_SIGN_TIMES_LHS_REAL:.*]] = arith.mulf %[[LHS_REAL]], %[[RHS_REAL_IS_INF_WITH_SIGN]] fastmath<nnan,contract> : f32
// CHECK: %[[RHS_IMAG_IS_INF_WITH_SIGN_TIMES_LHS_IMAG:.*]] = arith.mulf %[[LHS_IMAG]], %[[RHS_IMAG_IS_INF_WITH_SIGN]] fastmath<nnan,contract> : f32
// CHECK: %[[ZERO_MULTIPLICATOR_1:.*]] = arith.addf %[[RHS_REAL_IS_INF_WITH_SIGN_TIMES_LHS_REAL]], %[[RHS_IMAG_IS_INF_WITH_SIGN_TIMES_LHS_IMAG]] fastmath<nnan,contract> : f32
// CHECK: %[[RESULT_REAL_4:.*]] = arith.mulf %[[ZERO]], %[[ZERO_MULTIPLICATOR_1]] fastmath<nnan,contract> : f32
// CHECK: %[[RHS_REAL_IS_INF_WITH_SIGN_TIMES_LHS_IMAG:.*]] = arith.mulf %[[LHS_IMAG]], %[[RHS_REAL_IS_INF_WITH_SIGN]] fastmath<nnan,contract> : f32
// CHECK: %[[RHS_IMAG_IS_INF_WITH_SIGN_TIMES_LHS_REAL:.*]] = arith.mulf %[[LHS_REAL]], %[[RHS_IMAG_IS_INF_WITH_SIGN]] fastmath<nnan,contract> : f32
// CHECK: %[[ZERO_MULTIPLICATOR_2:.*]] = arith.subf %[[RHS_REAL_IS_INF_WITH_SIGN_TIMES_LHS_IMAG]], %[[RHS_IMAG_IS_INF_WITH_SIGN_TIMES_LHS_REAL]] fastmath<nnan,contract> : f32
// CHECK: %[[RESULT_IMAG_4:.*]] = arith.mulf %[[ZERO]], %[[ZERO_MULTIPLICATOR_2]] fastmath<nnan,contract> : f32

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
