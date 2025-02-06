// RUN: mlir-opt %s --convert-complex-to-standard --split-input-file |\
// RUN: FileCheck %s

// CHECK-LABEL: func @complex_abs
// CHECK-SAME: %[[ARG:.*]]: complex<f32>
func.func @complex_abs(%arg: complex<f32>) -> f32 {
  %abs = complex.abs %arg: complex<f32>
  return %abs : f32
}

// CHECK: %[[REAL:.*]] = complex.re %[[ARG]] : complex<f32>
// CHECK: %[[IMAG:.*]] = complex.im %[[ARG]] : complex<f32>
// CHECK: %[[ONE:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[ABS_REAL:.*]] = math.absf %[[REAL]] : f32
// CHECK: %[[ABS_IMAG:.*]] = math.absf %[[IMAG]] : f32
// CHECK: %[[MAX:.*]] = arith.maximumf %[[ABS_REAL]], %[[ABS_IMAG]] : f32
// CHECK: %[[MIN:.*]] = arith.minimumf %[[ABS_REAL]], %[[ABS_IMAG]] : f32
// CHECK: %[[RATIO:.*]] = arith.divf %[[MIN]], %[[MAX]] : f32
// CHECK: %[[RATIO_SQ:.*]] = arith.mulf %[[RATIO]], %[[RATIO]] : f32
// CHECK: %[[RATIO_SQ_PLUS_ONE:.*]] = arith.addf %[[RATIO_SQ]], %[[ONE]] : f32
// CHECK: %[[SQRT:.*]] = math.sqrt %[[RATIO_SQ_PLUS_ONE]] : f32
// CHECK: %[[ABS_OR_NAN:.*]] = arith.mulf %[[MAX]], %[[SQRT]] : f32
// CHECK: %[[IS_NAN:.*]] = arith.cmpf uno, %[[ABS_OR_NAN]], %[[ABS_OR_NAN]] : f32
// CHECK: %[[ABS:.*]] = arith.select %[[IS_NAN]], %[[MIN]], %[[ABS_OR_NAN]] : f32
// CHECK: return %[[ABS]] : f32

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

// CHECK-LABEL: func.func @complex_expm1(
// CHECK-SAME:    %[[ARG:.*]]: complex<f32>) -> complex<f32> {
func.func @complex_expm1(%arg: complex<f32>) -> complex<f32> {
  %expm1 = complex.expm1 %arg fastmath<nnan,contract> : complex<f32>
  return %expm1 : complex<f32>
}
// CHECK:  %[[REAL:.*]] = complex.re %[[ARG]] : complex<f32>
// CHECK:  %[[IMAG:.*]] = complex.im %[[ARG]] : complex<f32>
// CHECK-DAG:  %[[C0_F32:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:  %[[C1_F32:.*]] = arith.constant 1.000000e+00 : f32
// CHECK:  %[[EXPM1:.*]] = math.expm1 %[[REAL]] fastmath<nnan,contract> : f32
// CHECK:  %[[VAL_6:.*]] = arith.addf %[[EXPM1]], %[[C1_F32]] fastmath<nnan,contract> : f32
// CHECK:  %[[VAL_7:.*]] = math.sin %[[IMAG]] fastmath<nnan,contract> : f32
// CHECK:  %[[VAL_8:.*]] = arith.constant -5.000000e-01 : f32
// CHECK:  %[[VAL_9:.*]] = arith.constant -1.000000e+00 : f32
// CHECK:  %[[VAL_10:.*]] = math.cos %[[IMAG]] fastmath<nnan,contract> : f32
// CHECK:  %[[VAL_11:.*]] = arith.addf %[[VAL_10]], %[[VAL_9]] fastmath<nnan,contract> : f32
// CHECK:  %[[VAL_12:.*]] = arith.mulf %[[IMAG]], %[[IMAG]] fastmath<nnan,contract> : f32
// CHECK:  %[[VAL_13:.*]] = arith.mulf %[[VAL_12]], %[[VAL_12]] fastmath<nnan,contract> : f32
// CHECK-DAG:  %[[COEF0:.*]] = arith.constant 4.73775072E-14 : f32
// CHECK-DAG:  %[[COEF1:.*]] = arith.constant -1.14702848E-11 : f32
// CHECK:  %[[FMA0:.*]] = math.fma %[[COEF0]], %[[VAL_12]], %[[COEF1]] fastmath<nnan,contract> : f32
// CHECK:  %[[COEF2:.*]] = arith.constant 2.08767537E-9 : f32
// CHECK:  %[[FMA1:.*]] = math.fma %[[FMA0]], %[[VAL_12]], %[[COEF2]] fastmath<nnan,contract> : f32
// CHECK:  %[[COEF3:.*]] = arith.constant -2.755732E-7 : f32
// CHECK:  %[[FMA2:.*]] = math.fma %[[FMA1]], %[[VAL_12]], %[[COEF3]] fastmath<nnan,contract> : f32
// CHECK:  %[[COEF4:.*]] = arith.constant 2.48015876E-5 : f32
// CHECK:  %[[FMA3:.*]] = math.fma %[[FMA2]], %[[VAL_12]], %[[COEF4]] fastmath<nnan,contract> : f32
// CHECK:  %[[COEF5:.*]] = arith.constant -0.00138888892 : f32
// CHECK:  %[[FMA4:.*]] = math.fma %[[FMA3]], %[[VAL_12]], %[[COEF5]] fastmath<nnan,contract> : f32
// CHECK:  %[[COEF6:.*]] = arith.constant 0.0416666679 : f32
// CHECK:  %[[FMA5:.*]] = math.fma %[[FMA4]], %[[VAL_12]], %[[COEF6]] fastmath<nnan,contract> : f32
// CHECK-DAG:  %[[VAL_27:.*]] = arith.mulf %[[VAL_13]], %[[FMA5]] fastmath<nnan,contract> : f32
// CHECK-DAG:  %[[VAL_28:.*]] = arith.mulf %[[VAL_8]], %[[VAL_12]] fastmath<nnan,contract> : f32
// CHECK:  %[[VAL_29:.*]] = arith.addf %[[VAL_27]], %[[VAL_28]] : f32
// CHECK:  %[[VAL_30:.*]] = arith.constant 6.168500e-01 : f32
// CHECK:  %[[VAL_31:.*]] = arith.cmpf oge, %[[VAL_12]], %[[VAL_30]] fastmath<nnan,contract> : f32
// CHECK:  %[[VAL_32:.*]] = arith.select %[[VAL_31]], %[[VAL_11]], %[[VAL_29]] : f32
// CHECK:  %[[VAL_33:.*]] = arith.addf %[[VAL_32]], %[[C1_F32]] fastmath<nnan,contract> : f32
// CHECK:  %[[VAL_34:.*]] = arith.mulf %[[EXPM1]], %[[VAL_33]] fastmath<nnan,contract> : f32
// CHECK:  %[[VAL_35:.*]] = arith.addf %[[VAL_34]], %[[VAL_32]] fastmath<nnan,contract> : f32
// CHECK:  %[[VAL_36:.*]] = arith.cmpf oeq, %[[IMAG]], %[[C0_F32]] fastmath<nnan,contract> : f32
// CHECK:  %[[VAL_37:.*]] = arith.mulf %[[VAL_6]], %[[VAL_7]] fastmath<nnan,contract> : f32
// CHECK:  %[[VAL_38:.*]] = arith.select %[[VAL_36]], %[[C0_F32]], %[[VAL_37]] : f32
// CHECK:  %[[RESULT:.*]] = complex.create %[[VAL_35]], %[[VAL_38]] : complex<f32>
// CHECK:  return %[[RESULT]] : complex<f32>

// -----

// CHECK-LABEL: func @complex_log
// CHECK-SAME: %[[ARG:.*]]: complex<f32>
func.func @complex_log(%arg: complex<f32>) -> complex<f32> {
  %log = complex.log %arg: complex<f32>
  return %log : complex<f32>
}
// CHECK: %[[REAL:.*]] = complex.re %[[ARG]] : complex<f32>
// CHECK: %[[IMAG:.*]] = complex.im %[[ARG]] : complex<f32>
// CHECK: %[[ONE:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[ABS_REAL:.*]] = math.absf %[[REAL]] : f32
// CHECK: %[[ABS_IMAG:.*]] = math.absf %[[IMAG]] : f32
// CHECK: %[[MAX:.*]] = arith.maximumf %[[ABS_REAL]], %[[ABS_IMAG]] : f32
// CHECK: %[[MIN:.*]] = arith.minimumf %[[ABS_REAL]], %[[ABS_IMAG]] : f32
// CHECK: %[[RATIO:.*]] = arith.divf %[[MIN]], %[[MAX]] : f32
// CHECK: %[[RATIO_SQ:.*]] = arith.mulf %[[RATIO]], %[[RATIO]] : f32
// CHECK: %[[RATIO_SQ_PLUS_ONE:.*]] = arith.addf %[[RATIO_SQ]], %[[ONE]] : f32
// CHECK: %[[SQRT:.*]] = math.sqrt %[[RATIO_SQ_PLUS_ONE]] : f32
// CHECK: %[[RESULT:.*]] = arith.mulf %[[MAX]], %[[SQRT]] : f32
// CHECK: %[[IS_NAN:.*]] = arith.cmpf uno, %[[RESULT]], %[[RESULT]] : f32
// CHECK: %[[ABS:.*]] = arith.select %[[IS_NAN]], %[[MIN]], %[[RESULT]] : f32
// CHECK: %[[RESULT_REAL:.*]] = math.log %[[ABS]] : f32
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
// CHECK: %[[REAL_PLUS_ONE:.*]] = arith.addf %[[REAL]], %[[ONE]] : f32
// CHECK: %[[ABS_REAL_PLUS_ONE:.*]] = math.absf %[[REAL_PLUS_ONE]] : f32
// CHECK: %[[ABS_IMAG:.*]] = math.absf %[[IMAG]] : f32
// CHECK: %[[MAX:.*]] = arith.maximumf %[[ABS_REAL_PLUS_ONE]], %[[ABS_IMAG]] : f32
// CHECK: %[[MIN:.*]] = arith.minimumf %[[ABS_REAL_PLUS_ONE]], %[[ABS_IMAG]] : f32
// CHECK: %[[CMPF:.*]] = arith.cmpf ogt, %[[REAL_PLUS_ONE]], %[[ABS_IMAG]] : f32
// CHECK: %[[MAX_MINUS_ONE:.*]] = arith.subf %[[MAX]], %[[ONE]] : f32
// CHECK: %[[SELECT:.*]] = arith.select %[[CMPF]], %0, %[[MAX_MINUS_ONE]] : f32
// CHECK: %[[MIN_MAX_RATIO:.*]] = arith.divf %[[MIN]], %[[MAX]] : f32
// CHECK: %[[LOG_1:.*]] = math.log1p %[[SELECT]] : f32
// CHECK: %[[RATIO_SQ:.*]] = arith.mulf %[[MIN_MAX_RATIO]], %[[MIN_MAX_RATIO]] : f32
// CHECK: %[[LOG_SQ:.*]] = math.log1p %[[RATIO_SQ]] : f32
// CHECK: %[[HALF_LOG_SQ:.*]] = arith.mulf %cst, %[[LOG_SQ]] : f32
// CHECK: %[[R:.*]] = arith.addf %[[HALF_LOG_SQ]], %[[LOG_1]] : f32
// CHECK: %[[ISNAN:.*]] = arith.cmpf uno, %[[R]], %[[R]] : f32
// CHECK: %[[RESULT_REAL:.*]] = arith.select %[[ISNAN]], %[[MIN]], %[[R]] : f32
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
// CHECK: %[[LHS_IMAG:.*]] = complex.im %[[LHS]] : complex<f32>
// CHECK: %[[RHS_REAL:.*]] = complex.re %[[RHS]] : complex<f32>
// CHECK: %[[RHS_IMAG:.*]] = complex.im %[[RHS]] : complex<f32>

// CHECK: %[[LHS_REAL_TIMES_RHS_REAL:.*]] = arith.mulf %[[LHS_REAL]], %[[RHS_REAL]] : f32
// CHECK: %[[LHS_IMAG_TIMES_RHS_IMAG:.*]] = arith.mulf %[[LHS_IMAG]], %[[RHS_IMAG]] : f32
// CHECK: %[[REAL:.*]] = arith.subf %[[LHS_REAL_TIMES_RHS_REAL]], %[[LHS_IMAG_TIMES_RHS_IMAG]] : f32

// CHECK: %[[LHS_IMAG_TIMES_RHS_REAL:.*]] = arith.mulf %[[LHS_IMAG]], %[[RHS_REAL]] : f32
// CHECK: %[[LHS_REAL_TIMES_RHS_IMAG:.*]] = arith.mulf %[[LHS_REAL]], %[[RHS_IMAG]] : f32
// CHECK: %[[IMAG:.*]] = arith.addf %[[LHS_IMAG_TIMES_RHS_REAL]], %[[LHS_REAL_TIMES_RHS_IMAG]] : f32

// CHECK: %[[RESULT:.*]] = complex.create %[[REAL]], %[[IMAG]] : complex<f32>
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
// CHECK: %[[REAL2:.*]] = complex.re %[[ARG]] : complex<f32>
// CHECK: %[[IMAG2:.*]] = complex.im %[[ARG]] : complex<f32>
// CHECK: %[[ONE:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[ABS_REAL:.*]] = math.absf %[[REAL2]] : f32
// CHECK: %[[ABS_IMAG:.*]] = math.absf %[[IMAG2]] : f32
// CHECK: %[[MAX:.*]] = arith.maximumf %[[ABS_REAL]], %[[ABS_IMAG]] : f32
// CHECK: %[[MIN:.*]] = arith.minimumf %[[ABS_REAL]], %[[ABS_IMAG]] : f32
// CHECK: %[[RATIO:.*]] = arith.divf %[[MIN]], %[[MAX]] : f32
// CHECK: %[[RATIO_SQ:.*]] = arith.mulf %[[RATIO]], %[[RATIO]] : f32
// CHECK: %[[RATIO_SQ_PLUS_ONE:.*]] = arith.addf %[[RATIO_SQ]], %[[ONE]] : f32
// CHECK: %[[SQRT:.*]] = math.sqrt %[[RATIO_SQ_PLUS_ONE]] : f32
// CHECK: %[[ABS_OR_NAN:.*]] = arith.mulf %[[MAX]], %[[SQRT]] : f32
// CHECK: %[[IS_NAN:.*]] = arith.cmpf uno, %[[ABS_OR_NAN]], %[[ABS_OR_NAN]] : f32
// CHECK: %[[ABS:.*]] = arith.select %[[IS_NAN]], %[[MIN]], %[[ABS_OR_NAN]] : f32
// CHECK: %[[REAL_SIGN:.*]] = arith.divf %[[REAL]], %[[ABS]] : f32
// CHECK: %[[IMAG_SIGN:.*]] = arith.divf %[[IMAG]], %[[ABS]] : f32
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

// CHECK: %[[IMAG:.*]] = complex.re %[[ARG]] : complex<f32>
// CHECK: %[[V0:.*]] = complex.im %[[ARG]] : complex<f32>
// CHECK: %[[NEG_ONE:.*]] = arith.constant -1.000000e+00 : f32
// CHECK: %[[REAL:.*]] = arith.mulf %[[V0]], %[[NEG_ONE]] : f32
// CHECK: %[[INF:.*]] = arith.constant 0x7F800000 : f32
// CHECK: %[[FOUR:.*]] = arith.constant 4.000000e+00 : f32
// CHECK: %[[TWO_REAL:.*]] = arith.addf %[[REAL]], %[[REAL]] : f32
// CHECK: %[[NEG_TWO_REAL:.*]] = arith.mulf %[[NEG_ONE]], %[[TWO_REAL]] : f32
// CHECK: %[[EXPM1:.*]] = math.expm1 %[[TWO_REAL]] : f32
// CHECK: %[[EXPM1_2:.*]] = math.expm1 %[[NEG_TWO_REAL]] : f32
// CHECK: %[[REAL_NUM:.*]] = arith.subf %[[EXPM1]], %[[EXPM1_2]] : f32
// CHECK: %[[COS:.*]] = math.cos %[[IMAG]] : f32
// CHECK: %[[COS_SQ:.*]] = arith.mulf %[[COS]], %[[COS]] : f32
// CHECK: %[[FOUR_COS_SQ:.*]] = arith.mulf %[[COS_SQ]], %[[FOUR]] : f32
// CHECK: %[[SIN:.*]] = math.sin %[[IMAG]] : f32
// CHECK: %[[MUL:.*]] = arith.mulf %[[COS]], %[[SIN]] : f32
// CHECK: %[[IMAG_NUM:.*]] = arith.mulf %[[FOUR]], %[[MUL]] : f32
// CHECK: %[[ADD:.*]] = arith.addf %[[EXPM1]], %[[EXPM1_2]] : f32
// CHECK: %[[DENOM:.*]] = arith.addf %[[ADD]], %[[FOUR_COS_SQ]] : f32
// CHECK: %[[IS_INF:.*]] = arith.cmpf oeq, %[[ADD]], %[[INF]] : f32
// CHECK: %[[LIMIT:.*]] = math.copysign %[[NEG_ONE]], %[[REAL]] : f32
// CHECK: %[[RESULT_REAL:.*]] = arith.divf %[[REAL_NUM]], %[[DENOM]] : f32
// CHECK: %[[RESULT_REAL2:.*]] = arith.select %[[IS_INF]], %[[LIMIT]], %[[RESULT_REAL]] : f32
// CHECK: %[[RESULT_IMAG:.*]] = arith.divf %[[IMAG_NUM]], %[[DENOM]] : f32
// CHECK: %[[ABS_REAL:.*]] = math.absf %[[REAL]] : f32
// CHECK: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[NAN:.*]] = arith.constant 0x7FC00000 : f32
// CHECK: %[[ABS_REAL_INF:.*]] = arith.cmpf oeq, %[[ABS_REAL]], %[[INF]] : f32
// CHECK: %[[IMAG_ZERO:.*]] = arith.cmpf oeq, %[[IMAG]], %[[ZERO]] : f32
// CHECK: %true = arith.constant true
// CHECK: %[[ABS_REAL_NOT_INF:.*]] = arith.xori %[[ABS_REAL_INF]], %true : i1
// CHECK: %[[IMAG_IS_NAN:.*]] = arith.cmpf uno, %[[IMAG_NUM]], %[[IMAG_NUM]] : f32
// CHECK: %[[REAL_IS_NAN:.*]] = arith.andi %[[IMAG_IS_NAN]], %[[ABS_REAL_NOT_INF]] : i1
// CHECK: %[[AND:.*]] = arith.andi %[[ABS_REAL_INF]], %[[IMAG_IS_NAN]] : i1
// CHECK: %[[IMAG_IS_NAN2:.*]] = arith.ori %[[IMAG_ZERO]], %[[AND]] : i1
// CHECK: %[[RESULT_REAL3:.*]] = arith.select %[[REAL_IS_NAN]], %[[NAN]], %[[RESULT_REAL2]] : f32
// CHECK: %[[RESULT_REAL:.*]] = arith.select %[[IMAG_IS_NAN2]], %[[ZERO]], %[[RESULT_IMAG]] : f32
// CHECK: %[[RESULT_IMAG:.*]] = arith.mulf %[[RESULT_REAL3]], %[[NEG_ONE]]
// CHECK: %[[RESULT:.*]] = complex.create %[[RESULT_REAL]], %[[RESULT_IMAG]] : complex<f32>
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
// CHECK: %[[NEG_ONE:.*]] = arith.constant -1.000000e+00 : f32
// CHECK: %[[INF:.*]] = arith.constant 0x7F800000 : f32
// CHECK: %[[FOUR:.*]] = arith.constant 4.000000e+00 : f32
// CHECK: %[[TWO_REAL:.*]] = arith.addf %[[REAL]], %[[REAL]] : f32
// CHECK: %[[NEG_TWO_REAL:.*]] = arith.mulf %[[NEG_ONE]], %[[TWO_REAL]] : f32
// CHECK: %[[EXPM1:.*]] = math.expm1 %[[TWO_REAL]] : f32
// CHECK: %[[EXPM1_2:.*]] = math.expm1 %[[NEG_TWO_REAL]] : f32
// CHECK: %[[REAL_NUM:.*]] = arith.subf %[[EXPM1]], %[[EXPM1_2]] : f32
// CHECK: %[[COS:.*]] = math.cos %[[IMAG]] : f32
// CHECK: %[[COS_SQ:.*]] = arith.mulf %[[COS]], %[[COS]] : f32
// CHECK: %[[FOUR_COS_SQ:.*]] = arith.mulf %[[COS_SQ]], %[[FOUR]] : f32
// CHECK: %[[SIN:.*]] = math.sin %[[IMAG]] : f32
// CHECK: %[[MUL:.*]] = arith.mulf %[[COS]], %[[SIN]] : f32
// CHECK: %[[IMAG_NUM:.*]] = arith.mulf %[[FOUR]], %[[MUL]] : f32
// CHECK: %[[ADD:.*]] = arith.addf %[[EXPM1]], %[[EXPM1_2]] : f32
// CHECK: %[[DENOM:.*]] = arith.addf %[[ADD]], %[[FOUR_COS_SQ]] : f32
// CHECK: %[[IS_INF:.*]] = arith.cmpf oeq, %[[ADD]], %[[INF]] : f32
// CHECK: %[[LIMIT:.*]] = math.copysign %[[NEG_ONE]], %[[REAL]] : f32
// CHECK: %[[RESULT_REAL:.*]] = arith.divf %[[REAL_NUM]], %[[DENOM]] : f32
// CHECK: %[[RESULT_REAL2:.*]] = arith.select %[[IS_INF]], %[[LIMIT]], %[[RESULT_REAL]] : f32
// CHECK: %[[RESULT_IMAG:.*]] = arith.divf %[[IMAG_NUM]], %[[DENOM]] : f32
// CHECK: %[[ABS_REAL:.*]] = math.absf %[[REAL]] : f32
// CHECK: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[NAN:.*]] = arith.constant 0x7FC00000 : f32
// CHECK: %[[ABS_REAL_INF:.*]] = arith.cmpf oeq, %[[ABS_REAL]], %[[INF]] : f32
// CHECK: %[[IMAG_ZERO:.*]] = arith.cmpf oeq, %[[IMAG]], %[[ZERO]] : f32
// CHECK: %true = arith.constant true
// CHECK: %[[ABS_REAL_NOT_INF:.*]] = arith.xori %[[ABS_REAL_INF]], %true : i1
// CHECK: %[[IMAG_IS_NAN:.*]] = arith.cmpf uno, %[[IMAG_NUM]], %[[IMAG_NUM]] : f32
// CHECK: %[[REAL_IS_NAN:.*]] = arith.andi %[[IMAG_IS_NAN]], %[[ABS_REAL_NOT_INF]] : i1
// CHECK: %[[AND:.*]] = arith.andi %[[ABS_REAL_INF]], %[[IMAG_IS_NAN]] : i1
// CHECK: %[[IMAG_IS_NAN2:.*]] = arith.ori %[[IMAG_ZERO]], %[[AND]] : i1
// CHECK: %[[RESULT_REAL3:.*]] = arith.select %[[REAL_IS_NAN]], %[[NAN]], %[[RESULT_REAL2]] : f32
// CHECK: %[[RESULT_IMAG2:.*]] = arith.select %[[IMAG_IS_NAN2]], %[[ZERO]], %[[RESULT_IMAG]] : f32
// CHECK: %[[RESULT:.*]] = complex.create %[[RESULT_REAL3]], %[[RESULT_IMAG2]] : complex<f32>
// CHECK: return %[[RESULT]] : complex<f32>

// -----

// CHECK-LABEL: func @complex_sqrt
// CHECK-SAME: %[[ARG:.*]]: complex<f32>
func.func @complex_sqrt(%arg: complex<f32>) -> complex<f32> {
  %sqrt = complex.sqrt %arg : complex<f32>
  return %sqrt : complex<f32>
}

// CHECK: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[HALF:.*]] = arith.constant 5.000000e-01 : f32
// CHECK: %[[RE:.*]] = complex.re %[[ARG]] : complex<f32>
// CHECK: %[[IM:.*]] = complex.im %[[ARG]] : complex<f32>
// CHECK: %[[ONE:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[ABSRE:.*]] = math.absf %[[RE]] : f32
// CHECK: %[[ABSIM:.*]] = math.absf %[[IM]] : f32
// CHECK: %[[MAX:.*]] = arith.maximumf %[[ABSRE]], %[[ABSIM]] : f32
// CHECK: %[[MIN:.*]] = arith.minimumf %[[ABSRE]], %[[ABSIM]] : f32
// CHECK: %[[RATIO:.*]] = arith.divf %[[MIN]], %[[MAX]] : f32
// CHECK: %[[RATIO_SQ:.*]] = arith.mulf %[[RATIO]], %[[RATIO]] : f32
// CHECK: %[[RATIO_SQ_PLUS_ONE:.*]] = arith.addf %[[RATIO_SQ]], %[[ONE]] : f32
// CHECK: %[[QUARTER:.*]] = arith.constant 2.500000e-01 : f32
// CHECK: %[[SQRT_MAX:.*]] = math.sqrt %[[MAX]] : f32
// CHECK: %[[POW:.*]] = math.powf %[[RATIO_SQ_PLUS_ONE]], %[[QUARTER]] : f32
// CHECK: %[[SQRT_ABS_OR_NAN:.*]] = arith.mulf %[[SQRT_MAX]], %[[POW]] : f32
// CHECK: %[[IS_NAN:.*]] = arith.cmpf uno, %[[SQRT_ABS_OR_NAN]], %[[SQRT_ABS_OR_NAN]] : f32
// CHECK: %[[SQRT_ABS:.*]] = arith.select %[[IS_NAN]], %[[MIN]], %[[SQRT_ABS_OR_NAN]] : f32
// CHECK: %[[ARGARG:.*]] = math.atan2 %[[IM]], %[[RE]] : f32
// CHECK: %[[SQRTARG:.*]] = arith.mulf %[[ARGARG]], %[[HALF]] : f32
// CHECK: %[[COS:.*]] = math.cos %[[SQRTARG]] : f32
// CHECK: %[[SIN:.*]] = math.sin %[[SQRTARG]] : f32
// CHECK: %[[SIN_ZERO:.*]] = arith.cmpf oeq, %[[SIN]], %[[ZERO]] : f32
// CHECK: %[[RESULT_RE:.*]] = arith.mulf %[[SQRT_ABS]], %[[COS]] : f32
// CHECK: %[[RESULT_IM:.*]] = arith.mulf %[[SQRT_ABS]], %[[SIN]] : f32
// CHECK: %[[RESULT_IM2:.*]] = arith.select %[[SIN_ZERO]], %[[ZERO]], %[[RESULT_IM]] : f32
// CHECK: %[[INF:.*]] = arith.constant 0x7F800000 : f32
// CHECK: %[[NINF:.*]] = arith.constant 0xFF800000 : f32
// CHECK: %[[NAN:.*]] = arith.constant 0x7FC00000 : f32
// CHECK: %[[ABSIM:.*]] = math.absf %[[IM]] : f32
// CHECK: %[[ABSIMINF:.*]] = arith.cmpf oeq, %[[ABSIM]], %[[INF]] : f32
// CHECK: %[[ABSIMNOTINF:.*]] = arith.cmpf one, %[[ABSIM]], %[[INF]] : f32
// CHECK: %[[REINF:.*]] = arith.cmpf oeq, %[[RE]], %[[INF]] : f32
// CHECK: %[[RENINF:.*]] = arith.cmpf oeq, %[[RE]], %[[NINF]] : f32
// CHECK: %[[RESULT_RE_ZERO:.*]] = arith.andi %[[RENINF]], %[[ABSIMNOTINF]] : i1
// CHECK: %[[RESULT_RE2:.*]] = arith.select %[[RESULT_RE_ZERO]], %[[ZERO]], %[[RESULT_RE]] : f32
// CHECK: %[[RESUL_IM_INF:.*]] = arith.ori %[[ABSIMINF]], %[[REINF]] : i1
// CHECK: %[[RESULT_RE3:.*]] = arith.select %[[RESUL_IM_INF]], %[[INF]], %[[RESULT_RE2]] : f32
// CHECK: %[[INF_IM_SIGN:.*]] = math.copysign %[[INF]], %[[IM]] : f32
// CHECK: %[[RESULT_IM_NAN:.*]] = arith.cmpf uno, %[[SQRT_ABS]], %[[SQRT_ABS]] : f32
// CHECK: %[[RESULT_IM3:.*]] = arith.select %[[RESULT_IM_NAN]], %[[NAN]], %[[RESULT_IM2]] : f32
// CHECK: %[[RESULT_IM_INF:.*]] = arith.ori %[[ABSIMINF]], %[[RENINF]] : i1
// CHECK: %[[RESULT_IM4:.*]] = arith.select %[[RESULT_IM_INF]], %[[INF_IM_SIGN]], %[[RESULT_IM3]] : f32
// CHECK: %[[RESULT_ZERO:.*]] = arith.cmpf oeq, %[[SQRT_ABS]], %[[ZERO]] : f32
// CHECK: %[[RESULT_RE4:.*]] = arith.select %[[RESULT_ZERO]], %[[ZERO]], %[[RESULT_RE3]] : f32
// CHECK: %[[RESULT_IM5:.*]] = arith.select %[[RESULT_ZERO]], %[[ZERO]], %[[RESULT_IM4]] : f32
// CHECK: %[[RESULT:.*]] = complex.create %[[RESULT_RE4]], %[[RESULT_IM5]] : complex<f32>
// CHECK: return %[[RESULT]] : complex<f32>

// -----

// CHECK-LABEL: func @complex_sqrt_nnan_ninf
// CHECK-SAME: %[[ARG:.*]]: complex<f32>
func.func @complex_sqrt_nnan_ninf(%arg: complex<f32>) -> complex<f32> {
  %sqrt = complex.sqrt %arg fastmath<nnan,ninf> : complex<f32>
  return %sqrt : complex<f32>
}

// CHECK: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[HALF:.*]] = arith.constant 5.000000e-01 : f32
// CHECK: %[[RE:.*]] = complex.re %[[ARG]] : complex<f32>
// CHECK: %[[IM:.*]] = complex.im %[[ARG]] : complex<f32>
// CHECK: %[[ONE:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[ABSRE:.*]] = math.absf %[[RE]] fastmath<nnan,ninf> : f32
// CHECK: %[[ABSIM:.*]] = math.absf %[[IM]] fastmath<nnan,ninf> : f32
// CHECK: %[[MAX:.*]] = arith.maximumf %[[ABSRE]], %[[ABSIM]] fastmath<nnan,ninf> : f32
// CHECK: %[[MIN:.*]] = arith.minimumf %[[ABSRE]], %[[ABSIM]] fastmath<nnan,ninf> : f32
// CHECK: %[[RATIO:.*]] = arith.divf %[[MIN]], %[[MAX]] : f32
// CHECK: %[[RATIO_SQ:.*]] = arith.mulf %[[RATIO]], %[[RATIO]] : f32
// CHECK: %[[RATIO_SQ_PLUS_ONE:.*]] = arith.addf %[[RATIO_SQ]], %[[ONE]] : f32
// CHECK: %[[QUARTER:.*]] = arith.constant 2.500000e-01 : f32
// CHECK: %[[SQRT_MAX:.*]] = math.sqrt %[[MAX]] : f32
// CHECK: %[[POW:.*]] = math.powf %[[RATIO_SQ_PLUS_ONE]], %[[QUARTER]] : f32
// CHECK: %[[SQRT_ABS_OR_NAN:.*]] = arith.mulf %[[SQRT_MAX]], %[[POW]] : f32
// CHECK: %[[IS_NAN:.*]] = arith.cmpf uno, %[[SQRT_ABS_OR_NAN]], %[[SQRT_ABS_OR_NAN]] : f32
// CHECK: %[[SQRT_ABS:.*]] = arith.select %[[IS_NAN]], %[[MIN]], %[[SQRT_ABS_OR_NAN]] : f32
// CHECK: %[[ARGARG:.*]] = math.atan2 %[[IM]], %[[RE]] fastmath<nnan,ninf> : f32
// CHECK: %[[SQRTARG:.*]] = arith.mulf %[[ARGARG]], %[[HALF]] fastmath<nnan,ninf> : f32
// CHECK: %[[COS:.*]] = math.cos %[[SQRTARG]] fastmath<nnan,ninf> : f32
// CHECK: %[[SIN:.*]] = math.sin %[[SQRTARG]] fastmath<nnan,ninf> : f32
// CHECK: %[[SIN_ZERO:.*]] = arith.cmpf oeq, %[[SIN]], %[[ZERO]] fastmath<nnan,ninf> : f32
// CHECK: %[[RESULT_RE:.*]] = arith.mulf %[[SQRT_ABS]], %[[COS]] fastmath<nnan,ninf> : f32
// CHECK: %[[RESULT_IM:.*]] = arith.mulf %[[SQRT_ABS]], %[[SIN]] fastmath<nnan,ninf> : f32
// CHECK: %[[RESULT_IM2:.*]] = arith.select %[[SIN_ZERO]], %[[ZERO]], %[[RESULT_IM]] : f32
// CHECK: %[[RESULT_ZERO:.*]] = arith.cmpf oeq, %[[SQRT_ABS]], %[[ZERO]] fastmath<nnan,ninf> : f32
// CHECK: %[[RESULT_RE2:.*]] = arith.select %[[RESULT_ZERO]], %[[ZERO]], %[[RESULT_RE]] : f32
// CHECK: %[[RESULT_IM3:.*]] = arith.select %[[RESULT_ZERO]], %[[ZERO]], %[[RESULT_IM2]] : f32
// CHECK: %[[RESULT:.*]] = complex.create %[[RESULT_RE2]], %[[RESULT_IM3]] : complex<f32>
// CHECK: return %[[RESULT]] : complex<f32>

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

// CHECK-LABEL: func.func @complex_pow
// CHECK-SAME: %[[LHS:.*]]: complex<f32>, %[[RHS:.*]]: complex<f32>
func.func @complex_pow(%lhs: complex<f32>,
                       %rhs: complex<f32>) -> complex<f32> {
  %pow = complex.pow %lhs, %rhs : complex<f32>
  return %pow : complex<f32>
}

// CHECK: %[[A:.*]] = complex.re %[[LHS]]
// CHECK: %[[B:.*]] = complex.im %[[LHS]]
// CHECK: math.atan2 %[[B]], %[[A]] : f32

// -----

// CHECK-LABEL: func.func @complex_pow_with_fmf
// CHECK-SAME: %[[LHS:.*]]: complex<f32>, %[[RHS:.*]]: complex<f32>
func.func @complex_pow_with_fmf(%lhs: complex<f32>,
                                %rhs: complex<f32>) -> complex<f32> {
  %pow = complex.pow %lhs, %rhs fastmath<nnan,contract> : complex<f32>
  return %pow : complex<f32>
}

// CHECK: %[[A:.*]] = complex.re %[[LHS]]
// CHECK: %[[B:.*]] = complex.im %[[LHS]]
// CHECK: math.atan2 %[[B]], %[[A]] fastmath<nnan,contract> : f32

// -----

// CHECK-LABEL:   func.func @complex_rsqrt
func.func @complex_rsqrt(%arg: complex<f32>) -> complex<f32> {
  %rsqrt = complex.rsqrt %arg : complex<f32>
  return %rsqrt : complex<f32>
}

// CHECK-COUNT-5: arith.select
// CHECK-NOT: arith.select

// -----

// CHECK-LABEL: func @complex_rsqrt_nnan_ninf
// CHECK-SAME: %[[ARG:.*]]: complex<f32>
func.func @complex_rsqrt_nnan_ninf(%arg: complex<f32>) -> complex<f32> {
  %sqrt = complex.rsqrt %arg fastmath<nnan,ninf> : complex<f32>
  return %sqrt : complex<f32>
}

// CHECK-COUNT-3: arith.select
// CHECK-NOT: arith.select

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
// CHECK: %[[REAL:.*]] = complex.re %[[ARG]] : complex<f32>
// CHECK: %[[IMAG:.*]] = complex.im %[[ARG]] : complex<f32>
// CHECK: %[[ONE:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[ABS_REAL:.*]] = math.absf %[[REAL]] fastmath<nnan,contract> : f32
// CHECK: %[[ABS_IMAG:.*]] = math.absf %[[IMAG]] fastmath<nnan,contract> : f32
// CHECK: %[[MAX:.*]] = arith.maximumf %[[ABS_REAL]], %[[ABS_IMAG]] fastmath<nnan,contract> : f32
// CHECK: %[[MIN:.*]] = arith.minimumf %[[ABS_REAL]], %[[ABS_IMAG]] fastmath<nnan,contract> : f32
// CHECK: %[[RATIO:.*]] = arith.divf %[[MIN]], %[[MAX]] fastmath<contract> : f32
// CHECK: %[[RATIO_SQ:.*]] = arith.mulf %[[RATIO]], %[[RATIO]] fastmath<contract> : f32
// CHECK: %[[RATIO_SQ_PLUS_ONE:.*]] = arith.addf %[[RATIO_SQ]], %[[ONE]] fastmath<contract> : f32
// CHECK: %[[SQRT:.*]] = math.sqrt %[[RATIO_SQ_PLUS_ONE]] fastmath<contract> : f32
// CHECK: %[[ABS_OR_NAN:.*]] = arith.mulf %[[MAX]], %[[SQRT]] fastmath<contract> : f32
// CHECK: %[[IS_NAN:.*]] = arith.cmpf uno, %[[ABS_OR_NAN]], %[[ABS_OR_NAN]] fastmath<contract> : f32
// CHECK: %[[ABS:.*]] = arith.select %[[IS_NAN]], %[[MIN]], %[[ABS_OR_NAN]] : f32
// CHECK: return %[[ABS]] : f32

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

// CHECK-LABEL: func @complex_log_with_fmf
// CHECK-SAME: %[[ARG:.*]]: complex<f32>
func.func @complex_log_with_fmf(%arg: complex<f32>) -> complex<f32> {
  %log = complex.log %arg fastmath<nnan,contract> : complex<f32>
  return %log : complex<f32>
}
// CHECK: %[[REAL:.*]] = complex.re %[[ARG]] : complex<f32>
// CHECK: %[[IMAG:.*]] = complex.im %[[ARG]] : complex<f32>
// CHECK: %[[ONE:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[ABS_REAL:.*]] = math.absf %[[REAL]] fastmath<nnan,contract> : f32
// CHECK: %[[ABS_IMAG:.*]] = math.absf %[[IMAG]] fastmath<nnan,contract> : f32
// CHECK: %[[MAX:.*]] = arith.maximumf %[[ABS_REAL]], %[[ABS_IMAG]] fastmath<nnan,contract> : f32
// CHECK: %[[MIN:.*]] = arith.minimumf %[[ABS_REAL]], %[[ABS_IMAG]] fastmath<nnan,contract> : f32
// CHECK: %[[RATIO:.*]] = arith.divf %[[MIN]], %[[MAX]] fastmath<contract> : f32
// CHECK: %[[RATIO_SQ:.*]] = arith.mulf %[[RATIO]], %[[RATIO]] fastmath<contract> : f32
// CHECK: %[[RATIO_SQ_PLUS_ONE:.*]] = arith.addf %[[RATIO_SQ]], %[[ONE]] fastmath<contract> : f32
// CHECK: %[[SQRT:.*]] = math.sqrt %[[RATIO_SQ_PLUS_ONE]] fastmath<contract> : f32
// CHECK: %[[ABS_OR_NAN:.*]] = arith.mulf %[[MAX]], %[[SQRT]] fastmath<contract> : f32
// CHECK: %[[IS_NAN:.*]] = arith.cmpf uno, %[[ABS_OR_NAN]], %[[ABS_OR_NAN]] fastmath<contract> : f32
// CHECK: %[[ABS:.*]] = arith.select %[[IS_NAN]], %[[MIN]], %[[ABS_OR_NAN]] : f32
// CHECK: %[[RESULT_REAL:.*]] = math.log %[[ABS]] fastmath<nnan,contract> : f32
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
// CHECK: %[[REAL_PLUS_ONE:.*]] = arith.addf %[[REAL]], %[[ONE]] fastmath<nnan,contract>  : f32
// CHECK: %[[ABS_REAL_PLUS_ONE:.*]] = math.absf %[[REAL_PLUS_ONE]] fastmath<nnan,contract>  : f32
// CHECK: %[[ABS_IMAG:.*]] = math.absf %[[IMAG]] fastmath<nnan,contract>  : f32
// CHECK: %[[MAX:.*]] = arith.maximumf %[[ABS_REAL_PLUS_ONE]], %[[ABS_IMAG]] fastmath<nnan,contract>  : f32
// CHECK: %[[MIN:.*]] = arith.minimumf %[[ABS_REAL_PLUS_ONE]], %[[ABS_IMAG]] fastmath<nnan,contract>  : f32
// CHECK: %[[CMPF:.*]] = arith.cmpf ogt, %[[REAL_PLUS_ONE]], %[[ABS_IMAG]] fastmath<nnan,contract>  : f32
// CHECK: %[[MAX_MINUS_ONE:.*]] = arith.subf %[[MAX]], %[[ONE]] fastmath<nnan,contract>  : f32
// CHECK: %[[SELECT:.*]] = arith.select %[[CMPF]], %[[REAL]], %[[MAX_MINUS_ONE]] : f32
// CHECK: %[[MIN_MAX_RATIO:.*]] = arith.divf %[[MIN]], %[[MAX]] fastmath<contract> : f32
// CHECK: %[[LOG_1:.*]] = math.log1p %[[SELECT]] fastmath<nnan,contract> : f32
// CHECK: %[[RATIO_SQ:.*]] = arith.mulf %[[MIN_MAX_RATIO]], %[[MIN_MAX_RATIO]] fastmath<contract> : f32
// CHECK: %[[LOG_SQ:.*]] = math.log1p %[[RATIO_SQ]] fastmath<contract> : f32
// CHECK: %[[HALF_LOG_SQ:.*]] = arith.mulf %cst, %[[LOG_SQ]] fastmath<contract> : f32
// CHECK: %[[R:.*]] = arith.addf %[[HALF_LOG_SQ]], %[[LOG_1]] fastmath<contract> : f32
// CHECK: %[[ISNAN:.*]] = arith.cmpf uno, %[[R]], %[[R]] fastmath<contract> : f32
// CHECK: %[[RESULT_REAL:.*]] = arith.select %[[ISNAN]], %[[MIN]], %[[R]] : f32
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
// CHECK: %[[LHS_IMAG:.*]] = complex.im %[[LHS]] : complex<f32>
// CHECK: %[[RHS_REAL:.*]] = complex.re %[[RHS]] : complex<f32>
// CHECK: %[[RHS_IMAG:.*]] = complex.im %[[RHS]] : complex<f32>
// CHECK: %[[LHS_REAL_TIMES_RHS_REAL:.*]] = arith.mulf %[[LHS_REAL]], %[[RHS_REAL]] fastmath<nnan,contract> : f32
// CHECK: %[[LHS_IMAG_TIMES_RHS_IMAG:.*]] = arith.mulf %[[LHS_IMAG]], %[[RHS_IMAG]] fastmath<nnan,contract> : f32
// CHECK: %[[REAL:.*]] = arith.subf %[[LHS_REAL_TIMES_RHS_REAL]], %[[LHS_IMAG_TIMES_RHS_IMAG]] fastmath<nnan,contract> : f32
// CHECK: %[[LHS_IMAG_TIMES_RHS_REAL:.*]] = arith.mulf %[[LHS_IMAG]], %[[RHS_REAL]] fastmath<nnan,contract> : f32
// CHECK: %[[LHS_REAL_TIMES_RHS_IMAG:.*]] = arith.mulf %[[LHS_REAL]], %[[RHS_IMAG]] fastmath<nnan,contract> : f32
// CHECK: %[[IMAG:.*]] = arith.addf %[[LHS_IMAG_TIMES_RHS_REAL]], %[[LHS_REAL_TIMES_RHS_IMAG]] fastmath<nnan,contract> : f32
// CHECK: %[[RESULT:.*]] = complex.create %[[REAL]], %[[IMAG]] : complex<f32>
// CHECK: return %[[RESULT]] : complex<f32>

// -----

// CHECK-LABEL: func @complex_atan2_with_fmf
func.func @complex_atan2_with_fmf(%lhs: complex<f32>,
                         %rhs: complex<f32>) -> complex<f32> {
  %atan2 = complex.atan2 %lhs, %rhs fastmath<nnan,contract> : complex<f32>
  return %atan2 : complex<f32>
}

// CHECK: %[[VAR0:.*]] = complex.re %arg1 : complex<f32>
// CHECK: %[[VAR2:.*]] = complex.im %arg1 : complex<f32>
// CHECK: %[[VAR4:.*]] = complex.re %arg1 : complex<f32>
// CHECK: %[[VAR6:.*]] = complex.im %arg1 : complex<f32>
// CHECK: %[[VAR8:.*]] = arith.mulf %[[VAR0]], %[[VAR4]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR10:.*]] = arith.mulf %[[VAR2]], %[[VAR6]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR12:.*]] = arith.subf %[[VAR8]], %[[VAR10]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR13:.*]] = arith.mulf %[[VAR2]], %[[VAR4]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR15:.*]] = arith.mulf %[[VAR0]], %[[VAR6]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR17:.*]] = arith.addf %[[VAR13]], %[[VAR15]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR89:.*]] = complex.create %[[VAR12]], %[[VAR17]] : complex<f32>
// CHECK: %[[VAR90:.*]] = complex.re %arg0 : complex<f32>
// CHECK: %[[VAR92:.*]] = complex.im %arg0 : complex<f32>
// CHECK: %[[VAR94:.*]] = complex.re %arg0 : complex<f32>
// CHECK: %[[VAR96:.*]] = complex.im %arg0 : complex<f32>
// CHECK: %[[VAR98:.*]] = arith.mulf %[[VAR90]], %[[VAR94]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR100:.*]] = arith.mulf %[[VAR92]], %[[VAR96]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR102:.*]] = arith.subf %[[VAR98]], %[[VAR100]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR103:.*]] = arith.mulf %[[VAR92]], %[[VAR94]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR105:.*]] = arith.mulf %[[VAR90]], %[[VAR96]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR107:.*]] = arith.addf %[[VAR103]], %[[VAR105]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR179:.*]] = complex.create %[[VAR102]], %[[VAR107]] : complex<f32>
// CHECK: %[[VAR180:.*]] = complex.re %[[VAR89]] : complex<f32>
// CHECK: %[[VAR181:.*]] = complex.re %[[VAR179]] : complex<f32>
// CHECK: %[[VAR182:.*]] = arith.addf %[[VAR180]], %[[VAR181]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR183:.*]] = complex.im %[[VAR89]] : complex<f32>
// CHECK: %[[VAR184:.*]] = complex.im %[[VAR179]] : complex<f32>
// CHECK: %[[VAR185:.*]] = arith.addf %[[VAR183]], %[[VAR184]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR186:.*]] = complex.create %[[VAR182]], %[[VAR185]] : complex<f32>
// CHECK: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[HALF:.*]] = arith.constant 5.000000e-01 : f32
// CHECK: %[[RE:.*]] = complex.re %[[VAR186]] : complex<f32>
// CHECK: %[[IM:.*]] = complex.im %[[VAR186]] : complex<f32>
// CHECK: %[[ONE:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[ABSRE:.*]] = math.absf %[[RE]] fastmath<nnan,contract> : f32
// CHECK: %[[ABSIM:.*]] = math.absf %[[IM]] fastmath<nnan,contract> : f32
// CHECK: %[[MAX:.*]] = arith.maximumf %[[ABSRE]], %[[ABSIM]] fastmath<nnan,contract> : f32
// CHECK: %[[MIN:.*]] = arith.minimumf %[[ABSRE]], %[[ABSIM]] fastmath<nnan,contract> : f32
// CHECK: %[[RATIO:.*]] = arith.divf %[[MIN]], %[[MAX]] fastmath<contract> : f32
// CHECK: %[[RATIO_SQ:.*]] = arith.mulf %[[RATIO]], %[[RATIO]] fastmath<contract> : f32
// CHECK: %[[RATIO_SQ_PLUS_ONE:.*]] = arith.addf %[[RATIO_SQ]], %[[ONE]] fastmath<contract> : f32
// CHECK: %[[QUARTER:.*]] = arith.constant 2.500000e-01 : f32
// CHECK: %[[SQRT_MAX:.*]] = math.sqrt %[[MAX]] fastmath<contract> : f32
// CHECK: %[[POW:.*]] = math.powf %[[RATIO_SQ_PLUS_ONE]], %[[QUARTER]] fastmath<contract> : f32
// CHECK: %[[SQRT_ABS_OR_NAN:.*]] = arith.mulf %[[SQRT_MAX]], %[[POW]] fastmath<contract> : f32
// CHECK: %[[IS_NAN:.*]] = arith.cmpf uno, %[[SQRT_ABS_OR_NAN]], %[[SQRT_ABS_OR_NAN]] fastmath<contract> : f32
// CHECK: %[[SQRT_ABS:.*]] = arith.select %[[IS_NAN]], %[[MIN]], %[[SQRT_ABS_OR_NAN]] : f32
// CHECK: %[[ARGARG:.*]] = math.atan2 %[[IM]], %[[RE]] fastmath<nnan,contract> : f32
// CHECK: %[[SQRTARG:.*]] = arith.mulf %[[ARGARG]], %[[HALF]] fastmath<nnan,contract> : f32
// CHECK: %[[COS:.*]] = math.cos %[[SQRTARG]] fastmath<nnan,contract> : f32
// CHECK: %[[SIN:.*]] = math.sin %[[SQRTARG]] fastmath<nnan,contract> : f32
// CHECK: %[[SIN_ZERO:.*]] = arith.cmpf oeq, %[[SIN]], %[[ZERO]] fastmath<nnan,contract> : f32
// CHECK: %[[RESULT_RE:.*]] = arith.mulf %[[SQRT_ABS]], %[[COS]] fastmath<nnan,contract> : f32
// CHECK: %[[RESULT_IM:.*]] = arith.mulf %[[SQRT_ABS]], %[[SIN]] fastmath<nnan,contract> : f32
// CHECK: %[[RESULT_IM2:.*]] = arith.select %[[SIN_ZERO]], %[[ZERO]], %[[RESULT_IM]] : f32
// CHECK: %[[INF:.*]] = arith.constant 0x7F800000 : f32
// CHECK: %[[NINF:.*]] = arith.constant 0xFF800000 : f32
// CHECK: %[[NAN:.*]] = arith.constant 0x7FC00000 : f32
// CHECK: %[[ABSIM:.*]] = math.absf %[[IM]] fastmath<nnan,contract> : f32
// CHECK: %[[ABSIMINF:.*]] = arith.cmpf oeq, %[[ABSIM]], %[[INF]] fastmath<nnan,contract> : f32
// CHECK: %[[ABSIMNOTINF:.*]] = arith.cmpf one, %[[ABSIM]], %[[INF]] fastmath<nnan,contract> : f32
// CHECK: %[[REINF:.*]] = arith.cmpf oeq, %[[RE]], %[[INF]] fastmath<nnan,contract> : f32
// CHECK: %[[RENINF:.*]] = arith.cmpf oeq, %[[RE]], %[[NINF]] fastmath<nnan,contract> : f32
// CHECK: %[[RESULT_RE_ZERO:.*]] = arith.andi %[[RENINF]], %[[ABSIMNOTINF]] : i1
// CHECK: %[[RESULT_RE2:.*]] = arith.select %[[RESULT_RE_ZERO]], %[[ZERO]], %[[RESULT_RE]] : f32
// CHECK: %[[RESUL_IM_INF:.*]] = arith.ori %[[ABSIMINF]], %[[REINF]] : i1
// CHECK: %[[RESULT_RE3:.*]] = arith.select %[[RESUL_IM_INF]], %[[INF]], %[[RESULT_RE2]] : f32
// CHECK: %[[INF_IM_SIGN:.*]] = math.copysign %[[INF]], %[[IM]] fastmath<nnan,contract> : f32
// CHECK: %[[RESULT_IM_NAN:.*]] = arith.cmpf uno, %[[SQRT_ABS]], %[[SQRT_ABS]] : f32
// CHECK: %[[RESULT_IM3:.*]] = arith.select %[[RESULT_IM_NAN]], %[[NAN]], %[[RESULT_IM2]] : f32
// CHECK: %[[RESULT_IM_INF:.*]] = arith.ori %[[ABSIMINF]], %[[RENINF]] : i1
// CHECK: %[[RESULT_IM4:.*]] = arith.select %[[RESULT_IM_INF]], %[[INF_IM_SIGN]], %[[RESULT_IM3]] : f32
// CHECK: %[[RESULT_ZERO:.*]] = arith.cmpf oeq, %[[SQRT_ABS]], %[[ZERO]] fastmath<nnan,contract> : f32
// CHECK: %[[RESULT_RE4:.*]] = arith.select %[[RESULT_ZERO]], %[[ZERO]], %[[RESULT_RE3]] : f32
// CHECK: %[[RESULT_IM5:.*]] = arith.select %[[RESULT_ZERO]], %[[ZERO]], %[[RESULT_IM4]] : f32
// CHECK: %[[VAR228:.*]] = complex.create %[[RESULT_RE4]], %[[RESULT_IM5]] : complex<f32>
// CHECK: %[[CST_10:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[CST_11:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[VAR229:.*]] = complex.create %[[CST_10]], %[[CST_11]] : complex<f32>
// CHECK: %[[VAR230:.*]] = complex.re %[[VAR229]] : complex<f32>
// CHECK: %[[VAR232:.*]] = complex.im %[[VAR229]] : complex<f32>
// CHECK: %[[VAR234:.*]] = complex.re %arg0 : complex<f32>
// CHECK: %[[VAR236:.*]] = complex.im %arg0 : complex<f32>
// CHECK: %[[VAR238:.*]] = arith.mulf %[[VAR230]], %[[VAR234]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR240:.*]] = arith.mulf %[[VAR232]], %[[VAR236]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR242:.*]] = arith.subf %[[VAR238]], %[[VAR240]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR243:.*]] = arith.mulf %[[VAR232]], %[[VAR234]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR245:.*]] = arith.mulf %[[VAR230]], %[[VAR236]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR247:.*]] = arith.addf %[[VAR243]], %[[VAR245]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR319:.*]] = complex.create %[[VAR242]], %[[VAR247]] : complex<f32>
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
// CHECK: %[[REAL:.*]] = complex.re %[[VAR415]] : complex<f32>
// CHECK: %[[IMAG:.*]] = complex.im %[[VAR415]] : complex<f32>
// CHECK: %[[ONE:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[ABS_REAL:.*]] = math.absf %[[REAL]] fastmath<nnan,contract> : f32
// CHECK: %[[ABS_IMAG:.*]] = math.absf %[[IMAG]] fastmath<nnan,contract> : f32
// CHECK: %[[MAX:.*]] = arith.maximumf %[[ABS_REAL]], %[[ABS_IMAG]] fastmath<nnan,contract> : f32
// CHECK: %[[MIN:.*]] = arith.minimumf %[[ABS_REAL]], %[[ABS_IMAG]] fastmath<nnan,contract> : f32
// CHECK: %[[RATIO:.*]] = arith.divf %[[MIN]], %[[MAX]] fastmath<contract> : f32
// CHECK: %[[RATIO_SQ:.*]] = arith.mulf %[[RATIO]], %[[RATIO]] fastmath<contract> : f32
// CHECK: %[[RATIO_SQ_PLUS_ONE:.*]] = arith.addf %[[RATIO_SQ]], %[[ONE]] fastmath<contract> : f32
// CHECK: %[[SQRT:.*]] = math.sqrt %[[RATIO_SQ_PLUS_ONE]] fastmath<contract> : f32
// CHECK: %[[ABS_OR_NAN:.*]] = arith.mulf %[[MAX]], %[[SQRT]] fastmath<contract> : f32
// CHECK: %[[IS_NAN:.*]] = arith.cmpf uno, %[[ABS_OR_NAN]], %[[ABS_OR_NAN]] fastmath<contract> : f32
// CHECK: %[[ABS:.*]] = arith.select %[[IS_NAN]], %[[MIN]], %[[ABS_OR_NAN]] : f32
// CHECK: %[[VAR436:.*]] = math.log %[[ABS]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR437:.*]] = complex.re %[[VAR415]] : complex<f32>
// CHECK: %[[VAR438:.*]] = complex.im %[[VAR415]] : complex<f32>
// CHECK: %[[VAR439:.*]] = math.atan2 %[[VAR438]], %[[VAR437]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR440:.*]] = complex.create %[[VAR436]], %[[VAR439]] : complex<f32>
// CHECK: %[[CST_21:.*]] = arith.constant -1.000000e+00 : f32
// CHECK: %[[VAR441:.*]] = complex.create %[[CST_10]], %[[CST_21]] : complex<f32>
// CHECK: %[[VAR442:.*]] = complex.re %[[VAR441]] : complex<f32>
// CHECK: %[[VAR444:.*]] = complex.im %[[VAR441]] : complex<f32>
// CHECK: %[[VAR446:.*]] = complex.re %[[VAR440]] : complex<f32>
// CHECK: %[[VAR448:.*]] = complex.im %[[VAR440]] : complex<f32>
// CHECK: %[[VAR450:.*]] = arith.mulf %[[VAR442]], %[[VAR446]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR452:.*]] = arith.mulf %[[VAR444]], %[[VAR448]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR454:.*]] = arith.subf %[[VAR450]], %[[VAR452]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR455:.*]] = arith.mulf %[[VAR444]], %[[VAR446]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR457:.*]] = arith.mulf %[[VAR442]], %[[VAR448]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR459:.*]] = arith.addf %[[VAR455]], %[[VAR457]] fastmath<nnan,contract> : f32
// CHECK: %[[VAR531:.*]] = complex.create %[[VAR454]], %[[VAR459]] : complex<f32>
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

// -----

// CHECK-LABEL: func @complex_sqrt_with_fmf
// CHECK-SAME: %[[ARG:.*]]: complex<f32>
func.func @complex_sqrt_with_fmf(%arg: complex<f32>) -> complex<f32> {
  %sqrt = complex.sqrt %arg fastmath<nnan,contract> : complex<f32>
  return %sqrt : complex<f32>
}

// CHECK: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[HALF:.*]] = arith.constant 5.000000e-01 : f32
// CHECK: %[[RE:.*]] = complex.re %[[ARG]] : complex<f32>
// CHECK: %[[IM:.*]] = complex.im %[[ARG]] : complex<f32>
// CHECK: %[[ONE:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[ABSRE:.*]] = math.absf %[[RE]] fastmath<nnan,contract> : f32
// CHECK: %[[ABSIM:.*]] = math.absf %[[IM]] fastmath<nnan,contract> : f32
// CHECK: %[[MAX:.*]] = arith.maximumf %[[ABSRE]], %[[ABSIM]] fastmath<nnan,contract> : f32
// CHECK: %[[MIN:.*]] = arith.minimumf %[[ABSRE]], %[[ABSIM]] fastmath<nnan,contract> : f32
// CHECK: %[[RATIO:.*]] = arith.divf %[[MIN]], %[[MAX]] fastmath<contract> : f32
// CHECK: %[[RATIO_SQ:.*]] = arith.mulf %[[RATIO]], %[[RATIO]] fastmath<contract> : f32
// CHECK: %[[RATIO_SQ_PLUS_ONE:.*]] = arith.addf %[[RATIO_SQ]], %[[ONE]] fastmath<contract> : f32
// CHECK: %[[QUARTER:.*]] = arith.constant 2.500000e-01 : f32
// CHECK: %[[SQRT_MAX:.*]] = math.sqrt %[[MAX]] fastmath<contract> : f32
// CHECK: %[[POW:.*]] = math.powf %[[RATIO_SQ_PLUS_ONE]], %[[QUARTER]] fastmath<contract> : f32
// CHECK: %[[SQRT_ABS_OR_NAN:.*]] = arith.mulf %[[SQRT_MAX]], %[[POW]] fastmath<contract> : f32
// CHECK: %[[IS_NAN:.*]] = arith.cmpf uno, %[[SQRT_ABS_OR_NAN]], %[[SQRT_ABS_OR_NAN]] fastmath<contract> : f32
// CHECK: %[[SQRT_ABS:.*]] = arith.select %[[IS_NAN]], %[[MIN]], %[[SQRT_ABS_OR_NAN]] : f32
// CHECK: %[[ARGARG:.*]] = math.atan2 %[[IM]], %[[RE]] fastmath<nnan,contract> : f32
// CHECK: %[[SQRTARG:.*]] = arith.mulf %[[ARGARG]], %[[HALF]] fastmath<nnan,contract> : f32
// CHECK: %[[COS:.*]] = math.cos %[[SQRTARG]] fastmath<nnan,contract> : f32
// CHECK: %[[SIN:.*]] = math.sin %[[SQRTARG]] fastmath<nnan,contract> : f32
// CHECK: %[[SIN_ZERO:.*]] = arith.cmpf oeq, %[[SIN]], %[[ZERO]] fastmath<nnan,contract> : f32
// CHECK: %[[RESULT_RE:.*]] = arith.mulf %[[SQRT_ABS]], %[[COS]] fastmath<nnan,contract> : f32
// CHECK: %[[RESULT_IM:.*]] = arith.mulf %[[SQRT_ABS]], %[[SIN]] fastmath<nnan,contract> : f32
// CHECK: %[[RESULT_IM2:.*]] = arith.select %[[SIN_ZERO]], %[[ZERO]], %[[RESULT_IM]] : f32
// CHECK: %[[INF:.*]] = arith.constant 0x7F800000 : f32
// CHECK: %[[NINF:.*]] = arith.constant 0xFF800000 : f32
// CHECK: %[[NAN:.*]] = arith.constant 0x7FC00000 : f32
// CHECK: %[[ABSIM:.*]] = math.absf %[[IM]] fastmath<nnan,contract> : f32
// CHECK: %[[ABSIMINF:.*]] = arith.cmpf oeq, %[[ABSIM]], %[[INF]] fastmath<nnan,contract> : f32
// CHECK: %[[ABSIMNOTINF:.*]] = arith.cmpf one, %[[ABSIM]], %[[INF]] fastmath<nnan,contract> : f32
// CHECK: %[[REINF:.*]] = arith.cmpf oeq, %[[RE]], %[[INF]] fastmath<nnan,contract> : f32
// CHECK: %[[RENINF:.*]] = arith.cmpf oeq, %[[RE]], %[[NINF]] fastmath<nnan,contract> : f32
// CHECK: %[[RESULT_RE_ZERO:.*]] = arith.andi %[[RENINF]], %[[ABSIMNOTINF]] : i1
// CHECK: %[[RESULT_RE2:.*]] = arith.select %[[RESULT_RE_ZERO]], %[[ZERO]], %[[RESULT_RE]] : f32
// CHECK: %[[RESUL_IM_INF:.*]] = arith.ori %[[ABSIMINF]], %[[REINF]] : i1
// CHECK: %[[RESULT_RE3:.*]] = arith.select %[[RESUL_IM_INF]], %[[INF]], %[[RESULT_RE2]] : f32
// CHECK: %[[INF_IM_SIGN:.*]] = math.copysign %[[INF]], %[[IM]] fastmath<nnan,contract> : f32
// CHECK: %[[RESULT_IM_NAN:.*]] = arith.cmpf uno, %[[SQRT_ABS]], %[[SQRT_ABS]] : f32
// CHECK: %[[RESULT_IM3:.*]] = arith.select %[[RESULT_IM_NAN]], %[[NAN]], %[[RESULT_IM2]] : f32
// CHECK: %[[RESULT_IM_INF:.*]] = arith.ori %[[ABSIMINF]], %[[RENINF]] : i1
// CHECK: %[[RESULT_IM4:.*]] = arith.select %[[RESULT_IM_INF]], %[[INF_IM_SIGN]], %[[RESULT_IM3]] : f32
// CHECK: %[[RESULT_ZERO:.*]] = arith.cmpf oeq, %[[SQRT_ABS]], %[[ZERO]] fastmath<nnan,contract> : f32
// CHECK: %[[RESULT_RE4:.*]] = arith.select %[[RESULT_ZERO]], %[[ZERO]], %[[RESULT_RE3]] : f32
// CHECK: %[[RESULT_IM5:.*]] = arith.select %[[RESULT_ZERO]], %[[ZERO]], %[[RESULT_IM4]] : f32
// CHECK: %[[RESULT:.*]] = complex.create %[[RESULT_RE4]], %[[RESULT_IM5]] : complex<f32>
// CHECK: return %[[RESULT]] : complex<f32>

// -----

// CHECK-LABEL: func @complex_cos_with_fmf
// CHECK-SAME: %[[ARG:.*]]: complex<f32>
func.func @complex_cos_with_fmf(%arg: complex<f32>) -> complex<f32> {
  %cos = complex.cos %arg fastmath<nnan,contract> : complex<f32>
  return %cos : complex<f32>
}
// CHECK-DAG: %[[REAL:.*]] = complex.re %[[ARG]]
// CHECK-DAG: %[[IMAG:.*]] = complex.im %[[ARG]]
// CHECK-DAG: %[[HALF:.*]] = arith.constant 5.000000e-01 : f32
// CHECK-DAG: %[[EXP:.*]] = math.exp %[[IMAG]] fastmath<nnan,contract> : f32
// CHECK-DAG: %[[HALF_EXP:.*]] = arith.mulf %[[HALF]], %[[EXP]] fastmath<nnan,contract>
// CHECK-DAG: %[[HALF_REXP:.*]] = arith.divf %[[HALF]], %[[EXP]] fastmath<nnan,contract>
// CHECK-DAG: %[[SIN:.*]] = math.sin %[[REAL]] fastmath<nnan,contract> : f32
// CHECK-DAG: %[[COS:.*]] = math.cos %[[REAL]] fastmath<nnan,contract> : f32
// CHECK-DAG: %[[EXP_SUM:.*]] = arith.addf %[[HALF_REXP]], %[[HALF_EXP]] fastmath<nnan,contract>
// CHECK-DAG: %[[RESULT_REAL:.*]] = arith.mulf %[[EXP_SUM]], %[[COS]] fastmath<nnan,contract>
// CHECK-DAG: %[[EXP_DIFF:.*]] = arith.subf %[[HALF_REXP]], %[[HALF_EXP]] fastmath<nnan,contract>
// CHECK-DAG: %[[RESULT_IMAG:.*]] = arith.mulf %[[EXP_DIFF]], %[[SIN]] fastmath<nnan,contract>
// CHECK-DAG: %[[RESULT:.*]] = complex.create %[[RESULT_REAL]], %[[RESULT_IMAG]] : complex<f32>
// CHECK:     return %[[RESULT]]

// -----

// CHECK-LABEL: func @complex_sin_with_fmf
// CHECK-SAME: %[[ARG:.*]]: complex<f32>
func.func @complex_sin_with_fmf(%arg: complex<f32>) -> complex<f32> {
  %cos = complex.sin %arg fastmath<nnan,contract> : complex<f32>
  return %cos : complex<f32>
}
// CHECK-DAG: %[[REAL:.*]] = complex.re %[[ARG]]
// CHECK-DAG: %[[IMAG:.*]] = complex.im %[[ARG]]
// CHECK-DAG: %[[HALF:.*]] = arith.constant 5.000000e-01 : f32
// CHECK-DAG: %[[EXP:.*]] = math.exp %[[IMAG]] fastmath<nnan,contract> : f32
// CHECK-DAG: %[[HALF_EXP:.*]] = arith.mulf %[[HALF]], %[[EXP]] fastmath<nnan,contract>
// CHECK-DAG: %[[HALF_REXP:.*]] = arith.divf %[[HALF]], %[[EXP]] fastmath<nnan,contract>
// CHECK-DAG: %[[SIN:.*]] = math.sin %[[REAL]] fastmath<nnan,contract> : f32
// CHECK-DAG: %[[COS:.*]] = math.cos %[[REAL]] fastmath<nnan,contract> : f32
// CHECK-DAG: %[[EXP_SUM:.*]] = arith.addf %[[HALF_EXP]], %[[HALF_REXP]] fastmath<nnan,contract>
// CHECK-DAG: %[[RESULT_REAL:.*]] = arith.mulf %[[EXP_SUM]], %[[SIN]] fastmath<nnan,contract>
// CHECK-DAG: %[[EXP_DIFF:.*]] = arith.subf %[[HALF_EXP]], %[[HALF_REXP]] fastmath<nnan,contract>
// CHECK-DAG: %[[RESULT_IMAG:.*]] = arith.mulf %[[EXP_DIFF]], %[[COS]] fastmath<nnan,contract>
// CHECK-DAG: %[[RESULT:.*]] = complex.create %[[RESULT_REAL]], %[[RESULT_IMAG]] : complex<f32>
// CHECK:     return %[[RESULT]]

// -----

// CHECK-LABEL: func @complex_sign_with_fmf
// CHECK-SAME: %[[ARG:.*]]: complex<f32>
func.func @complex_sign_with_fmf(%arg: complex<f32>) -> complex<f32> {
  %sign = complex.sign %arg fastmath<nnan,contract> : complex<f32>
  return %sign : complex<f32>
}

// CHECK: %[[REAL:.*]] = complex.re %[[ARG]] : complex<f32>
// CHECK: %[[IMAG:.*]] = complex.im %[[ARG]] : complex<f32>
// CHECK: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[REAL_IS_ZERO:.*]] = arith.cmpf oeq, %[[REAL]], %[[ZERO]] : f32
// CHECK: %[[IMAG_IS_ZERO:.*]] = arith.cmpf oeq, %[[IMAG]], %[[ZERO]] : f32
// CHECK: %[[IS_ZERO:.*]] = arith.andi %[[REAL_IS_ZERO]], %[[IMAG_IS_ZERO]] : i1
// CHECK: %[[REAL2:.*]] = complex.re %[[ARG]] : complex<f32>
// CHECK: %[[IMAG2:.*]] = complex.im %[[ARG]] : complex<f32>
// CHECK: %[[ONE:.*]] = arith.constant 1.000000e+00 : f32
// CHECK: %[[ABS_REAL:.*]] = math.absf %[[REAL2]] fastmath<nnan,contract> : f32
// CHECK: %[[ABS_IMAG:.*]] = math.absf %[[IMAG2]] fastmath<nnan,contract> : f32
// CHECK: %[[MAX:.*]] = arith.maximumf %[[ABS_REAL]], %[[ABS_IMAG]] fastmath<nnan,contract> : f32
// CHECK: %[[MIN:.*]] = arith.minimumf %[[ABS_REAL]], %[[ABS_IMAG]] fastmath<nnan,contract> : f32
// CHECK: %[[RATIO:.*]] = arith.divf %[[MIN]], %[[MAX]] fastmath<contract> : f32
// CHECK: %[[RATIO_SQ:.*]] = arith.mulf %[[RATIO]], %[[RATIO]] fastmath<contract> : f32
// CHECK: %[[RATIO_SQ_PLUS_ONE:.*]] = arith.addf %[[RATIO_SQ]], %[[ONE]] fastmath<contract> : f32
// CHECK: %[[SQRT:.*]] = math.sqrt %[[RATIO_SQ_PLUS_ONE]] fastmath<contract> : f32
// CHECK: %[[ABS_OR_NAN:.*]] = arith.mulf %[[MAX]], %[[SQRT]] fastmath<contract> : f32
// CHECK: %[[IS_NAN:.*]] = arith.cmpf uno, %[[ABS_OR_NAN]], %[[ABS_OR_NAN]] fastmath<contract> : f32
// CHECK: %[[ABS:.*]] = arith.select %[[IS_NAN]], %[[MIN]], %[[ABS_OR_NAN]] : f32
// CHECK: %[[REAL_SIGN:.*]] = arith.divf %[[REAL]], %[[ABS]] fastmath<nnan,contract> : f32
// CHECK: %[[IMAG_SIGN:.*]] = arith.divf %[[IMAG]], %[[ABS]] fastmath<nnan,contract> : f32
// CHECK: %[[SIGN:.*]] = complex.create %[[REAL_SIGN]], %[[IMAG_SIGN]] : complex<f32>
// CHECK: %[[RESULT:.*]] = arith.select %[[IS_ZERO]], %[[ARG]], %[[SIGN]] : complex<f32>
// CHECK: return %[[RESULT]] : complex<f32>

// -----

// CHECK-LABEL: func @complex_tan_with_fmf
// CHECK-SAME: %[[ARG:.*]]: complex<f32>
func.func @complex_tan_with_fmf(%arg: complex<f32>) -> complex<f32> {
  %tan = complex.tan %arg fastmath<nnan,contract> : complex<f32>
  return %tan : complex<f32>
}

// CHECK: %[[IMAG:.*]] = complex.re %[[ARG]] : complex<f32>
// CHECK: %[[V0:.*]] = complex.im %[[ARG]] : complex<f32>
// CHECK: %[[NEG_ONE:.*]] = arith.constant -1.000000e+00 : f32
// CHECK: %[[REAL:.*]] = arith.mulf %[[V0]], %cst fastmath<nnan,contract> : f32
// CHECK: %[[INF:.*]] = arith.constant 0x7F800000 : f32
// CHECK: %[[FOUR:.*]] = arith.constant 4.000000e+00 : f32
// CHECK: %[[TWO_REAL:.*]] = arith.addf %[[REAL]], %[[REAL]] fastmath<nnan,contract> : f32
// CHECK: %[[NEG_TWO_REAL:.*]] = arith.mulf %[[NEG_ONE]], %[[TWO_REAL]] fastmath<nnan,contract> : f32
// CHECK: %[[EXPM1:.*]] = math.expm1 %[[TWO_REAL]] fastmath<nnan,contract> : f32
// CHECK: %[[EXPM1_2:.*]] = math.expm1 %[[NEG_TWO_REAL]] fastmath<nnan,contract> : f32
// CHECK: %[[REAL_NUM:.*]] = arith.subf %[[EXPM1]], %[[EXPM1_2]] fastmath<nnan,contract> : f32
// CHECK: %[[COS:.*]] = math.cos %[[IMAG]] fastmath<nnan,contract> : f32
// CHECK: %[[COS_SQ:.*]] = arith.mulf %[[COS]], %[[COS]] fastmath<nnan,contract> : f32
// CHECK: %[[FOUR_COS_SQ:.*]] = arith.mulf %[[COS_SQ]], %[[FOUR]] fastmath<nnan,contract> : f32
// CHECK: %[[SIN:.*]] = math.sin %[[IMAG]] fastmath<nnan,contract> : f32
// CHECK: %[[MUL:.*]] = arith.mulf %[[COS]], %[[SIN]] fastmath<nnan,contract> : f32
// CHECK: %[[IMAG_NUM:.*]] = arith.mulf %[[FOUR]], %[[MUL]] fastmath<nnan,contract> : f32
// CHECK: %[[ADD:.*]] = arith.addf %[[EXPM1]], %[[EXPM1_2]] fastmath<nnan,contract> : f32
// CHECK: %[[DENOM:.*]] = arith.addf %[[ADD]], %[[FOUR_COS_SQ]] fastmath<nnan,contract> : f32
// CHECK: %[[IS_INF:.*]] = arith.cmpf oeq, %[[ADD]], %[[INF]] fastmath<nnan,contract> : f32
// CHECK: %[[LIMIT:.*]] = math.copysign %[[NEG_ONE]], %[[REAL]] fastmath<nnan,contract> : f32
// CHECK: %[[RESULT_REAL:.*]] = arith.divf %[[REAL_NUM]], %[[DENOM]] fastmath<nnan,contract> : f32
// CHECK: %[[RESULT_REAL2:.*]] = arith.select %[[IS_INF]], %[[LIMIT]], %[[RESULT_REAL]] : f32
// CHECK: %[[RESULT_IMAG:.*]] = arith.divf %[[IMAG_NUM]], %[[DENOM]] fastmath<nnan,contract> : f32
// CHECK: %[[ABS_REAL:.*]] = math.absf %[[REAL]] fastmath<nnan,contract> : f32
// CHECK: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[NAN:.*]] = arith.constant 0x7FC00000 : f32
// CHECK: %[[ABS_REAL_INF:.*]] = arith.cmpf oeq, %[[ABS_REAL]], %[[INF]] fastmath<nnan,contract> : f32
// CHECK: %[[IMAG_ZERO:.*]] = arith.cmpf oeq, %[[IMAG]], %[[ZERO]] fastmath<nnan,contract> : f32
// CHECK: %true = arith.constant true
// CHECK: %[[ABS_REAL_NOT_INF:.*]] = arith.xori %[[ABS_REAL_INF]], %true : i1
// CHECK: %[[IMAG_IS_NAN:.*]] = arith.cmpf uno, %[[IMAG_NUM]], %[[IMAG_NUM]] fastmath<nnan,contract> : f32
// CHECK: %[[REAL_IS_NAN:.*]] = arith.andi %[[IMAG_IS_NAN]], %[[ABS_REAL_NOT_INF]] : i1
// CHECK: %[[AND:.*]] = arith.andi %[[ABS_REAL_INF]], %[[IMAG_IS_NAN]] : i1
// CHECK: %[[IMAG_IS_NAN2:.*]] = arith.ori %[[IMAG_ZERO]], %[[AND]] : i1
// CHECK: %[[RESULT_REAL3:.*]] = arith.select %[[REAL_IS_NAN]], %[[NAN]], %[[RESULT_REAL2]] : f32
// CHECK: %[[RESULT_REAL:.*]] = arith.select %[[IMAG_IS_NAN2]], %[[ZERO]], %[[RESULT_IMAG]] : f32
// CHECK: %[[RESULT_IMAG:.*]] = arith.mulf %[[RESULT_REAL3]], %[[NEG_ONE]] fastmath<nnan,contract> : f32
// CHECK: %[[RESULT:.*]] = complex.create %[[RESULT_REAL]], %[[RESULT_IMAG]] : complex<f32>
// CHECK: return %[[RESULT]] : complex<f32>

// -----

// CHECK-LABEL: func @complex_tanh_with_fmf
// CHECK-SAME: %[[ARG:.*]]: complex<f32>
func.func @complex_tanh_with_fmf(%arg: complex<f32>) -> complex<f32> {
  %tanh = complex.tanh %arg fastmath<nnan,contract> : complex<f32>
  return %tanh : complex<f32>
}

// CHECK: %[[REAL:.*]] = complex.re %[[ARG]] : complex<f32>
// CHECK: %[[IMAG:.*]] = complex.im %[[ARG]] : complex<f32>
// CHECK: %[[NEG_ONE:.*]] = arith.constant -1.000000e+00 : f32
// CHECK: %[[INF:.*]] = arith.constant 0x7F800000 : f32
// CHECK: %[[FOUR:.*]] = arith.constant 4.000000e+00 : f32
// CHECK: %[[TWO_REAL:.*]] = arith.addf %[[REAL]], %[[REAL]] fastmath<nnan,contract> : f32
// CHECK: %[[NEG_TWO_REAL:.*]] = arith.mulf %[[NEG_ONE]], %[[TWO_REAL]] fastmath<nnan,contract> : f32
// CHECK: %[[EXPM1:.*]] = math.expm1 %[[TWO_REAL]] fastmath<nnan,contract> : f32
// CHECK: %[[EXPM1_2:.*]] = math.expm1 %[[NEG_TWO_REAL]] fastmath<nnan,contract> : f32
// CHECK: %[[REAL_NUM:.*]] = arith.subf %[[EXPM1]], %[[EXPM1_2]] fastmath<nnan,contract> : f32
// CHECK: %[[COS:.*]] = math.cos %[[IMAG]] fastmath<nnan,contract> : f32
// CHECK: %[[COS_SQ:.*]] = arith.mulf %[[COS]], %[[COS]] fastmath<nnan,contract> : f32
// CHECK: %[[FOUR_COS_SQ:.*]] = arith.mulf %[[COS_SQ]], %[[FOUR]] fastmath<nnan,contract> : f32
// CHECK: %[[SIN:.*]] = math.sin %[[IMAG]] fastmath<nnan,contract> : f32
// CHECK: %[[MUL:.*]] = arith.mulf %[[COS]], %[[SIN]] fastmath<nnan,contract> : f32
// CHECK: %[[IMAG_NUM:.*]] = arith.mulf %[[FOUR]], %[[MUL]] fastmath<nnan,contract> : f32
// CHECK: %[[ADD:.*]] = arith.addf %[[EXPM1]], %[[EXPM1_2]] fastmath<nnan,contract> : f32
// CHECK: %[[DENOM:.*]] = arith.addf %[[ADD]], %[[FOUR_COS_SQ]] fastmath<nnan,contract> : f32
// CHECK: %[[IS_INF:.*]] = arith.cmpf oeq, %[[ADD]], %[[INF]] fastmath<nnan,contract> : f32
// CHECK: %[[LIMIT:.*]] = math.copysign %[[NEG_ONE]], %[[REAL]] fastmath<nnan,contract> : f32
// CHECK: %[[RESULT_REAL:.*]] = arith.divf %[[REAL_NUM]], %[[DENOM]] fastmath<nnan,contract> : f32
// CHECK: %[[RESULT_REAL2:.*]] = arith.select %[[IS_INF]], %[[LIMIT]], %[[RESULT_REAL]] : f32
// CHECK: %[[RESULT_IMAG:.*]] = arith.divf %[[IMAG_NUM]], %[[DENOM]] fastmath<nnan,contract> : f32
// CHECK: %[[ABS_REAL:.*]] = math.absf %[[REAL]] fastmath<nnan,contract> : f32
// CHECK: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[NAN:.*]] = arith.constant 0x7FC00000 : f32
// CHECK: %[[ABS_REAL_INF:.*]] = arith.cmpf oeq, %[[ABS_REAL]], %[[INF]] fastmath<nnan,contract> : f32
// CHECK: %[[IMAG_ZERO:.*]] = arith.cmpf oeq, %[[IMAG]], %[[ZERO]] fastmath<nnan,contract> : f32
// CHECK: %true = arith.constant true
// CHECK: %[[ABS_REAL_NOT_INF:.*]] = arith.xori %[[ABS_REAL_INF]], %true : i1
// CHECK: %[[IMAG_IS_NAN:.*]] = arith.cmpf uno, %[[IMAG_NUM]], %[[IMAG_NUM]] fastmath<nnan,contract> : f32
// CHECK: %[[REAL_IS_NAN:.*]] = arith.andi %[[IMAG_IS_NAN]], %[[ABS_REAL_NOT_INF]] : i1
// CHECK: %[[AND:.*]] = arith.andi %[[ABS_REAL_INF]], %[[IMAG_IS_NAN]] : i1
// CHECK: %[[IMAG_IS_NAN2:.*]] = arith.ori %[[IMAG_ZERO]], %[[AND]] : i1
// CHECK: %[[RESULT_REAL3:.*]] = arith.select %[[REAL_IS_NAN]], %[[NAN]], %[[RESULT_REAL2]] : f32
// CHECK: %[[RESULT_IMAG2:.*]] = arith.select %[[IMAG_IS_NAN2]], %[[ZERO]], %[[RESULT_IMAG]] : f32
// CHECK: %[[RESULT:.*]] = complex.create %[[RESULT_REAL3]], %[[RESULT_IMAG2]] : complex<f32>
// CHECK: return %[[RESULT]] : complex<f32>

// -----

// CHECK-LABEL: func @complex_tanh_nnan_ninf
// CHECK-SAME: %[[ARG:.*]]: complex<f32>
func.func @complex_tanh_nnan_ninf(%arg: complex<f32>) -> complex<f32> {
  %tanh = complex.tanh %arg fastmath<nnan,ninf> : complex<f32>
  return %tanh : complex<f32>
}

// CHECK-COUNT-1: arith.select
// CHECK-NOT: arith.select

// -----

// CHECK-LABEL:   func.func @complex_angle_with_fmf
// CHECK-SAME: %[[ARG:.*]]: complex<f32>
func.func @complex_angle_with_fmf(%arg: complex<f32>) -> f32 {
  %angle = complex.angle %arg fastmath<nnan,contract> : complex<f32>
  return %angle : f32
}
// CHECK: %[[REAL:.*]] = complex.re %[[ARG]] : complex<f32>
// CHECK: %[[IMAG:.*]] = complex.im %[[ARG]] : complex<f32>
// CHECK: %[[RESULT:.*]] = math.atan2 %[[IMAG]], %[[REAL]] fastmath<nnan,contract> : f32
// CHECK: return %[[RESULT]] : f32
