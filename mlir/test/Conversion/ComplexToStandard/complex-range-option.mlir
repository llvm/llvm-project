// RUN: mlir-opt %s -convert-complex-to-standard=complex-range=improved | FileCheck %s --check-prefix=DIV-SMITH
// RUN: mlir-opt %s -convert-complex-to-standard=complex-range=basic | FileCheck %s --check-prefix=DIV-ALGEBRAIC
// RUN: mlir-opt %s -convert-complex-to-standard=complex-range=none | FileCheck %s --check-prefix=DIV-ALGEBRAIC


func.func @complex_div(%lhs: complex<f32>, %rhs: complex<f32>) -> complex<f32> {
  %div = complex.div %lhs, %rhs : complex<f32>
  return %div : complex<f32>
}
// DIV-SMITH-LABEL: func @complex_div
// DIV-SMITH-SAME:    %[[LHS:.*]]: complex<f32>, %[[RHS:.*]]: complex<f32>

// DIV-SMITH: %[[LHS_REAL:.*]] = complex.re %[[LHS]] : complex<f32>
// DIV-SMITH: %[[LHS_IMAG:.*]] = complex.im %[[LHS]] : complex<f32>
// DIV-SMITH: %[[RHS_REAL:.*]] = complex.re %[[RHS]] : complex<f32>
// DIV-SMITH: %[[RHS_IMAG:.*]] = complex.im %[[RHS]] : complex<f32>

// DIV-SMITH: %[[RHS_REAL_IMAG_RATIO:.*]] = arith.divf %[[RHS_REAL]], %[[RHS_IMAG]] : f32
// DIV-SMITH: %[[RHS_REAL_TIMES_RHS_REAL_IMAG_RATIO:.*]] = arith.mulf %[[RHS_REAL_IMAG_RATIO]], %[[RHS_REAL]] : f32
// DIV-SMITH: %[[RHS_REAL_IMAG_DENOM:.*]] = arith.addf %[[RHS_IMAG]], %[[RHS_REAL_TIMES_RHS_REAL_IMAG_RATIO]] : f32
// DIV-SMITH: %[[LHS_REAL_TIMES_RHS_REAL_IMAG_RATIO:.*]] = arith.mulf %[[LHS_REAL]], %[[RHS_REAL_IMAG_RATIO]] : f32
// DIV-SMITH: %[[REAL_NUMERATOR_1:.*]] = arith.addf %[[LHS_REAL_TIMES_RHS_REAL_IMAG_RATIO]], %[[LHS_IMAG]] : f32
// DIV-SMITH: %[[RESULT_REAL_1:.*]] = arith.divf %[[REAL_NUMERATOR_1]], %[[RHS_REAL_IMAG_DENOM]] : f32
// DIV-SMITH: %[[LHS_IMAG_TIMES_RHS_REAL_IMAG_RATIO:.*]] = arith.mulf %[[LHS_IMAG]], %[[RHS_REAL_IMAG_RATIO]] : f32
// DIV-SMITH: %[[IMAG_NUMERATOR_1:.*]] = arith.subf %[[LHS_IMAG_TIMES_RHS_REAL_IMAG_RATIO]], %[[LHS_REAL]] : f32
// DIV-SMITH: %[[RESULT_IMAG_1:.*]] = arith.divf %[[IMAG_NUMERATOR_1]], %[[RHS_REAL_IMAG_DENOM]] : f32

// DIV-SMITH: %[[RHS_IMAG_REAL_RATIO:.*]] = arith.divf %[[RHS_IMAG]], %[[RHS_REAL]] : f32
// DIV-SMITH: %[[RHS_IMAG_TIMES_RHS_IMAG_REAL_RATIO:.*]] = arith.mulf %[[RHS_IMAG_REAL_RATIO]], %[[RHS_IMAG]] : f32
// DIV-SMITH: %[[RHS_IMAG_REAL_DENOM:.*]] = arith.addf %[[RHS_REAL]], %[[RHS_IMAG_TIMES_RHS_IMAG_REAL_RATIO]] : f32
// DIV-SMITH: %[[LHS_IMAG_TIMES_RHS_IMAG_REAL_RATIO:.*]] = arith.mulf %[[LHS_IMAG]], %[[RHS_IMAG_REAL_RATIO]] : f32
// DIV-SMITH: %[[REAL_NUMERATOR_2:.*]] = arith.addf %[[LHS_REAL]], %[[LHS_IMAG_TIMES_RHS_IMAG_REAL_RATIO]] : f32
// DIV-SMITH: %[[RESULT_REAL_2:.*]] = arith.divf %[[REAL_NUMERATOR_2]], %[[RHS_IMAG_REAL_DENOM]] : f32
// DIV-SMITH: %[[LHS_REAL_TIMES_RHS_IMAG_REAL_RATIO:.*]] = arith.mulf %[[LHS_REAL]], %[[RHS_IMAG_REAL_RATIO]] : f32
// DIV-SMITH: %[[IMAG_NUMERATOR_2:.*]] = arith.subf %[[LHS_IMAG]], %[[LHS_REAL_TIMES_RHS_IMAG_REAL_RATIO]] : f32
// DIV-SMITH: %[[RESULT_IMAG_2:.*]] = arith.divf %[[IMAG_NUMERATOR_2]], %[[RHS_IMAG_REAL_DENOM]] : f32

// Case 1. Zero denominator, numerator contains at most one NaN value.
// DIV-SMITH: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// DIV-SMITH: %[[RHS_REAL_ABS:.*]] = math.absf %[[RHS_REAL]] : f32
// DIV-SMITH: %[[RHS_REAL_ABS_IS_ZERO:.*]] = arith.cmpf oeq, %[[RHS_REAL_ABS]], %[[ZERO]] : f32
// DIV-SMITH: %[[RHS_IMAG_ABS:.*]] = math.absf %[[RHS_IMAG]] : f32
// DIV-SMITH: %[[RHS_IMAG_ABS_IS_ZERO:.*]] = arith.cmpf oeq, %[[RHS_IMAG_ABS]], %[[ZERO]] : f32
// DIV-SMITH: %[[LHS_REAL_IS_NOT_NAN:.*]] = arith.cmpf ord, %[[LHS_REAL]], %[[ZERO]] : f32
// DIV-SMITH: %[[LHS_IMAG_IS_NOT_NAN:.*]] = arith.cmpf ord, %[[LHS_IMAG]], %[[ZERO]] : f32
// DIV-SMITH: %[[LHS_CONTAINS_NOT_NAN_VALUE:.*]] = arith.ori %[[LHS_REAL_IS_NOT_NAN]], %[[LHS_IMAG_IS_NOT_NAN]] : i1
// DIV-SMITH: %[[RHS_IS_ZERO:.*]] = arith.andi %[[RHS_REAL_ABS_IS_ZERO]], %[[RHS_IMAG_ABS_IS_ZERO]] : i1
// DIV-SMITH: %[[RESULT_IS_INFINITY:.*]] = arith.andi %[[LHS_CONTAINS_NOT_NAN_VALUE]], %[[RHS_IS_ZERO]] : i1
// DIV-SMITH: %[[INF:.*]] = arith.constant 0x7F800000 : f32
// DIV-SMITH: %[[INF_WITH_SIGN_OF_RHS_REAL:.*]] = math.copysign %[[INF]], %[[RHS_REAL]] : f32
// DIV-SMITH: %[[INFINITY_RESULT_REAL:.*]] = arith.mulf %[[INF_WITH_SIGN_OF_RHS_REAL]], %[[LHS_REAL]] : f32
// DIV-SMITH: %[[INFINITY_RESULT_IMAG:.*]] = arith.mulf %[[INF_WITH_SIGN_OF_RHS_REAL]], %[[LHS_IMAG]] : f32

// Case 2. Infinite numerator, finite denominator.
// DIV-SMITH: %[[RHS_REAL_FINITE:.*]] = arith.cmpf one, %[[RHS_REAL_ABS]], %[[INF]] : f32
// DIV-SMITH: %[[RHS_IMAG_FINITE:.*]] = arith.cmpf one, %[[RHS_IMAG_ABS]], %[[INF]] : f32
// DIV-SMITH: %[[RHS_IS_FINITE:.*]] = arith.andi %[[RHS_REAL_FINITE]], %[[RHS_IMAG_FINITE]] : i1
// DIV-SMITH: %[[LHS_REAL_ABS:.*]] = math.absf %[[LHS_REAL]] : f32
// DIV-SMITH: %[[LHS_REAL_INFINITE:.*]] = arith.cmpf oeq, %[[LHS_REAL_ABS]], %[[INF]] : f32
// DIV-SMITH: %[[LHS_IMAG_ABS:.*]] = math.absf %[[LHS_IMAG]] : f32
// DIV-SMITH: %[[LHS_IMAG_INFINITE:.*]] = arith.cmpf oeq, %[[LHS_IMAG_ABS]], %[[INF]] : f32
// DIV-SMITH: %[[LHS_IS_INFINITE:.*]] = arith.ori %[[LHS_REAL_INFINITE]], %[[LHS_IMAG_INFINITE]] : i1
// DIV-SMITH: %[[INF_NUM_FINITE_DENOM:.*]] = arith.andi %[[LHS_IS_INFINITE]], %[[RHS_IS_FINITE]] : i1
// DIV-SMITH: %[[ONE:.*]] = arith.constant 1.000000e+00 : f32
// DIV-SMITH: %[[LHS_REAL_IS_INF:.*]] = arith.select %[[LHS_REAL_INFINITE]], %[[ONE]], %[[ZERO]] : f32
// DIV-SMITH: %[[LHS_REAL_IS_INF_WITH_SIGN:.*]] = math.copysign %[[LHS_REAL_IS_INF]], %[[LHS_REAL]] : f32
// DIV-SMITH: %[[LHS_IMAG_IS_INF:.*]] = arith.select %[[LHS_IMAG_INFINITE]], %[[ONE]], %[[ZERO]] : f32
// DIV-SMITH: %[[LHS_IMAG_IS_INF_WITH_SIGN:.*]] = math.copysign %[[LHS_IMAG_IS_INF]], %[[LHS_IMAG]] : f32
// DIV-SMITH: %[[LHS_REAL_IS_INF_WITH_SIGN_TIMES_RHS_REAL:.*]] = arith.mulf %[[LHS_REAL_IS_INF_WITH_SIGN]], %[[RHS_REAL]] : f32
// DIV-SMITH: %[[LHS_IMAG_IS_INF_WITH_SIGN_TIMES_RHS_IMAG:.*]] = arith.mulf %[[LHS_IMAG_IS_INF_WITH_SIGN]], %[[RHS_IMAG]] : f32
// DIV-SMITH: %[[INF_MULTIPLICATOR_1:.*]] = arith.addf %[[LHS_REAL_IS_INF_WITH_SIGN_TIMES_RHS_REAL]], %[[LHS_IMAG_IS_INF_WITH_SIGN_TIMES_RHS_IMAG]] : f32
// DIV-SMITH: %[[RESULT_REAL_3:.*]] = arith.mulf %[[INF]], %[[INF_MULTIPLICATOR_1]] : f32
// DIV-SMITH: %[[LHS_REAL_IS_INF_WITH_SIGN_TIMES_RHS_IMAG:.*]] = arith.mulf %[[LHS_REAL_IS_INF_WITH_SIGN]], %[[RHS_IMAG]] : f32
// DIV-SMITH: %[[LHS_IMAG_IS_INF_WITH_SIGN_TIMES_RHS_REAL:.*]] = arith.mulf %[[LHS_IMAG_IS_INF_WITH_SIGN]], %[[RHS_REAL]] : f32
// DIV-SMITH: %[[INF_MULTIPLICATOR_2:.*]] = arith.subf %[[LHS_IMAG_IS_INF_WITH_SIGN_TIMES_RHS_REAL]], %[[LHS_REAL_IS_INF_WITH_SIGN_TIMES_RHS_IMAG]] : f32
// DIV-SMITH: %[[RESULT_IMAG_3:.*]] = arith.mulf %[[INF]], %[[INF_MULTIPLICATOR_2]] : f32

// Case 3. Finite numerator, infinite denominator.
// DIV-SMITH: %[[LHS_REAL_FINITE:.*]] = arith.cmpf one, %[[LHS_REAL_ABS]], %[[INF]] : f32
// DIV-SMITH: %[[LHS_IMAG_FINITE:.*]] = arith.cmpf one, %[[LHS_IMAG_ABS]], %[[INF]] : f32
// DIV-SMITH: %[[LHS_IS_FINITE:.*]] = arith.andi %[[LHS_REAL_FINITE]], %[[LHS_IMAG_FINITE]] : i1
// DIV-SMITH: %[[RHS_REAL_INFINITE:.*]] = arith.cmpf oeq, %[[RHS_REAL_ABS]], %[[INF]] : f32
// DIV-SMITH: %[[RHS_IMAG_INFINITE:.*]] = arith.cmpf oeq, %[[RHS_IMAG_ABS]], %[[INF]] : f32
// DIV-SMITH: %[[RHS_IS_INFINITE:.*]] = arith.ori %[[RHS_REAL_INFINITE]], %[[RHS_IMAG_INFINITE]] : i1
// DIV-SMITH: %[[FINITE_NUM_INFINITE_DENOM:.*]] = arith.andi %[[LHS_IS_FINITE]], %[[RHS_IS_INFINITE]] : i1
// DIV-SMITH: %[[RHS_REAL_IS_INF:.*]] = arith.select %[[RHS_REAL_INFINITE]], %[[ONE]], %[[ZERO]] : f32
// DIV-SMITH: %[[RHS_REAL_IS_INF_WITH_SIGN:.*]] = math.copysign %[[RHS_REAL_IS_INF]], %[[RHS_REAL]] : f32
// DIV-SMITH: %[[RHS_IMAG_IS_INF:.*]] = arith.select %[[RHS_IMAG_INFINITE]], %[[ONE]], %[[ZERO]] : f32
// DIV-SMITH: %[[RHS_IMAG_IS_INF_WITH_SIGN:.*]] = math.copysign %[[RHS_IMAG_IS_INF]], %[[RHS_IMAG]] : f32
// DIV-SMITH: %[[RHS_REAL_IS_INF_WITH_SIGN_TIMES_LHS_REAL:.*]] = arith.mulf %[[LHS_REAL]], %[[RHS_REAL_IS_INF_WITH_SIGN]] : f32
// DIV-SMITH: %[[RHS_IMAG_IS_INF_WITH_SIGN_TIMES_LHS_IMAG:.*]] = arith.mulf %[[LHS_IMAG]], %[[RHS_IMAG_IS_INF_WITH_SIGN]] : f32
// DIV-SMITH: %[[ZERO_MULTIPLICATOR_1:.*]] = arith.addf %[[RHS_REAL_IS_INF_WITH_SIGN_TIMES_LHS_REAL]], %[[RHS_IMAG_IS_INF_WITH_SIGN_TIMES_LHS_IMAG]] : f32
// DIV-SMITH: %[[RESULT_REAL_4:.*]] = arith.mulf %[[ZERO]], %[[ZERO_MULTIPLICATOR_1]] : f32
// DIV-SMITH: %[[RHS_REAL_IS_INF_WITH_SIGN_TIMES_LHS_IMAG:.*]] = arith.mulf %[[LHS_IMAG]], %[[RHS_REAL_IS_INF_WITH_SIGN]] : f32
// DIV-SMITH: %[[RHS_IMAG_IS_INF_WITH_SIGN_TIMES_LHS_REAL:.*]] = arith.mulf %[[LHS_REAL]], %[[RHS_IMAG_IS_INF_WITH_SIGN]] : f32
// DIV-SMITH: %[[ZERO_MULTIPLICATOR_2:.*]] = arith.subf %[[RHS_REAL_IS_INF_WITH_SIGN_TIMES_LHS_IMAG]], %[[RHS_IMAG_IS_INF_WITH_SIGN_TIMES_LHS_REAL]] : f32
// DIV-SMITH: %[[RESULT_IMAG_4:.*]] = arith.mulf %[[ZERO]], %[[ZERO_MULTIPLICATOR_2]] : f32

// DIV-SMITH: %[[REAL_ABS_SMALLER_THAN_IMAG_ABS:.*]] = arith.cmpf olt, %[[RHS_REAL_ABS]], %[[RHS_IMAG_ABS]] : f32
// DIV-SMITH: %[[RESULT_REAL:.*]] = arith.select %[[REAL_ABS_SMALLER_THAN_IMAG_ABS]], %[[RESULT_REAL_1]], %[[RESULT_REAL_2]] : f32
// DIV-SMITH: %[[RESULT_IMAG:.*]] = arith.select %[[REAL_ABS_SMALLER_THAN_IMAG_ABS]], %[[RESULT_IMAG_1]], %[[RESULT_IMAG_2]] : f32
// DIV-SMITH: %[[RESULT_REAL_SPECIAL_CASE_3:.*]] = arith.select %[[FINITE_NUM_INFINITE_DENOM]], %[[RESULT_REAL_4]], %[[RESULT_REAL]] : f32
// DIV-SMITH: %[[RESULT_IMAG_SPECIAL_CASE_3:.*]] = arith.select %[[FINITE_NUM_INFINITE_DENOM]], %[[RESULT_IMAG_4]], %[[RESULT_IMAG]] : f32
// DIV-SMITH: %[[RESULT_REAL_SPECIAL_CASE_2:.*]] = arith.select %[[INF_NUM_FINITE_DENOM]], %[[RESULT_REAL_3]], %[[RESULT_REAL_SPECIAL_CASE_3]] : f32
// DIV-SMITH: %[[RESULT_IMAG_SPECIAL_CASE_2:.*]] = arith.select %[[INF_NUM_FINITE_DENOM]], %[[RESULT_IMAG_3]], %[[RESULT_IMAG_SPECIAL_CASE_3]] : f32
// DIV-SMITH: %[[RESULT_REAL_SPECIAL_CASE_1:.*]] = arith.select %[[RESULT_IS_INFINITY]], %[[INFINITY_RESULT_REAL]], %[[RESULT_REAL_SPECIAL_CASE_2]] : f32
// DIV-SMITH: %[[RESULT_IMAG_SPECIAL_CASE_1:.*]] = arith.select %[[RESULT_IS_INFINITY]], %[[INFINITY_RESULT_IMAG]], %[[RESULT_IMAG_SPECIAL_CASE_2]] : f32
// DIV-SMITH: %[[RESULT_REAL_IS_NAN:.*]] = arith.cmpf uno, %[[RESULT_REAL]], %[[ZERO]] : f32
// DIV-SMITH: %[[RESULT_IMAG_IS_NAN:.*]] = arith.cmpf uno, %[[RESULT_IMAG]], %[[ZERO]] : f32
// DIV-SMITH: %[[RESULT_IS_NAN:.*]] = arith.andi %[[RESULT_REAL_IS_NAN]], %[[RESULT_IMAG_IS_NAN]] : i1
// DIV-SMITH: %[[RESULT_REAL_WITH_SPECIAL_CASES:.*]] = arith.select %[[RESULT_IS_NAN]], %[[RESULT_REAL_SPECIAL_CASE_1]], %[[RESULT_REAL]] : f32
// DIV-SMITH: %[[RESULT_IMAG_WITH_SPECIAL_CASES:.*]] = arith.select %[[RESULT_IS_NAN]], %[[RESULT_IMAG_SPECIAL_CASE_1]], %[[RESULT_IMAG]] : f32
// DIV-SMITH: %[[RESULT:.*]] = complex.create %[[RESULT_REAL_WITH_SPECIAL_CASES]], %[[RESULT_IMAG_WITH_SPECIAL_CASES]] : complex<f32>
// DIV-SMITH: return %[[RESULT]] : complex<f32>


// DIV-ALGEBRAIC-LABEL: func @complex_div
// DIV-ALGEBRAIC-SAME:    %[[LHS:.*]]: complex<f32>, %[[RHS:.*]]: complex<f32>

// DIV-ALGEBRAIC: %[[LHS_RE:.*]] = complex.re %[[LHS]] : complex<f32>
// DIV-ALGEBRAIC: %[[LHS_IM:.*]] = complex.im %[[LHS]] : complex<f32>
// DIV-ALGEBRAIC: %[[RHS_RE:.*]] = complex.re %[[RHS]] : complex<f32>
// DIV-ALGEBRAIC: %[[RHS_IM:.*]] = complex.im %[[RHS]] : complex<f32>

// DIV-ALGEBRAIC-DAG: %[[RHS_RE_SQ:.*]] = arith.mulf %[[RHS_RE]], %[[RHS_RE]]  : f32
// DIV-ALGEBRAIC-DAG: %[[RHS_IM_SQ:.*]] = arith.mulf %[[RHS_IM]], %[[RHS_IM]]  : f32
// DIV-ALGEBRAIC: %[[SQ_NORM:.*]] = arith.addf %[[RHS_RE_SQ]], %[[RHS_IM_SQ]]  : f32

// DIV-ALGEBRAIC-DAG: %[[REAL_TMP_0:.*]] = arith.mulf %[[LHS_RE]], %[[RHS_RE]]  : f32
// DIV-ALGEBRAIC-DAG: %[[REAL_TMP_1:.*]] = arith.mulf %[[LHS_IM]], %[[RHS_IM]]  : f32
// DIV-ALGEBRAIC: %[[REAL_TMP_2:.*]] = arith.addf %[[REAL_TMP_0]], %[[REAL_TMP_1]]  : f32

// DIV-ALGEBRAIC-DAG: %[[IMAG_TMP_0:.*]] = arith.mulf %[[LHS_IM]], %[[RHS_RE]]  : f32
// DIV-ALGEBRAIC-DAG: %[[IMAG_TMP_1:.*]] = arith.mulf %[[LHS_RE]], %[[RHS_IM]]  : f32
// DIV-ALGEBRAIC: %[[IMAG_TMP_2:.*]] = arith.subf %[[IMAG_TMP_0]], %[[IMAG_TMP_1]]  : f32

// DIV-ALGEBRAIC: %[[REAL:.*]] = arith.divf %[[REAL_TMP_2]], %[[SQ_NORM]]  : f32
// DIV-ALGEBRAIC: %[[IMAG:.*]] = arith.divf %[[IMAG_TMP_2]], %[[SQ_NORM]]  : f32
// DIV-ALGEBRAIC: %[[RESULT:.*]] = complex.create %[[REAL]], %[[IMAG]] : complex<f32>
// DIV-ALGEBRAIC: return %[[RESULT]] : complex<f32>


func.func @complex_div_with_fmf(%lhs: complex<f32>, %rhs: complex<f32>) -> complex<f32> {
  %div = complex.div %lhs, %rhs fastmath<nsz,arcp> : complex<f32>
  return %div : complex<f32>
}
// DIV-SMITH-LABEL: func @complex_div_with_fmf
// DIV-SMITH-SAME:    %[[LHS:.*]]: complex<f32>, %[[RHS:.*]]: complex<f32>

// DIV-SMITH: %[[LHS_REAL:.*]] = complex.re %[[LHS]] : complex<f32>
// DIV-SMITH: %[[LHS_IMAG:.*]] = complex.im %[[LHS]] : complex<f32>
// DIV-SMITH: %[[RHS_REAL:.*]] = complex.re %[[RHS]] : complex<f32>
// DIV-SMITH: %[[RHS_IMAG:.*]] = complex.im %[[RHS]] : complex<f32>

// DIV-SMITH: %[[RHS_REAL_IMAG_RATIO:.*]] = arith.divf %[[RHS_REAL]], %[[RHS_IMAG]] fastmath<nsz,arcp> : f32
// DIV-SMITH: %[[RHS_REAL_TIMES_RHS_REAL_IMAG_RATIO:.*]] = arith.mulf %[[RHS_REAL_IMAG_RATIO]], %[[RHS_REAL]] fastmath<nsz,arcp> : f32
// DIV-SMITH: %[[RHS_REAL_IMAG_DENOM:.*]] = arith.addf %[[RHS_IMAG]], %[[RHS_REAL_TIMES_RHS_REAL_IMAG_RATIO]] fastmath<nsz,arcp> : f32
// DIV-SMITH: %[[LHS_REAL_TIMES_RHS_REAL_IMAG_RATIO:.*]] = arith.mulf %[[LHS_REAL]], %[[RHS_REAL_IMAG_RATIO]] fastmath<nsz,arcp> : f32
// DIV-SMITH: %[[REAL_NUMERATOR_1:.*]] = arith.addf %[[LHS_REAL_TIMES_RHS_REAL_IMAG_RATIO]], %[[LHS_IMAG]] fastmath<nsz,arcp> : f32
// DIV-SMITH: %[[RESULT_REAL_1:.*]] = arith.divf %[[REAL_NUMERATOR_1]], %[[RHS_REAL_IMAG_DENOM]] fastmath<nsz,arcp> : f32
// DIV-SMITH: %[[LHS_IMAG_TIMES_RHS_REAL_IMAG_RATIO:.*]] = arith.mulf %[[LHS_IMAG]], %[[RHS_REAL_IMAG_RATIO]] fastmath<nsz,arcp> : f32
// DIV-SMITH: %[[IMAG_NUMERATOR_1:.*]] = arith.subf %[[LHS_IMAG_TIMES_RHS_REAL_IMAG_RATIO]], %[[LHS_REAL]] fastmath<nsz,arcp> : f32
// DIV-SMITH: %[[RESULT_IMAG_1:.*]] = arith.divf %[[IMAG_NUMERATOR_1]], %[[RHS_REAL_IMAG_DENOM]] fastmath<nsz,arcp> : f32

// DIV-SMITH: %[[RHS_IMAG_REAL_RATIO:.*]] = arith.divf %[[RHS_IMAG]], %[[RHS_REAL]] fastmath<nsz,arcp> : f32
// DIV-SMITH: %[[RHS_IMAG_TIMES_RHS_IMAG_REAL_RATIO:.*]] = arith.mulf %[[RHS_IMAG_REAL_RATIO]], %[[RHS_IMAG]] fastmath<nsz,arcp> : f32
// DIV-SMITH: %[[RHS_IMAG_REAL_DENOM:.*]] = arith.addf %[[RHS_REAL]], %[[RHS_IMAG_TIMES_RHS_IMAG_REAL_RATIO]] fastmath<nsz,arcp> : f32
// DIV-SMITH: %[[LHS_IMAG_TIMES_RHS_IMAG_REAL_RATIO:.*]] = arith.mulf %[[LHS_IMAG]], %[[RHS_IMAG_REAL_RATIO]] fastmath<nsz,arcp> : f32
// DIV-SMITH: %[[REAL_NUMERATOR_2:.*]] = arith.addf %[[LHS_REAL]], %[[LHS_IMAG_TIMES_RHS_IMAG_REAL_RATIO]] fastmath<nsz,arcp> : f32
// DIV-SMITH: %[[RESULT_REAL_2:.*]] = arith.divf %[[REAL_NUMERATOR_2]], %[[RHS_IMAG_REAL_DENOM]] fastmath<nsz,arcp> : f32
// DIV-SMITH: %[[LHS_REAL_TIMES_RHS_IMAG_REAL_RATIO:.*]] = arith.mulf %[[LHS_REAL]], %[[RHS_IMAG_REAL_RATIO]] fastmath<nsz,arcp> : f32
// DIV-SMITH: %[[IMAG_NUMERATOR_2:.*]] = arith.subf %[[LHS_IMAG]], %[[LHS_REAL_TIMES_RHS_IMAG_REAL_RATIO]] fastmath<nsz,arcp> : f32
// DIV-SMITH: %[[RESULT_IMAG_2:.*]] = arith.divf %[[IMAG_NUMERATOR_2]], %[[RHS_IMAG_REAL_DENOM]] fastmath<nsz,arcp> : f32

// Case 1. Zero denominator, numerator contains at most one NaN value.
// DIV-SMITH: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// DIV-SMITH: %[[RHS_REAL_ABS:.*]] = math.absf %[[RHS_REAL]] fastmath<nsz,arcp> : f32
// DIV-SMITH: %[[RHS_REAL_ABS_IS_ZERO:.*]] = arith.cmpf oeq, %[[RHS_REAL_ABS]], %[[ZERO]] : f32
// DIV-SMITH: %[[RHS_IMAG_ABS:.*]] = math.absf %[[RHS_IMAG]] fastmath<nsz,arcp> : f32
// DIV-SMITH: %[[RHS_IMAG_ABS_IS_ZERO:.*]] = arith.cmpf oeq, %[[RHS_IMAG_ABS]], %[[ZERO]] : f32
// DIV-SMITH: %[[LHS_REAL_IS_NOT_NAN:.*]] = arith.cmpf ord, %[[LHS_REAL]], %[[ZERO]] : f32
// DIV-SMITH: %[[LHS_IMAG_IS_NOT_NAN:.*]] = arith.cmpf ord, %[[LHS_IMAG]], %[[ZERO]] : f32
// DIV-SMITH: %[[LHS_CONTAINS_NOT_NAN_VALUE:.*]] = arith.ori %[[LHS_REAL_IS_NOT_NAN]], %[[LHS_IMAG_IS_NOT_NAN]] : i1
// DIV-SMITH: %[[RHS_IS_ZERO:.*]] = arith.andi %[[RHS_REAL_ABS_IS_ZERO]], %[[RHS_IMAG_ABS_IS_ZERO]] : i1
// DIV-SMITH: %[[RESULT_IS_INFINITY:.*]] = arith.andi %[[LHS_CONTAINS_NOT_NAN_VALUE]], %[[RHS_IS_ZERO]] : i1
// DIV-SMITH: %[[INF:.*]] = arith.constant 0x7F800000 : f32
// DIV-SMITH: %[[INF_WITH_SIGN_OF_RHS_REAL:.*]] = math.copysign %[[INF]], %[[RHS_REAL]] : f32
// DIV-SMITH: %[[INFINITY_RESULT_REAL:.*]] = arith.mulf %[[INF_WITH_SIGN_OF_RHS_REAL]], %[[LHS_REAL]] fastmath<nsz,arcp> : f32
// DIV-SMITH: %[[INFINITY_RESULT_IMAG:.*]] = arith.mulf %[[INF_WITH_SIGN_OF_RHS_REAL]], %[[LHS_IMAG]] fastmath<nsz,arcp> : f32

// Case 2. Infinite numerator, finite denominator.
// DIV-SMITH: %[[RHS_REAL_FINITE:.*]] = arith.cmpf one, %[[RHS_REAL_ABS]], %[[INF]] : f32
// DIV-SMITH: %[[RHS_IMAG_FINITE:.*]] = arith.cmpf one, %[[RHS_IMAG_ABS]], %[[INF]] : f32
// DIV-SMITH: %[[RHS_IS_FINITE:.*]] = arith.andi %[[RHS_REAL_FINITE]], %[[RHS_IMAG_FINITE]] : i1
// DIV-SMITH: %[[LHS_REAL_ABS:.*]] = math.absf %[[LHS_REAL]] fastmath<nsz,arcp> : f32
// DIV-SMITH: %[[LHS_REAL_INFINITE:.*]] = arith.cmpf oeq, %[[LHS_REAL_ABS]], %[[INF]] : f32
// DIV-SMITH: %[[LHS_IMAG_ABS:.*]] = math.absf %[[LHS_IMAG]] fastmath<nsz,arcp> : f32
// DIV-SMITH: %[[LHS_IMAG_INFINITE:.*]] = arith.cmpf oeq, %[[LHS_IMAG_ABS]], %[[INF]] : f32
// DIV-SMITH: %[[LHS_IS_INFINITE:.*]] = arith.ori %[[LHS_REAL_INFINITE]], %[[LHS_IMAG_INFINITE]] : i1
// DIV-SMITH: %[[INF_NUM_FINITE_DENOM:.*]] = arith.andi %[[LHS_IS_INFINITE]], %[[RHS_IS_FINITE]] : i1
// DIV-SMITH: %[[ONE:.*]] = arith.constant 1.000000e+00 : f32
// DIV-SMITH: %[[LHS_REAL_IS_INF:.*]] = arith.select %[[LHS_REAL_INFINITE]], %[[ONE]], %[[ZERO]] : f32
// DIV-SMITH: %[[LHS_REAL_IS_INF_WITH_SIGN:.*]] = math.copysign %[[LHS_REAL_IS_INF]], %[[LHS_REAL]] : f32
// DIV-SMITH: %[[LHS_IMAG_IS_INF:.*]] = arith.select %[[LHS_IMAG_INFINITE]], %[[ONE]], %[[ZERO]] : f32
// DIV-SMITH: %[[LHS_IMAG_IS_INF_WITH_SIGN:.*]] = math.copysign %[[LHS_IMAG_IS_INF]], %[[LHS_IMAG]] : f32
// DIV-SMITH: %[[LHS_REAL_IS_INF_WITH_SIGN_TIMES_RHS_REAL:.*]] = arith.mulf %[[LHS_REAL_IS_INF_WITH_SIGN]], %[[RHS_REAL]] fastmath<nsz,arcp> : f32
// DIV-SMITH: %[[LHS_IMAG_IS_INF_WITH_SIGN_TIMES_RHS_IMAG:.*]] = arith.mulf %[[LHS_IMAG_IS_INF_WITH_SIGN]], %[[RHS_IMAG]] fastmath<nsz,arcp> : f32
// DIV-SMITH: %[[INF_MULTIPLICATOR_1:.*]] = arith.addf %[[LHS_REAL_IS_INF_WITH_SIGN_TIMES_RHS_REAL]], %[[LHS_IMAG_IS_INF_WITH_SIGN_TIMES_RHS_IMAG]] fastmath<nsz,arcp> : f32
// DIV-SMITH: %[[RESULT_REAL_3:.*]] = arith.mulf %[[INF]], %[[INF_MULTIPLICATOR_1]] fastmath<nsz,arcp> : f32
// DIV-SMITH: %[[LHS_REAL_IS_INF_WITH_SIGN_TIMES_RHS_IMAG:.*]] = arith.mulf %[[LHS_REAL_IS_INF_WITH_SIGN]], %[[RHS_IMAG]] fastmath<nsz,arcp> : f32
// DIV-SMITH: %[[LHS_IMAG_IS_INF_WITH_SIGN_TIMES_RHS_REAL:.*]] = arith.mulf %[[LHS_IMAG_IS_INF_WITH_SIGN]], %[[RHS_REAL]] fastmath<nsz,arcp> : f32
// DIV-SMITH: %[[INF_MULTIPLICATOR_2:.*]] = arith.subf %[[LHS_IMAG_IS_INF_WITH_SIGN_TIMES_RHS_REAL]], %[[LHS_REAL_IS_INF_WITH_SIGN_TIMES_RHS_IMAG]] fastmath<nsz,arcp> : f32
// DIV-SMITH: %[[RESULT_IMAG_3:.*]] = arith.mulf %[[INF]], %[[INF_MULTIPLICATOR_2]] fastmath<nsz,arcp> : f32

// Case 3. Finite numerator, infinite denominator.
// DIV-SMITH: %[[LHS_REAL_FINITE:.*]] = arith.cmpf one, %[[LHS_REAL_ABS]], %[[INF]] : f32
// DIV-SMITH: %[[LHS_IMAG_FINITE:.*]] = arith.cmpf one, %[[LHS_IMAG_ABS]], %[[INF]] : f32
// DIV-SMITH: %[[LHS_IS_FINITE:.*]] = arith.andi %[[LHS_REAL_FINITE]], %[[LHS_IMAG_FINITE]] : i1
// DIV-SMITH: %[[RHS_REAL_INFINITE:.*]] = arith.cmpf oeq, %[[RHS_REAL_ABS]], %[[INF]] : f32
// DIV-SMITH: %[[RHS_IMAG_INFINITE:.*]] = arith.cmpf oeq, %[[RHS_IMAG_ABS]], %[[INF]] : f32
// DIV-SMITH: %[[RHS_IS_INFINITE:.*]] = arith.ori %[[RHS_REAL_INFINITE]], %[[RHS_IMAG_INFINITE]] : i1
// DIV-SMITH: %[[FINITE_NUM_INFINITE_DENOM:.*]] = arith.andi %[[LHS_IS_FINITE]], %[[RHS_IS_INFINITE]] : i1
// DIV-SMITH: %[[RHS_REAL_IS_INF:.*]] = arith.select %[[RHS_REAL_INFINITE]], %[[ONE]], %[[ZERO]] : f32
// DIV-SMITH: %[[RHS_REAL_IS_INF_WITH_SIGN:.*]] = math.copysign %[[RHS_REAL_IS_INF]], %[[RHS_REAL]] : f32
// DIV-SMITH: %[[RHS_IMAG_IS_INF:.*]] = arith.select %[[RHS_IMAG_INFINITE]], %[[ONE]], %[[ZERO]] : f32
// DIV-SMITH: %[[RHS_IMAG_IS_INF_WITH_SIGN:.*]] = math.copysign %[[RHS_IMAG_IS_INF]], %[[RHS_IMAG]] : f32
// DIV-SMITH: %[[RHS_REAL_IS_INF_WITH_SIGN_TIMES_LHS_REAL:.*]] = arith.mulf %[[LHS_REAL]], %[[RHS_REAL_IS_INF_WITH_SIGN]] fastmath<nsz,arcp> : f32
// DIV-SMITH: %[[RHS_IMAG_IS_INF_WITH_SIGN_TIMES_LHS_IMAG:.*]] = arith.mulf %[[LHS_IMAG]], %[[RHS_IMAG_IS_INF_WITH_SIGN]] fastmath<nsz,arcp> : f32
// DIV-SMITH: %[[ZERO_MULTIPLICATOR_1:.*]] = arith.addf %[[RHS_REAL_IS_INF_WITH_SIGN_TIMES_LHS_REAL]], %[[RHS_IMAG_IS_INF_WITH_SIGN_TIMES_LHS_IMAG]] fastmath<nsz,arcp> : f32
// DIV-SMITH: %[[RESULT_REAL_4:.*]] = arith.mulf %[[ZERO]], %[[ZERO_MULTIPLICATOR_1]] fastmath<nsz,arcp> : f32
// DIV-SMITH: %[[RHS_REAL_IS_INF_WITH_SIGN_TIMES_LHS_IMAG:.*]] = arith.mulf %[[LHS_IMAG]], %[[RHS_REAL_IS_INF_WITH_SIGN]] fastmath<nsz,arcp> : f32
// DIV-SMITH: %[[RHS_IMAG_IS_INF_WITH_SIGN_TIMES_LHS_REAL:.*]] = arith.mulf %[[LHS_REAL]], %[[RHS_IMAG_IS_INF_WITH_SIGN]] fastmath<nsz,arcp> : f32
// DIV-SMITH: %[[ZERO_MULTIPLICATOR_2:.*]] = arith.subf %[[RHS_REAL_IS_INF_WITH_SIGN_TIMES_LHS_IMAG]], %[[RHS_IMAG_IS_INF_WITH_SIGN_TIMES_LHS_REAL]] fastmath<nsz,arcp> : f32
// DIV-SMITH: %[[RESULT_IMAG_4:.*]] = arith.mulf %[[ZERO]], %[[ZERO_MULTIPLICATOR_2]] fastmath<nsz,arcp> : f32

// DIV-SMITH: %[[REAL_ABS_SMALLER_THAN_IMAG_ABS:.*]] = arith.cmpf olt, %[[RHS_REAL_ABS]], %[[RHS_IMAG_ABS]] : f32
// DIV-SMITH: %[[RESULT_REAL:.*]] = arith.select %[[REAL_ABS_SMALLER_THAN_IMAG_ABS]], %[[RESULT_REAL_1]], %[[RESULT_REAL_2]] : f32
// DIV-SMITH: %[[RESULT_IMAG:.*]] = arith.select %[[REAL_ABS_SMALLER_THAN_IMAG_ABS]], %[[RESULT_IMAG_1]], %[[RESULT_IMAG_2]] : f32
// DIV-SMITH: %[[RESULT_REAL_SPECIAL_CASE_3:.*]] = arith.select %[[FINITE_NUM_INFINITE_DENOM]], %[[RESULT_REAL_4]], %[[RESULT_REAL]] : f32
// DIV-SMITH: %[[RESULT_IMAG_SPECIAL_CASE_3:.*]] = arith.select %[[FINITE_NUM_INFINITE_DENOM]], %[[RESULT_IMAG_4]], %[[RESULT_IMAG]] : f32
// DIV-SMITH: %[[RESULT_REAL_SPECIAL_CASE_2:.*]] = arith.select %[[INF_NUM_FINITE_DENOM]], %[[RESULT_REAL_3]], %[[RESULT_REAL_SPECIAL_CASE_3]] : f32
// DIV-SMITH: %[[RESULT_IMAG_SPECIAL_CASE_2:.*]] = arith.select %[[INF_NUM_FINITE_DENOM]], %[[RESULT_IMAG_3]], %[[RESULT_IMAG_SPECIAL_CASE_3]] : f32
// DIV-SMITH: %[[RESULT_REAL_SPECIAL_CASE_1:.*]] = arith.select %[[RESULT_IS_INFINITY]], %[[INFINITY_RESULT_REAL]], %[[RESULT_REAL_SPECIAL_CASE_2]] : f32
// DIV-SMITH: %[[RESULT_IMAG_SPECIAL_CASE_1:.*]] = arith.select %[[RESULT_IS_INFINITY]], %[[INFINITY_RESULT_IMAG]], %[[RESULT_IMAG_SPECIAL_CASE_2]] : f32
// DIV-SMITH: %[[RESULT_REAL_IS_NAN:.*]] = arith.cmpf uno, %[[RESULT_REAL]], %[[ZERO]] : f32
// DIV-SMITH: %[[RESULT_IMAG_IS_NAN:.*]] = arith.cmpf uno, %[[RESULT_IMAG]], %[[ZERO]] : f32
// DIV-SMITH: %[[RESULT_IS_NAN:.*]] = arith.andi %[[RESULT_REAL_IS_NAN]], %[[RESULT_IMAG_IS_NAN]] : i1
// DIV-SMITH: %[[RESULT_REAL_WITH_SPECIAL_CASES:.*]] = arith.select %[[RESULT_IS_NAN]], %[[RESULT_REAL_SPECIAL_CASE_1]], %[[RESULT_REAL]] : f32
// DIV-SMITH: %[[RESULT_IMAG_WITH_SPECIAL_CASES:.*]] = arith.select %[[RESULT_IS_NAN]], %[[RESULT_IMAG_SPECIAL_CASE_1]], %[[RESULT_IMAG]] : f32
// DIV-SMITH: %[[RESULT:.*]] = complex.create %[[RESULT_REAL_WITH_SPECIAL_CASES]], %[[RESULT_IMAG_WITH_SPECIAL_CASES]] : complex<f32>
// DIV-SMITH: return %[[RESULT]] : complex<f32>


// DIV-ALGEBRAIC-LABEL: func @complex_div_with_fmf
// DIV-ALGEBRAIC-SAME:    %[[LHS:.*]]: complex<f32>, %[[RHS:.*]]: complex<f32>

// DIV-ALGEBRAIC: %[[LHS_RE:.*]] = complex.re %[[LHS]] : complex<f32>
// DIV-ALGEBRAIC: %[[LHS_IM:.*]] = complex.im %[[LHS]] : complex<f32>
// DIV-ALGEBRAIC: %[[RHS_RE:.*]] = complex.re %[[RHS]] : complex<f32>
// DIV-ALGEBRAIC: %[[RHS_IM:.*]] = complex.im %[[RHS]] : complex<f32>

// DIV-ALGEBRAIC-DAG: %[[RHS_RE_SQ:.*]] = arith.mulf %[[RHS_RE]], %[[RHS_RE]] fastmath<nsz,arcp> : f32
// DIV-ALGEBRAIC-DAG: %[[RHS_IM_SQ:.*]] = arith.mulf %[[RHS_IM]], %[[RHS_IM]] fastmath<nsz,arcp> : f32
// DIV-ALGEBRAIC: %[[SQ_NORM:.*]] = arith.addf %[[RHS_RE_SQ]], %[[RHS_IM_SQ]] fastmath<nsz,arcp> : f32

// DIV-ALGEBRAIC-DAG: %[[REAL_TMP_0:.*]] = arith.mulf %[[LHS_RE]], %[[RHS_RE]] fastmath<nsz,arcp> : f32
// DIV-ALGEBRAIC-DAG: %[[REAL_TMP_1:.*]] = arith.mulf %[[LHS_IM]], %[[RHS_IM]] fastmath<nsz,arcp> : f32
// DIV-ALGEBRAIC: %[[REAL_TMP_2:.*]] = arith.addf %[[REAL_TMP_0]], %[[REAL_TMP_1]] fastmath<nsz,arcp> : f32

// DIV-ALGEBRAIC-DAG: %[[IMAG_TMP_0:.*]] = arith.mulf %[[LHS_IM]], %[[RHS_RE]] fastmath<nsz,arcp> : f32
// DIV-ALGEBRAIC-DAG: %[[IMAG_TMP_1:.*]] = arith.mulf %[[LHS_RE]], %[[RHS_IM]] fastmath<nsz,arcp> : f32
// DIV-ALGEBRAIC: %[[IMAG_TMP_2:.*]] = arith.subf %[[IMAG_TMP_0]], %[[IMAG_TMP_1]] fastmath<nsz,arcp> : f32

// DIV-ALGEBRAIC: %[[REAL:.*]] = arith.divf %[[REAL_TMP_2]], %[[SQ_NORM]] fastmath<nsz,arcp> : f32
// DIV-ALGEBRAIC: %[[IMAG:.*]] = arith.divf %[[IMAG_TMP_2]], %[[SQ_NORM]] fastmath<nsz,arcp> : f32
// DIV-ALGEBRAIC: %[[RESULT:.*]] = complex.create %[[REAL]], %[[IMAG]] : complex<f32>
// DIV-ALGEBRAIC: return %[[RESULT]] : complex<f32>
