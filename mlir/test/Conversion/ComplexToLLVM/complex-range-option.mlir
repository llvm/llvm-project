// RUN: mlir-opt %s -convert-complex-to-llvm=complex-range=improved | FileCheck %s --check-prefix=DIV-SMITH
// RUN: mlir-opt %s -convert-complex-to-llvm=complex-range=basic | FileCheck %s --check-prefix=DIV-ALGEBRAIC
// RUN: mlir-opt %s -convert-complex-to-llvm=complex-range=none | FileCheck %s --check-prefix=DIV-ALGEBRAIC


func.func @complex_div(%lhs: complex<f32>, %rhs: complex<f32>) -> complex<f32> {
  %div = complex.div %lhs, %rhs : complex<f32>
  return %div : complex<f32>
}
// DIV-SMITH-LABEL: func @complex_div
// DIV-SMITH-SAME:    %[[LHS:.*]]: complex<f32>, %[[RHS:.*]]: complex<f32>
// DIV-SMITH-DAG: %[[CASTED_LHS:.*]] = builtin.unrealized_conversion_cast %[[LHS]] : complex<f32> to ![[C_TY:.*>]]
// DIV-SMITH-DAG: %[[CASTED_RHS:.*]] = builtin.unrealized_conversion_cast %[[RHS]] : complex<f32> to ![[C_TY]]

// DIV-SMITH: %[[LHS_REAL:.*]] = llvm.extractvalue %[[CASTED_LHS]][0] : ![[C_TY]]
// DIV-SMITH: %[[LHS_IMAG:.*]] = llvm.extractvalue %[[CASTED_LHS]][1] : ![[C_TY]]
// DIV-SMITH: %[[RHS_REAL:.*]] = llvm.extractvalue %[[CASTED_RHS]][0] : ![[C_TY]]
// DIV-SMITH: %[[RHS_IMAG:.*]] = llvm.extractvalue %[[CASTED_RHS]][1] : ![[C_TY]]

// DIV-SMITH: %[[RESULT_0:.*]] = llvm.mlir.poison : ![[C_TY]]

// DIV-SMITH: %[[RHS_REAL_IMAG_RATIO:.*]] = llvm.fdiv %[[RHS_REAL]], %[[RHS_IMAG]] : f32
// DIV-SMITH: %[[RHS_REAL_TIMES_RHS_REAL_IMAG_RATIO:.*]] = llvm.fmul %[[RHS_REAL_IMAG_RATIO]], %[[RHS_REAL]] : f32
// DIV-SMITH: %[[RHS_REAL_IMAG_DENOM:.*]] = llvm.fadd %[[RHS_IMAG]], %[[RHS_REAL_TIMES_RHS_REAL_IMAG_RATIO]] : f32
// DIV-SMITH: %[[LHS_REAL_TIMES_RHS_REAL_IMAG_RATIO:.*]] = llvm.fmul %[[LHS_REAL]], %[[RHS_REAL_IMAG_RATIO]] : f32
// DIV-SMITH: %[[REAL_NUMERATOR_1:.*]] = llvm.fadd %[[LHS_REAL_TIMES_RHS_REAL_IMAG_RATIO]], %[[LHS_IMAG]] : f32
// DIV-SMITH: %[[RESULT_REAL_1:.*]] = llvm.fdiv %[[REAL_NUMERATOR_1]], %[[RHS_REAL_IMAG_DENOM]] : f32
// DIV-SMITH: %[[LHS_IMAG_TIMES_RHS_REAL_IMAG_RATIO:.*]] = llvm.fmul %[[LHS_IMAG]], %[[RHS_REAL_IMAG_RATIO]] : f32
// DIV-SMITH: %[[IMAG_NUMERATOR_1:.*]] = llvm.fsub %[[LHS_IMAG_TIMES_RHS_REAL_IMAG_RATIO]], %[[LHS_REAL]] : f32
// DIV-SMITH: %[[RESULT_IMAG_1:.*]] = llvm.fdiv %[[IMAG_NUMERATOR_1]], %[[RHS_REAL_IMAG_DENOM]] : f32

// DIV-SMITH: %[[RHS_IMAG_REAL_RATIO:.*]] = llvm.fdiv %[[RHS_IMAG]], %[[RHS_REAL]] : f32
// DIV-SMITH: %[[RHS_IMAG_TIMES_RHS_IMAG_REAL_RATIO:.*]] = llvm.fmul %[[RHS_IMAG_REAL_RATIO]], %[[RHS_IMAG]] : f32
// DIV-SMITH: %[[RHS_IMAG_REAL_DENOM:.*]] = llvm.fadd %[[RHS_REAL]], %[[RHS_IMAG_TIMES_RHS_IMAG_REAL_RATIO]] : f32
// DIV-SMITH: %[[LHS_IMAG_TIMES_RHS_IMAG_REAL_RATIO:.*]] = llvm.fmul %[[LHS_IMAG]], %[[RHS_IMAG_REAL_RATIO]] : f32
// DIV-SMITH: %[[REAL_NUMERATOR_2:.*]] = llvm.fadd %[[LHS_REAL]], %[[LHS_IMAG_TIMES_RHS_IMAG_REAL_RATIO]] : f32
// DIV-SMITH: %[[RESULT_REAL_2:.*]] = llvm.fdiv %[[REAL_NUMERATOR_2]], %[[RHS_IMAG_REAL_DENOM]] : f32
// DIV-SMITH: %[[LHS_REAL_TIMES_RHS_IMAG_REAL_RATIO:.*]] = llvm.fmul %[[LHS_REAL]], %[[RHS_IMAG_REAL_RATIO]] : f32
// DIV-SMITH: %[[IMAG_NUMERATOR_2:.*]] = llvm.fsub %[[LHS_IMAG]], %[[LHS_REAL_TIMES_RHS_IMAG_REAL_RATIO]] : f32
// DIV-SMITH: %[[RESULT_IMAG_2:.*]] = llvm.fdiv %[[IMAG_NUMERATOR_2]], %[[RHS_IMAG_REAL_DENOM]] : f32

// Case 1. Zero denominator, numerator contains at most one NaN value.
// DIV-SMITH: %[[ZERO:.*]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// DIV-SMITH: %[[RHS_REAL_ABS:.*]] = llvm.intr.fabs(%[[RHS_REAL]]) : (f32) -> f32
// DIV-SMITH: %[[RHS_REAL_ABS_IS_ZERO:.*]] = llvm.fcmp "oeq" %[[RHS_REAL_ABS]], %[[ZERO]] : f32
// DIV-SMITH: %[[RHS_IMAG_ABS:.*]] = llvm.intr.fabs(%[[RHS_IMAG]]) : (f32) -> f32
// DIV-SMITH: %[[RHS_IMAG_ABS_IS_ZERO:.*]] = llvm.fcmp "oeq" %[[RHS_IMAG_ABS]], %[[ZERO]] : f32
// DIV-SMITH: %[[LHS_REAL_IS_NOT_NAN:.*]] = llvm.fcmp "ord" %[[LHS_REAL]], %[[ZERO]] : f32
// DIV-SMITH: %[[LHS_IMAG_IS_NOT_NAN:.*]] = llvm.fcmp "ord" %[[LHS_IMAG]], %[[ZERO]] : f32
// DIV-SMITH: %[[LHS_CONTAINS_NOT_NAN_VALUE:.*]] = llvm.or %[[LHS_REAL_IS_NOT_NAN]], %[[LHS_IMAG_IS_NOT_NAN]] : i1
// DIV-SMITH: %[[RHS_IS_ZERO:.*]] = llvm.and %[[RHS_REAL_ABS_IS_ZERO]], %[[RHS_IMAG_ABS_IS_ZERO]] : i1
// DIV-SMITH: %[[RESULT_IS_INFINITY:.*]] = llvm.and %[[LHS_CONTAINS_NOT_NAN_VALUE]], %[[RHS_IS_ZERO]] : i1
// DIV-SMITH: %[[INF:.*]] = llvm.mlir.constant(0x7F800000 : f32) : f32
// DIV-SMITH: %[[INF_WITH_SIGN_OF_RHS_REAL:.*]] = llvm.intr.copysign(%[[INF]], %[[RHS_REAL]]) : (f32, f32) -> f32
// DIV-SMITH: %[[INFINITY_RESULT_REAL:.*]] = llvm.fmul %[[INF_WITH_SIGN_OF_RHS_REAL]], %[[LHS_REAL]] : f32
// DIV-SMITH: %[[INFINITY_RESULT_IMAG:.*]] = llvm.fmul %[[INF_WITH_SIGN_OF_RHS_REAL]], %[[LHS_IMAG]] : f32

// Case 2. Infinite numerator, finite denominator.
// DIV-SMITH: %[[RHS_REAL_FINITE:.*]] = llvm.fcmp "one" %[[RHS_REAL_ABS]], %[[INF]] : f32
// DIV-SMITH: %[[RHS_IMAG_FINITE:.*]] = llvm.fcmp "one" %[[RHS_IMAG_ABS]], %[[INF]] : f32
// DIV-SMITH: %[[RHS_IS_FINITE:.*]] = llvm.and %[[RHS_REAL_FINITE]], %[[RHS_IMAG_FINITE]] : i1
// DIV-SMITH: %[[LHS_REAL_ABS:.*]] = llvm.intr.fabs(%[[LHS_REAL]]) : (f32) -> f32
// DIV-SMITH: %[[LHS_REAL_INFINITE:.*]] = llvm.fcmp "oeq" %[[LHS_REAL_ABS]], %[[INF]] : f32
// DIV-SMITH: %[[LHS_IMAG_ABS:.*]] = llvm.intr.fabs(%[[LHS_IMAG]]) : (f32) -> f32
// DIV-SMITH: %[[LHS_IMAG_INFINITE:.*]] = llvm.fcmp "oeq" %[[LHS_IMAG_ABS]], %[[INF]] : f32
// DIV-SMITH: %[[LHS_IS_INFINITE:.*]] = llvm.or %[[LHS_REAL_INFINITE]], %[[LHS_IMAG_INFINITE]] : i1
// DIV-SMITH: %[[INF_NUM_FINITE_DENOM:.*]] = llvm.and %[[LHS_IS_INFINITE]], %[[RHS_IS_FINITE]] : i1
// DIV-SMITH: %[[ONE:.*]] = llvm.mlir.constant(1.000000e+00 : f32) : f32
// DIV-SMITH: %[[LHS_REAL_IS_INF:.*]] = llvm.select %[[LHS_REAL_INFINITE]], %[[ONE]], %[[ZERO]] : i1, f32
// DIV-SMITH: %[[LHS_REAL_IS_INF_WITH_SIGN:.*]] = llvm.intr.copysign(%[[LHS_REAL_IS_INF]], %[[LHS_REAL]]) : (f32, f32) -> f32
// DIV-SMITH: %[[LHS_IMAG_IS_INF:.*]] = llvm.select %[[LHS_IMAG_INFINITE]], %[[ONE]], %[[ZERO]] : i1, f32
// DIV-SMITH: %[[LHS_IMAG_IS_INF_WITH_SIGN:.*]] = llvm.intr.copysign(%[[LHS_IMAG_IS_INF]], %[[LHS_IMAG]]) : (f32, f32) -> f32
// DIV-SMITH: %[[LHS_REAL_IS_INF_WITH_SIGN_TIMES_RHS_REAL:.*]] = llvm.fmul %[[LHS_REAL_IS_INF_WITH_SIGN]], %[[RHS_REAL]] : f32
// DIV-SMITH: %[[LHS_IMAG_IS_INF_WITH_SIGN_TIMES_RHS_IMAG:.*]] = llvm.fmul %[[LHS_IMAG_IS_INF_WITH_SIGN]], %[[RHS_IMAG]] : f32
// DIV-SMITH: %[[INF_MULTIPLICATOR_1:.*]] = llvm.fadd %[[LHS_REAL_IS_INF_WITH_SIGN_TIMES_RHS_REAL]], %[[LHS_IMAG_IS_INF_WITH_SIGN_TIMES_RHS_IMAG]] : f32
// DIV-SMITH: %[[RESULT_REAL_3:.*]] = llvm.fmul %[[INF]], %[[INF_MULTIPLICATOR_1]] : f32
// DIV-SMITH: %[[LHS_REAL_IS_INF_WITH_SIGN_TIMES_RHS_IMAG:.*]] = llvm.fmul %[[LHS_REAL_IS_INF_WITH_SIGN]], %[[RHS_IMAG]] : f32
// DIV-SMITH: %[[LHS_IMAG_IS_INF_WITH_SIGN_TIMES_RHS_REAL:.*]] = llvm.fmul %[[LHS_IMAG_IS_INF_WITH_SIGN]], %[[RHS_REAL]] : f32
// DIV-SMITH: %[[INF_MULTIPLICATOR_2:.*]] = llvm.fsub %[[LHS_IMAG_IS_INF_WITH_SIGN_TIMES_RHS_REAL]], %[[LHS_REAL_IS_INF_WITH_SIGN_TIMES_RHS_IMAG]] : f32
// DIV-SMITH: %[[RESULT_IMAG_3:.*]] = llvm.fmul %[[INF]], %[[INF_MULTIPLICATOR_2]] : f32

// Case 3. Finite numerator, infinite denominator.
// DIV-SMITH: %[[LHS_REAL_FINITE:.*]] = llvm.fcmp "one" %[[LHS_REAL_ABS]], %[[INF]] : f32
// DIV-SMITH: %[[LHS_IMAG_FINITE:.*]] = llvm.fcmp "one" %[[LHS_IMAG_ABS]], %[[INF]] : f32
// DIV-SMITH: %[[LHS_IS_FINITE:.*]] = llvm.and %[[LHS_REAL_FINITE]], %[[LHS_IMAG_FINITE]] : i1
// DIV-SMITH: %[[RHS_REAL_INFINITE:.*]] = llvm.fcmp "oeq" %[[RHS_REAL_ABS]], %[[INF]] : f32
// DIV-SMITH: %[[RHS_IMAG_INFINITE:.*]] = llvm.fcmp "oeq" %[[RHS_IMAG_ABS]], %[[INF]] : f32
// DIV-SMITH: %[[RHS_IS_INFINITE:.*]] = llvm.or %[[RHS_REAL_INFINITE]], %[[RHS_IMAG_INFINITE]] : i1
// DIV-SMITH: %[[FINITE_NUM_INFINITE_DENOM:.*]] = llvm.and %[[LHS_IS_FINITE]], %[[RHS_IS_INFINITE]] : i1
// DIV-SMITH: %[[RHS_REAL_IS_INF:.*]] = llvm.select %[[RHS_REAL_INFINITE]], %[[ONE]], %[[ZERO]] : i1, f32
// DIV-SMITH: %[[RHS_REAL_IS_INF_WITH_SIGN:.*]] = llvm.intr.copysign(%[[RHS_REAL_IS_INF]], %[[RHS_REAL]]) : (f32, f32) -> f32
// DIV-SMITH: %[[RHS_IMAG_IS_INF:.*]] = llvm.select %[[RHS_IMAG_INFINITE]], %[[ONE]], %[[ZERO]] : i1, f32
// DIV-SMITH: %[[RHS_IMAG_IS_INF_WITH_SIGN:.*]] = llvm.intr.copysign(%[[RHS_IMAG_IS_INF]], %[[RHS_IMAG]]) : (f32, f32) -> f32
// DIV-SMITH: %[[RHS_REAL_IS_INF_WITH_SIGN_TIMES_LHS_REAL:.*]] = llvm.fmul %[[LHS_REAL]], %[[RHS_REAL_IS_INF_WITH_SIGN]] : f32
// DIV-SMITH: %[[RHS_IMAG_IS_INF_WITH_SIGN_TIMES_LHS_IMAG:.*]] = llvm.fmul %[[LHS_IMAG]], %[[RHS_IMAG_IS_INF_WITH_SIGN]] : f32
// DIV-SMITH: %[[ZERO_MULTIPLICATOR_1:.*]] = llvm.fadd %[[RHS_REAL_IS_INF_WITH_SIGN_TIMES_LHS_REAL]], %[[RHS_IMAG_IS_INF_WITH_SIGN_TIMES_LHS_IMAG]] : f32
// DIV-SMITH: %[[RESULT_REAL_4:.*]] = llvm.fmul %[[ZERO]], %[[ZERO_MULTIPLICATOR_1]] : f32
// DIV-SMITH: %[[RHS_REAL_IS_INF_WITH_SIGN_TIMES_LHS_IMAG:.*]] = llvm.fmul %[[LHS_IMAG]], %[[RHS_REAL_IS_INF_WITH_SIGN]] : f32
// DIV-SMITH: %[[RHS_IMAG_IS_INF_WITH_SIGN_TIMES_LHS_REAL:.*]] = llvm.fmul %[[LHS_REAL]], %[[RHS_IMAG_IS_INF_WITH_SIGN]] : f32
// DIV-SMITH: %[[ZERO_MULTIPLICATOR_2:.*]] = llvm.fsub %[[RHS_REAL_IS_INF_WITH_SIGN_TIMES_LHS_IMAG]], %[[RHS_IMAG_IS_INF_WITH_SIGN_TIMES_LHS_REAL]] : f32
// DIV-SMITH: %[[RESULT_IMAG_4:.*]] = llvm.fmul %[[ZERO]], %[[ZERO_MULTIPLICATOR_2]] : f32

// DIV-SMITH: %[[REAL_ABS_SMALLER_THAN_IMAG_ABS:.*]] = llvm.fcmp "olt" %[[RHS_REAL_ABS]], %[[RHS_IMAG_ABS]] : f32
// DIV-SMITH: %[[RESULT_REAL:.*]] = llvm.select %[[REAL_ABS_SMALLER_THAN_IMAG_ABS]], %[[RESULT_REAL_1]], %[[RESULT_REAL_2]] : i1, f32
// DIV-SMITH: %[[RESULT_IMAG:.*]] = llvm.select %[[REAL_ABS_SMALLER_THAN_IMAG_ABS]], %[[RESULT_IMAG_1]], %[[RESULT_IMAG_2]] : i1, f32
// DIV-SMITH: %[[RESULT_REAL_SPECIAL_CASE_3:.*]] = llvm.select %[[FINITE_NUM_INFINITE_DENOM]], %[[RESULT_REAL_4]], %[[RESULT_REAL]] : i1, f32
// DIV-SMITH: %[[RESULT_IMAG_SPECIAL_CASE_3:.*]] = llvm.select %[[FINITE_NUM_INFINITE_DENOM]], %[[RESULT_IMAG_4]], %[[RESULT_IMAG]] : i1, f32
// DIV-SMITH: %[[RESULT_REAL_SPECIAL_CASE_2:.*]] = llvm.select %[[INF_NUM_FINITE_DENOM]], %[[RESULT_REAL_3]], %[[RESULT_REAL_SPECIAL_CASE_3]] : i1, f32
// DIV-SMITH: %[[RESULT_IMAG_SPECIAL_CASE_2:.*]] = llvm.select %[[INF_NUM_FINITE_DENOM]], %[[RESULT_IMAG_3]], %[[RESULT_IMAG_SPECIAL_CASE_3]] : i1, f32
// DIV-SMITH: %[[RESULT_REAL_SPECIAL_CASE_1:.*]] = llvm.select %[[RESULT_IS_INFINITY]], %[[INFINITY_RESULT_REAL]], %[[RESULT_REAL_SPECIAL_CASE_2]] : i1, f32
// DIV-SMITH: %[[RESULT_IMAG_SPECIAL_CASE_1:.*]] = llvm.select %[[RESULT_IS_INFINITY]], %[[INFINITY_RESULT_IMAG]], %[[RESULT_IMAG_SPECIAL_CASE_2]] : i1, f32
// DIV-SMITH: %[[RESULT_REAL_IS_NAN:.*]] = llvm.fcmp "uno" %[[RESULT_REAL]], %[[ZERO]] : f32
// DIV-SMITH: %[[RESULT_IMAG_IS_NAN:.*]] = llvm.fcmp "uno" %[[RESULT_IMAG]], %[[ZERO]] : f32
// DIV-SMITH: %[[RESULT_IS_NAN:.*]] = llvm.and %[[RESULT_REAL_IS_NAN]], %[[RESULT_IMAG_IS_NAN]] : i1
// DIV-SMITH: %[[RESULT_REAL_WITH_SPECIAL_CASES:.*]] = llvm.select %[[RESULT_IS_NAN]], %[[RESULT_REAL_SPECIAL_CASE_1]], %[[RESULT_REAL]] : i1, f32
// DIV-SMITH: %[[RESULT_IMAG_WITH_SPECIAL_CASES:.*]] = llvm.select %[[RESULT_IS_NAN]], %[[RESULT_IMAG_SPECIAL_CASE_1]], %[[RESULT_IMAG]] : i1, f32
// DIV-SMITH: %[[RESULT_1:.*]] = llvm.insertvalue %[[RESULT_REAL_WITH_SPECIAL_CASES]], %[[RESULT_0]][0] : ![[C_TY]]
// DIV-SMITH: %[[RESULT_2:.*]] = llvm.insertvalue %[[RESULT_IMAG_WITH_SPECIAL_CASES]], %[[RESULT_1]][1] : ![[C_TY]]
// DIV-SMITH: %[[CASTED_RESULT:.*]] = builtin.unrealized_conversion_cast %[[RESULT_2]] : ![[C_TY]] to complex<f32>
// DIV-SMITH: return %[[CASTED_RESULT]] : complex<f32>


// DIV-ALGEBRAIC-LABEL: func @complex_div
// DIV-ALGEBRAIC-SAME:    %[[LHS:.*]]: complex<f32>, %[[RHS:.*]]: complex<f32>
// DIV-ALGEBRAIC-DAG: %[[CASTED_LHS:.*]] = builtin.unrealized_conversion_cast %[[LHS]] : complex<f32> to ![[C_TY:.*>]]
// DIV-ALGEBRAIC-DAG: %[[CASTED_RHS:.*]] = builtin.unrealized_conversion_cast %[[RHS]] : complex<f32> to ![[C_TY]]

// DIV-ALGEBRAIC: %[[LHS_RE:.*]] = llvm.extractvalue %[[CASTED_LHS]][0] : ![[C_TY]]
// DIV-ALGEBRAIC: %[[LHS_IM:.*]] = llvm.extractvalue %[[CASTED_LHS]][1] : ![[C_TY]]
// DIV-ALGEBRAIC: %[[RHS_RE:.*]] = llvm.extractvalue %[[CASTED_RHS]][0] : ![[C_TY]]
// DIV-ALGEBRAIC: %[[RHS_IM:.*]] = llvm.extractvalue %[[CASTED_RHS]][1] : ![[C_TY]]

// DIV-ALGEBRAIC: %[[RESULT_0:.*]] = llvm.mlir.poison : ![[C_TY]]

// DIV-ALGEBRAIC-DAG: %[[RHS_RE_SQ:.*]] = llvm.fmul %[[RHS_RE]], %[[RHS_RE]]  : f32
// DIV-ALGEBRAIC-DAG: %[[RHS_IM_SQ:.*]] = llvm.fmul %[[RHS_IM]], %[[RHS_IM]]  : f32
// DIV-ALGEBRAIC: %[[SQ_NORM:.*]] = llvm.fadd %[[RHS_RE_SQ]], %[[RHS_IM_SQ]]  : f32

// DIV-ALGEBRAIC-DAG: %[[REAL_TMP_0:.*]] = llvm.fmul %[[LHS_RE]], %[[RHS_RE]]  : f32
// DIV-ALGEBRAIC-DAG: %[[REAL_TMP_1:.*]] = llvm.fmul %[[LHS_IM]], %[[RHS_IM]]  : f32
// DIV-ALGEBRAIC: %[[REAL_TMP_2:.*]] = llvm.fadd %[[REAL_TMP_0]], %[[REAL_TMP_1]]  : f32

// DIV-ALGEBRAIC-DAG: %[[IMAG_TMP_0:.*]] = llvm.fmul %[[LHS_IM]], %[[RHS_RE]]  : f32
// DIV-ALGEBRAIC-DAG: %[[IMAG_TMP_1:.*]] = llvm.fmul %[[LHS_RE]], %[[RHS_IM]]  : f32
// DIV-ALGEBRAIC: %[[IMAG_TMP_2:.*]] = llvm.fsub %[[IMAG_TMP_0]], %[[IMAG_TMP_1]]  : f32

// DIV-ALGEBRAIC: %[[REAL:.*]] = llvm.fdiv %[[REAL_TMP_2]], %[[SQ_NORM]]  : f32
// DIV-ALGEBRAIC: %[[IMAG:.*]] = llvm.fdiv %[[IMAG_TMP_2]], %[[SQ_NORM]]  : f32
// DIV-ALGEBRAIC: %[[RESULT_1:.*]] = llvm.insertvalue %[[REAL]], %[[RESULT_0]][0] : ![[C_TY]]
// DIV-ALGEBRAIC: %[[RESULT_2:.*]] = llvm.insertvalue %[[IMAG]], %[[RESULT_1]][1] : ![[C_TY]]
//
// DIV-ALGEBRAIC: %[[CASTED_RESULT:.*]] = builtin.unrealized_conversion_cast %[[RESULT_2]] : ![[C_TY]] to complex<f32>
// DIV-ALGEBRAIC: return %[[CASTED_RESULT]] : complex<f32>


func.func @complex_div_with_fmf(%lhs: complex<f32>, %rhs: complex<f32>) -> complex<f32> {
  %div = complex.div %lhs, %rhs fastmath<nsz,arcp> : complex<f32>
  return %div : complex<f32>
}
// DIV-SMITH-LABEL: func @complex_div_with_fmf
// DIV-SMITH-SAME:    %[[LHS:.*]]: complex<f32>, %[[RHS:.*]]: complex<f32>
// DIV-SMITH-DAG: %[[CASTED_LHS:.*]] = builtin.unrealized_conversion_cast %[[LHS]] : complex<f32> to ![[C_TY:.*>]]
// DIV-SMITH-DAG: %[[CASTED_RHS:.*]] = builtin.unrealized_conversion_cast %[[RHS]] : complex<f32> to ![[C_TY]]

// DIV-SMITH: %[[LHS_REAL:.*]] = llvm.extractvalue %[[CASTED_LHS]][0] : ![[C_TY]]
// DIV-SMITH: %[[LHS_IMAG:.*]] = llvm.extractvalue %[[CASTED_LHS]][1] : ![[C_TY]]
// DIV-SMITH: %[[RHS_REAL:.*]] = llvm.extractvalue %[[CASTED_RHS]][0] : ![[C_TY]]
// DIV-SMITH: %[[RHS_IMAG:.*]] = llvm.extractvalue %[[CASTED_RHS]][1] : ![[C_TY]]

// DIV-SMITH: %[[RESULT_0:.*]] = llvm.mlir.poison : ![[C_TY]]

// DIV-SMITH: %[[RHS_REAL_IMAG_RATIO:.*]] = llvm.fdiv %[[RHS_REAL]], %[[RHS_IMAG]] {fastmathFlags = #llvm.fastmath<nsz, arcp>} : f32
// DIV-SMITH: %[[RHS_REAL_TIMES_RHS_REAL_IMAG_RATIO:.*]] = llvm.fmul %[[RHS_REAL_IMAG_RATIO]], %[[RHS_REAL]] {fastmathFlags = #llvm.fastmath<nsz, arcp>} : f32
// DIV-SMITH: %[[RHS_REAL_IMAG_DENOM:.*]] = llvm.fadd %[[RHS_IMAG]], %[[RHS_REAL_TIMES_RHS_REAL_IMAG_RATIO]] {fastmathFlags = #llvm.fastmath<nsz, arcp>} : f32
// DIV-SMITH: %[[LHS_REAL_TIMES_RHS_REAL_IMAG_RATIO:.*]] = llvm.fmul %[[LHS_REAL]], %[[RHS_REAL_IMAG_RATIO]] {fastmathFlags = #llvm.fastmath<nsz, arcp>} : f32
// DIV-SMITH: %[[REAL_NUMERATOR_1:.*]] = llvm.fadd %[[LHS_REAL_TIMES_RHS_REAL_IMAG_RATIO]], %[[LHS_IMAG]] {fastmathFlags = #llvm.fastmath<nsz, arcp>} : f32
// DIV-SMITH: %[[RESULT_REAL_1:.*]] = llvm.fdiv %[[REAL_NUMERATOR_1]], %[[RHS_REAL_IMAG_DENOM]] {fastmathFlags = #llvm.fastmath<nsz, arcp>} : f32
// DIV-SMITH: %[[LHS_IMAG_TIMES_RHS_REAL_IMAG_RATIO:.*]] = llvm.fmul %[[LHS_IMAG]], %[[RHS_REAL_IMAG_RATIO]] {fastmathFlags = #llvm.fastmath<nsz, arcp>} : f32
// DIV-SMITH: %[[IMAG_NUMERATOR_1:.*]] = llvm.fsub %[[LHS_IMAG_TIMES_RHS_REAL_IMAG_RATIO]], %[[LHS_REAL]] {fastmathFlags = #llvm.fastmath<nsz, arcp>} : f32
// DIV-SMITH: %[[RESULT_IMAG_1:.*]] = llvm.fdiv %[[IMAG_NUMERATOR_1]], %[[RHS_REAL_IMAG_DENOM]] {fastmathFlags = #llvm.fastmath<nsz, arcp>} : f32

// DIV-SMITH: %[[RHS_IMAG_REAL_RATIO:.*]] = llvm.fdiv %[[RHS_IMAG]], %[[RHS_REAL]] {fastmathFlags = #llvm.fastmath<nsz, arcp>} : f32
// DIV-SMITH: %[[RHS_IMAG_TIMES_RHS_IMAG_REAL_RATIO:.*]] = llvm.fmul %[[RHS_IMAG_REAL_RATIO]], %[[RHS_IMAG]] {fastmathFlags = #llvm.fastmath<nsz, arcp>} : f32
// DIV-SMITH: %[[RHS_IMAG_REAL_DENOM:.*]] = llvm.fadd %[[RHS_REAL]], %[[RHS_IMAG_TIMES_RHS_IMAG_REAL_RATIO]] {fastmathFlags = #llvm.fastmath<nsz, arcp>} : f32
// DIV-SMITH: %[[LHS_IMAG_TIMES_RHS_IMAG_REAL_RATIO:.*]] = llvm.fmul %[[LHS_IMAG]], %[[RHS_IMAG_REAL_RATIO]] {fastmathFlags = #llvm.fastmath<nsz, arcp>} : f32
// DIV-SMITH: %[[REAL_NUMERATOR_2:.*]] = llvm.fadd %[[LHS_REAL]], %[[LHS_IMAG_TIMES_RHS_IMAG_REAL_RATIO]] {fastmathFlags = #llvm.fastmath<nsz, arcp>} : f32
// DIV-SMITH: %[[RESULT_REAL_2:.*]] = llvm.fdiv %[[REAL_NUMERATOR_2]], %[[RHS_IMAG_REAL_DENOM]] {fastmathFlags = #llvm.fastmath<nsz, arcp>} : f32
// DIV-SMITH: %[[LHS_REAL_TIMES_RHS_IMAG_REAL_RATIO:.*]] = llvm.fmul %[[LHS_REAL]], %[[RHS_IMAG_REAL_RATIO]] {fastmathFlags = #llvm.fastmath<nsz, arcp>} : f32
// DIV-SMITH: %[[IMAG_NUMERATOR_2:.*]] = llvm.fsub %[[LHS_IMAG]], %[[LHS_REAL_TIMES_RHS_IMAG_REAL_RATIO]] {fastmathFlags = #llvm.fastmath<nsz, arcp>} : f32
// DIV-SMITH: %[[RESULT_IMAG_2:.*]] = llvm.fdiv %[[IMAG_NUMERATOR_2]], %[[RHS_IMAG_REAL_DENOM]] {fastmathFlags = #llvm.fastmath<nsz, arcp>} : f32

// Case 1. Zero denominator, numerator contains at most one NaN value.
// DIV-SMITH: %[[ZERO:.*]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// DIV-SMITH: %[[RHS_REAL_ABS:.*]] = llvm.intr.fabs(%[[RHS_REAL]]) {fastmathFlags = #llvm.fastmath<nsz, arcp>} : (f32) -> f32
// DIV-SMITH: %[[RHS_REAL_ABS_IS_ZERO:.*]] = llvm.fcmp "oeq" %[[RHS_REAL_ABS]], %[[ZERO]] : f32
// DIV-SMITH: %[[RHS_IMAG_ABS:.*]] = llvm.intr.fabs(%[[RHS_IMAG]]) {fastmathFlags = #llvm.fastmath<nsz, arcp>} : (f32) -> f32
// DIV-SMITH: %[[RHS_IMAG_ABS_IS_ZERO:.*]] = llvm.fcmp "oeq" %[[RHS_IMAG_ABS]], %[[ZERO]] : f32
// DIV-SMITH: %[[LHS_REAL_IS_NOT_NAN:.*]] = llvm.fcmp "ord" %[[LHS_REAL]], %[[ZERO]] : f32
// DIV-SMITH: %[[LHS_IMAG_IS_NOT_NAN:.*]] = llvm.fcmp "ord" %[[LHS_IMAG]], %[[ZERO]] : f32
// DIV-SMITH: %[[LHS_CONTAINS_NOT_NAN_VALUE:.*]] = llvm.or %[[LHS_REAL_IS_NOT_NAN]], %[[LHS_IMAG_IS_NOT_NAN]] : i1
// DIV-SMITH: %[[RHS_IS_ZERO:.*]] = llvm.and %[[RHS_REAL_ABS_IS_ZERO]], %[[RHS_IMAG_ABS_IS_ZERO]] : i1
// DIV-SMITH: %[[RESULT_IS_INFINITY:.*]] = llvm.and %[[LHS_CONTAINS_NOT_NAN_VALUE]], %[[RHS_IS_ZERO]] : i1
// DIV-SMITH: %[[INF:.*]] = llvm.mlir.constant(0x7F800000 : f32) : f32
// DIV-SMITH: %[[INF_WITH_SIGN_OF_RHS_REAL:.*]] = llvm.intr.copysign(%[[INF]], %[[RHS_REAL]]) : (f32, f32) -> f32
// DIV-SMITH: %[[INFINITY_RESULT_REAL:.*]] = llvm.fmul %[[INF_WITH_SIGN_OF_RHS_REAL]], %[[LHS_REAL]] {fastmathFlags = #llvm.fastmath<nsz, arcp>} : f32
// DIV-SMITH: %[[INFINITY_RESULT_IMAG:.*]] = llvm.fmul %[[INF_WITH_SIGN_OF_RHS_REAL]], %[[LHS_IMAG]] {fastmathFlags = #llvm.fastmath<nsz, arcp>} : f32

// Case 2. Infinite numerator, finite denominator.
// DIV-SMITH: %[[RHS_REAL_FINITE:.*]] = llvm.fcmp "one" %[[RHS_REAL_ABS]], %[[INF]] : f32
// DIV-SMITH: %[[RHS_IMAG_FINITE:.*]] = llvm.fcmp "one" %[[RHS_IMAG_ABS]], %[[INF]] : f32
// DIV-SMITH: %[[RHS_IS_FINITE:.*]] = llvm.and %[[RHS_REAL_FINITE]], %[[RHS_IMAG_FINITE]] : i1
// DIV-SMITH: %[[LHS_REAL_ABS:.*]] = llvm.intr.fabs(%[[LHS_REAL]]) {fastmathFlags = #llvm.fastmath<nsz, arcp>} : (f32) -> f32
// DIV-SMITH: %[[LHS_REAL_INFINITE:.*]] = llvm.fcmp "oeq" %[[LHS_REAL_ABS]], %[[INF]] : f32
// DIV-SMITH: %[[LHS_IMAG_ABS:.*]] = llvm.intr.fabs(%[[LHS_IMAG]]) {fastmathFlags = #llvm.fastmath<nsz, arcp>} : (f32) -> f32
// DIV-SMITH: %[[LHS_IMAG_INFINITE:.*]] = llvm.fcmp "oeq" %[[LHS_IMAG_ABS]], %[[INF]] : f32
// DIV-SMITH: %[[LHS_IS_INFINITE:.*]] = llvm.or %[[LHS_REAL_INFINITE]], %[[LHS_IMAG_INFINITE]] : i1
// DIV-SMITH: %[[INF_NUM_FINITE_DENOM:.*]] = llvm.and %[[LHS_IS_INFINITE]], %[[RHS_IS_FINITE]] : i1
// DIV-SMITH: %[[ONE:.*]] = llvm.mlir.constant(1.000000e+00 : f32) : f32
// DIV-SMITH: %[[LHS_REAL_IS_INF:.*]] = llvm.select %[[LHS_REAL_INFINITE]], %[[ONE]], %[[ZERO]] : i1, f32
// DIV-SMITH: %[[LHS_REAL_IS_INF_WITH_SIGN:.*]] = llvm.intr.copysign(%[[LHS_REAL_IS_INF]], %[[LHS_REAL]]) : (f32, f32) -> f32
// DIV-SMITH: %[[LHS_IMAG_IS_INF:.*]] = llvm.select %[[LHS_IMAG_INFINITE]], %[[ONE]], %[[ZERO]] : i1, f32
// DIV-SMITH: %[[LHS_IMAG_IS_INF_WITH_SIGN:.*]] = llvm.intr.copysign(%[[LHS_IMAG_IS_INF]], %[[LHS_IMAG]]) : (f32, f32) -> f32
// DIV-SMITH: %[[LHS_REAL_IS_INF_WITH_SIGN_TIMES_RHS_REAL:.*]] = llvm.fmul %[[LHS_REAL_IS_INF_WITH_SIGN]], %[[RHS_REAL]] {fastmathFlags = #llvm.fastmath<nsz, arcp>} : f32
// DIV-SMITH: %[[LHS_IMAG_IS_INF_WITH_SIGN_TIMES_RHS_IMAG:.*]] = llvm.fmul %[[LHS_IMAG_IS_INF_WITH_SIGN]], %[[RHS_IMAG]] {fastmathFlags = #llvm.fastmath<nsz, arcp>} : f32
// DIV-SMITH: %[[INF_MULTIPLICATOR_1:.*]] = llvm.fadd %[[LHS_REAL_IS_INF_WITH_SIGN_TIMES_RHS_REAL]], %[[LHS_IMAG_IS_INF_WITH_SIGN_TIMES_RHS_IMAG]] {fastmathFlags = #llvm.fastmath<nsz, arcp>} : f32
// DIV-SMITH: %[[RESULT_REAL_3:.*]] = llvm.fmul %[[INF]], %[[INF_MULTIPLICATOR_1]] {fastmathFlags = #llvm.fastmath<nsz, arcp>} : f32
// DIV-SMITH: %[[LHS_REAL_IS_INF_WITH_SIGN_TIMES_RHS_IMAG:.*]] = llvm.fmul %[[LHS_REAL_IS_INF_WITH_SIGN]], %[[RHS_IMAG]] {fastmathFlags = #llvm.fastmath<nsz, arcp>} : f32
// DIV-SMITH: %[[LHS_IMAG_IS_INF_WITH_SIGN_TIMES_RHS_REAL:.*]] = llvm.fmul %[[LHS_IMAG_IS_INF_WITH_SIGN]], %[[RHS_REAL]] {fastmathFlags = #llvm.fastmath<nsz, arcp>} : f32
// DIV-SMITH: %[[INF_MULTIPLICATOR_2:.*]] = llvm.fsub %[[LHS_IMAG_IS_INF_WITH_SIGN_TIMES_RHS_REAL]], %[[LHS_REAL_IS_INF_WITH_SIGN_TIMES_RHS_IMAG]] {fastmathFlags = #llvm.fastmath<nsz, arcp>} : f32
// DIV-SMITH: %[[RESULT_IMAG_3:.*]] = llvm.fmul %[[INF]], %[[INF_MULTIPLICATOR_2]] {fastmathFlags = #llvm.fastmath<nsz, arcp>} : f32

// Case 3. Finite numerator, infinite denominator.
// DIV-SMITH: %[[LHS_REAL_FINITE:.*]] = llvm.fcmp "one" %[[LHS_REAL_ABS]], %[[INF]] : f32
// DIV-SMITH: %[[LHS_IMAG_FINITE:.*]] = llvm.fcmp "one" %[[LHS_IMAG_ABS]], %[[INF]] : f32
// DIV-SMITH: %[[LHS_IS_FINITE:.*]] = llvm.and %[[LHS_REAL_FINITE]], %[[LHS_IMAG_FINITE]] : i1
// DIV-SMITH: %[[RHS_REAL_INFINITE:.*]] = llvm.fcmp "oeq" %[[RHS_REAL_ABS]], %[[INF]] : f32
// DIV-SMITH: %[[RHS_IMAG_INFINITE:.*]] = llvm.fcmp "oeq" %[[RHS_IMAG_ABS]], %[[INF]] : f32
// DIV-SMITH: %[[RHS_IS_INFINITE:.*]] = llvm.or %[[RHS_REAL_INFINITE]], %[[RHS_IMAG_INFINITE]] : i1
// DIV-SMITH: %[[FINITE_NUM_INFINITE_DENOM:.*]] = llvm.and %[[LHS_IS_FINITE]], %[[RHS_IS_INFINITE]] : i1
// DIV-SMITH: %[[RHS_REAL_IS_INF:.*]] = llvm.select %[[RHS_REAL_INFINITE]], %[[ONE]], %[[ZERO]] : i1, f32
// DIV-SMITH: %[[RHS_REAL_IS_INF_WITH_SIGN:.*]] = llvm.intr.copysign(%[[RHS_REAL_IS_INF]], %[[RHS_REAL]]) : (f32, f32) -> f32
// DIV-SMITH: %[[RHS_IMAG_IS_INF:.*]] = llvm.select %[[RHS_IMAG_INFINITE]], %[[ONE]], %[[ZERO]] : i1, f32
// DIV-SMITH: %[[RHS_IMAG_IS_INF_WITH_SIGN:.*]] = llvm.intr.copysign(%[[RHS_IMAG_IS_INF]], %[[RHS_IMAG]]) : (f32, f32) -> f32
// DIV-SMITH: %[[RHS_REAL_IS_INF_WITH_SIGN_TIMES_LHS_REAL:.*]] = llvm.fmul %[[LHS_REAL]], %[[RHS_REAL_IS_INF_WITH_SIGN]] {fastmathFlags = #llvm.fastmath<nsz, arcp>} : f32
// DIV-SMITH: %[[RHS_IMAG_IS_INF_WITH_SIGN_TIMES_LHS_IMAG:.*]] = llvm.fmul %[[LHS_IMAG]], %[[RHS_IMAG_IS_INF_WITH_SIGN]] {fastmathFlags = #llvm.fastmath<nsz, arcp>} : f32
// DIV-SMITH: %[[ZERO_MULTIPLICATOR_1:.*]] = llvm.fadd %[[RHS_REAL_IS_INF_WITH_SIGN_TIMES_LHS_REAL]], %[[RHS_IMAG_IS_INF_WITH_SIGN_TIMES_LHS_IMAG]] {fastmathFlags = #llvm.fastmath<nsz, arcp>} : f32
// DIV-SMITH: %[[RESULT_REAL_4:.*]] = llvm.fmul %[[ZERO]], %[[ZERO_MULTIPLICATOR_1]] {fastmathFlags = #llvm.fastmath<nsz, arcp>} : f32
// DIV-SMITH: %[[RHS_REAL_IS_INF_WITH_SIGN_TIMES_LHS_IMAG:.*]] = llvm.fmul %[[LHS_IMAG]], %[[RHS_REAL_IS_INF_WITH_SIGN]] {fastmathFlags = #llvm.fastmath<nsz, arcp>} : f32
// DIV-SMITH: %[[RHS_IMAG_IS_INF_WITH_SIGN_TIMES_LHS_REAL:.*]] = llvm.fmul %[[LHS_REAL]], %[[RHS_IMAG_IS_INF_WITH_SIGN]] {fastmathFlags = #llvm.fastmath<nsz, arcp>} : f32
// DIV-SMITH: %[[ZERO_MULTIPLICATOR_2:.*]] = llvm.fsub %[[RHS_REAL_IS_INF_WITH_SIGN_TIMES_LHS_IMAG]], %[[RHS_IMAG_IS_INF_WITH_SIGN_TIMES_LHS_REAL]] {fastmathFlags = #llvm.fastmath<nsz, arcp>} : f32
// DIV-SMITH: %[[RESULT_IMAG_4:.*]] = llvm.fmul %[[ZERO]], %[[ZERO_MULTIPLICATOR_2]] {fastmathFlags = #llvm.fastmath<nsz, arcp>} : f32

// DIV-SMITH: %[[REAL_ABS_SMALLER_THAN_IMAG_ABS:.*]] = llvm.fcmp "olt" %[[RHS_REAL_ABS]], %[[RHS_IMAG_ABS]] : f32
// DIV-SMITH: %[[RESULT_REAL:.*]] = llvm.select %[[REAL_ABS_SMALLER_THAN_IMAG_ABS]], %[[RESULT_REAL_1]], %[[RESULT_REAL_2]] : i1, f32
// DIV-SMITH: %[[RESULT_IMAG:.*]] = llvm.select %[[REAL_ABS_SMALLER_THAN_IMAG_ABS]], %[[RESULT_IMAG_1]], %[[RESULT_IMAG_2]] : i1, f32
// DIV-SMITH: %[[RESULT_REAL_SPECIAL_CASE_3:.*]] = llvm.select %[[FINITE_NUM_INFINITE_DENOM]], %[[RESULT_REAL_4]], %[[RESULT_REAL]] : i1, f32
// DIV-SMITH: %[[RESULT_IMAG_SPECIAL_CASE_3:.*]] = llvm.select %[[FINITE_NUM_INFINITE_DENOM]], %[[RESULT_IMAG_4]], %[[RESULT_IMAG]] : i1, f32
// DIV-SMITH: %[[RESULT_REAL_SPECIAL_CASE_2:.*]] = llvm.select %[[INF_NUM_FINITE_DENOM]], %[[RESULT_REAL_3]], %[[RESULT_REAL_SPECIAL_CASE_3]] : i1, f32
// DIV-SMITH: %[[RESULT_IMAG_SPECIAL_CASE_2:.*]] = llvm.select %[[INF_NUM_FINITE_DENOM]], %[[RESULT_IMAG_3]], %[[RESULT_IMAG_SPECIAL_CASE_3]] : i1, f32
// DIV-SMITH: %[[RESULT_REAL_SPECIAL_CASE_1:.*]] = llvm.select %[[RESULT_IS_INFINITY]], %[[INFINITY_RESULT_REAL]], %[[RESULT_REAL_SPECIAL_CASE_2]] : i1, f32
// DIV-SMITH: %[[RESULT_IMAG_SPECIAL_CASE_1:.*]] = llvm.select %[[RESULT_IS_INFINITY]], %[[INFINITY_RESULT_IMAG]], %[[RESULT_IMAG_SPECIAL_CASE_2]] : i1, f32
// DIV-SMITH: %[[RESULT_REAL_IS_NAN:.*]] = llvm.fcmp "uno" %[[RESULT_REAL]], %[[ZERO]] : f32
// DIV-SMITH: %[[RESULT_IMAG_IS_NAN:.*]] = llvm.fcmp "uno" %[[RESULT_IMAG]], %[[ZERO]] : f32
// DIV-SMITH: %[[RESULT_IS_NAN:.*]] = llvm.and %[[RESULT_REAL_IS_NAN]], %[[RESULT_IMAG_IS_NAN]] : i1
// DIV-SMITH: %[[RESULT_REAL_WITH_SPECIAL_CASES:.*]] = llvm.select %[[RESULT_IS_NAN]], %[[RESULT_REAL_SPECIAL_CASE_1]], %[[RESULT_REAL]] : i1, f32
// DIV-SMITH: %[[RESULT_IMAG_WITH_SPECIAL_CASES:.*]] = llvm.select %[[RESULT_IS_NAN]], %[[RESULT_IMAG_SPECIAL_CASE_1]], %[[RESULT_IMAG]] : i1, f32
// DIV-SMITH: %[[RESULT_1:.*]] = llvm.insertvalue %[[RESULT_REAL_WITH_SPECIAL_CASES]], %[[RESULT_0]][0] : ![[C_TY]]
// DIV-SMITH: %[[RESULT_2:.*]] = llvm.insertvalue %[[RESULT_IMAG_WITH_SPECIAL_CASES]], %[[RESULT_1]][1] : ![[C_TY]]
// DIV-SMITH: %[[CASTED_RESULT:.*]] = builtin.unrealized_conversion_cast %[[RESULT_2]] : ![[C_TY]] to complex<f32>
// DIV-SMITH: return %[[CASTED_RESULT]] : complex<f32>


// DIV-ALGEBRAIC-LABEL: func @complex_div_with_fmf
// DIV-ALGEBRAIC-SAME:    %[[LHS:.*]]: complex<f32>, %[[RHS:.*]]: complex<f32>
// DIV-ALGEBRAIC-DAG: %[[CASTED_LHS:.*]] = builtin.unrealized_conversion_cast %[[LHS]] : complex<f32> to ![[C_TY:.*>]]
// DIV-ALGEBRAIC-DAG: %[[CASTED_RHS:.*]] = builtin.unrealized_conversion_cast %[[RHS]] : complex<f32> to ![[C_TY]]

// DIV-ALGEBRAIC: %[[LHS_RE:.*]] = llvm.extractvalue %[[CASTED_LHS]][0] : ![[C_TY]]
// DIV-ALGEBRAIC: %[[LHS_IM:.*]] = llvm.extractvalue %[[CASTED_LHS]][1] : ![[C_TY]]
// DIV-ALGEBRAIC: %[[RHS_RE:.*]] = llvm.extractvalue %[[CASTED_RHS]][0] : ![[C_TY]]
// DIV-ALGEBRAIC: %[[RHS_IM:.*]] = llvm.extractvalue %[[CASTED_RHS]][1] : ![[C_TY]]

// DIV-ALGEBRAIC: %[[RESULT_0:.*]] = llvm.mlir.poison : ![[C_TY]]

// DIV-ALGEBRAIC-DAG: %[[RHS_RE_SQ:.*]] = llvm.fmul %[[RHS_RE]], %[[RHS_RE]] {fastmathFlags = #llvm.fastmath<nsz, arcp>} : f32
// DIV-ALGEBRAIC-DAG: %[[RHS_IM_SQ:.*]] = llvm.fmul %[[RHS_IM]], %[[RHS_IM]] {fastmathFlags = #llvm.fastmath<nsz, arcp>} : f32
// DIV-ALGEBRAIC: %[[SQ_NORM:.*]] = llvm.fadd %[[RHS_RE_SQ]], %[[RHS_IM_SQ]] {fastmathFlags = #llvm.fastmath<nsz, arcp>} : f32

// DIV-ALGEBRAIC-DAG: %[[REAL_TMP_0:.*]] = llvm.fmul %[[LHS_RE]], %[[RHS_RE]] {fastmathFlags = #llvm.fastmath<nsz, arcp>} : f32
// DIV-ALGEBRAIC-DAG: %[[REAL_TMP_1:.*]] = llvm.fmul %[[LHS_IM]], %[[RHS_IM]] {fastmathFlags = #llvm.fastmath<nsz, arcp>} : f32
// DIV-ALGEBRAIC: %[[REAL_TMP_2:.*]] = llvm.fadd %[[REAL_TMP_0]], %[[REAL_TMP_1]] {fastmathFlags = #llvm.fastmath<nsz, arcp>} : f32

// DIV-ALGEBRAIC-DAG: %[[IMAG_TMP_0:.*]] = llvm.fmul %[[LHS_IM]], %[[RHS_RE]] {fastmathFlags = #llvm.fastmath<nsz, arcp>} : f32
// DIV-ALGEBRAIC-DAG: %[[IMAG_TMP_1:.*]] = llvm.fmul %[[LHS_RE]], %[[RHS_IM]] {fastmathFlags = #llvm.fastmath<nsz, arcp>} : f32
// DIV-ALGEBRAIC: %[[IMAG_TMP_2:.*]] = llvm.fsub %[[IMAG_TMP_0]], %[[IMAG_TMP_1]] {fastmathFlags = #llvm.fastmath<nsz, arcp>} : f32

// DIV-ALGEBRAIC: %[[REAL:.*]] = llvm.fdiv %[[REAL_TMP_2]], %[[SQ_NORM]] {fastmathFlags = #llvm.fastmath<nsz, arcp>} : f32
// DIV-ALGEBRAIC: %[[IMAG:.*]] = llvm.fdiv %[[IMAG_TMP_2]], %[[SQ_NORM]] {fastmathFlags = #llvm.fastmath<nsz, arcp>} : f32
// DIV-ALGEBRAIC: %[[RESULT_1:.*]] = llvm.insertvalue %[[REAL]], %[[RESULT_0]][0] : ![[C_TY]]
// DIV-ALGEBRAIC: %[[RESULT_2:.*]] = llvm.insertvalue %[[IMAG]], %[[RESULT_1]][1] : ![[C_TY]]
//
// DIV-ALGEBRAIC: %[[CASTED_RESULT:.*]] = builtin.unrealized_conversion_cast %[[RESULT_2]] : ![[C_TY]] to complex<f32>
// DIV-ALGEBRAIC: return %[[CASTED_RESULT]] : complex<f32>
