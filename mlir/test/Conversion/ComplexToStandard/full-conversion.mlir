// RUN: mlir-opt %s -pass-pipeline="builtin.module(func.func(convert-complex-to-standard),convert-complex-to-llvm,func.func(convert-math-to-llvm,convert-arith-to-llvm),convert-func-to-llvm,reconcile-unrealized-casts)" | FileCheck %s

// CHECK-LABEL: llvm.func @complex_abs
// CHECK-SAME: %[[ARG:.*]]: ![[C_TY:.*]])
func.func @complex_abs(%arg: complex<f32>) -> f32 {
  %abs = complex.abs %arg: complex<f32>
  return %abs : f32
}
// CHECK: %[[REAL:.*]] = llvm.extractvalue %[[ARG]][0] : ![[C_TY]]
// CHECK: %[[IMAG:.*]] = llvm.extractvalue %[[ARG]][1] : ![[C_TY]]

// CHECK: %[[ONE:.*]] = llvm.mlir.constant(1.000000e+00 : f32) : f32
// CHECK: %[[ABS_REAL:.*]] = llvm.intr.fabs(%[[REAL]]) : (f32) -> f32
// CHECK: %[[ABS_IMAG:.*]] = llvm.intr.fabs(%[[IMAG]]) : (f32) -> f32
// CHECK: %[[MAX:.*]] = llvm.intr.maximum(%[[ABS_REAL]], %[[ABS_IMAG]]) : (f32, f32) -> f32
// CHECK: %[[MIN:.*]] = llvm.intr.minimum(%[[ABS_REAL]], %[[ABS_IMAG]]) : (f32, f32) -> f32
// CHECK: %[[RATIO:.*]] = llvm.fdiv %[[MIN]], %[[MAX]] : f32
// CHECK: %[[RATIO_SQ:.*]] = llvm.fmul %[[RATIO]], %[[RATIO]] : f32
// CHECK: %[[RATIO_SQ_PLUS_ONE:.*]] = llvm.fadd %[[RATIO_SQ]], %[[ONE]] : f32
// CHECK: %[[SQRT:.*]] = llvm.intr.sqrt(%[[RATIO_SQ_PLUS_ONE]]) : (f32) -> f32
// CHECK: %[[RESULT:.*]] = llvm.fmul %[[MAX]], %[[SQRT]] : f32
// CHECK: %[[IS_NAN:.*]] = llvm.fcmp "uno" %[[RESULT]], %11 : f32
// CHECK: %[[RET:.*]] = llvm.select %[[IS_NAN]], %[[MIN]], %[[RESULT]] : i1, f32
// CHECK: llvm.return %[[RET]] : f32

// CHECK-LABEL: llvm.func @complex_eq
// CHECK-SAME: %[[LHS:.*]]: ![[C_TY:.*]], %[[RHS:.*]]: ![[C_TY:.*]])
func.func @complex_eq(%lhs: complex<f32>, %rhs: complex<f32>) -> i1 {
  %eq = complex.eq %lhs, %rhs: complex<f32>
  return %eq : i1
}
// CHECK: %[[REAL_LHS:.*]] = llvm.extractvalue %[[LHS]][0] : ![[C_TY]]
// CHECK: %[[IMAG_LHS:.*]] = llvm.extractvalue %[[LHS]][1] : ![[C_TY]]
// CHECK: %[[REAL_RHS:.*]] = llvm.extractvalue %[[RHS]][0] : ![[C_TY]]
// CHECK: %[[IMAG_RHS:.*]] = llvm.extractvalue %[[RHS]][1] : ![[C_TY]]
// CHECK-DAG: %[[REAL_EQUAL:.*]] = llvm.fcmp "oeq" %[[REAL_LHS]], %[[REAL_RHS]]  : f32
// CHECK-DAG: %[[IMAG_EQUAL:.*]] = llvm.fcmp "oeq" %[[IMAG_LHS]], %[[IMAG_RHS]]  : f32
// CHECK: %[[EQUAL:.*]] = llvm.and %[[REAL_EQUAL]], %[[IMAG_EQUAL]] : i1
// CHECK: llvm.return %[[EQUAL]] : i1

// CHECK-LABEL: llvm.func @complex_neq
// CHECK-SAME: %[[LHS:.*]]: ![[C_TY:.*]], %[[RHS:.*]]: ![[C_TY:.*]])
func.func @complex_neq(%lhs: complex<f32>, %rhs: complex<f32>) -> i1 {
  %neq = complex.neq %lhs, %rhs: complex<f32>
  return %neq : i1
}
// CHECK: %[[REAL_LHS:.*]] = llvm.extractvalue %[[LHS]][0] : ![[C_TY]]
// CHECK: %[[IMAG_LHS:.*]] = llvm.extractvalue %[[LHS]][1] : ![[C_TY]]
// CHECK: %[[REAL_RHS:.*]] = llvm.extractvalue %[[RHS]][0] : ![[C_TY]]
// CHECK: %[[IMAG_RHS:.*]] = llvm.extractvalue %[[RHS]][1] : ![[C_TY]]
// CHECK-DAG: %[[REAL_NOT_EQUAL:.*]] = llvm.fcmp "une" %[[REAL_LHS]], %[[REAL_RHS]]  : f32
// CHECK-DAG: %[[IMAG_NOT_EQUAL:.*]] = llvm.fcmp "une" %[[IMAG_LHS]], %[[IMAG_RHS]]  : f32
// CHECK: %[[NOT_EQUAL:.*]] = llvm.or %[[REAL_NOT_EQUAL]], %[[IMAG_NOT_EQUAL]] : i1
// CHECK: llvm.return %[[NOT_EQUAL]] : i1
