// RUN: mlir-opt %s -pass-pipeline="builtin.module(func.func(convert-complex-to-standard),convert-complex-to-llvm,func.func(convert-math-to-llvm,convert-arith-to-llvm),convert-func-to-llvm,reconcile-unrealized-casts)" | FileCheck %s

// CHECK-LABEL: llvm.func @complex_abs
// CHECK-SAME: %[[ARG:.*]]: ![[C_TY:.*]])
func.func @complex_abs(%arg: complex<f32>) -> f32 {
  %abs = complex.abs %arg: complex<f32>
  return %abs : f32
}
// CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK: %[[ONE:.*]] = llvm.mlir.constant(1.000000e+00 : f32) : f32
// CHECK: %[[REAL:.*]] = llvm.extractvalue %[[ARG]][0] : ![[C_TY]]
// CHECK: %[[IMAG:.*]] = llvm.extractvalue %[[ARG]][1] : ![[C_TY]]
// CHECK: %[[REAL_IS_ZERO:.*]] = llvm.fcmp "oeq" %[[REAL]], %[[ZERO]] : f32
// CHECK: %[[IMAG_IS_ZERO:.*]] = llvm.fcmp "oeq" %[[IMAG]], %[[ZERO]] : f32

// CHECK: %[[IMAG_DIV_REAL:.*]] = llvm.fdiv %[[IMAG]], %[[REAL]] : f32
// CHECK: %[[IMAG_SQ:.*]] = llvm.fmul %[[IMAG_DIV_REAL]], %[[IMAG_DIV_REAL]]  : f32
// CHECK: %[[IMAG_SQ_PLUS_ONE:.*]] = llvm.fadd %[[IMAG_SQ]], %[[ONE]] : f32
// CHECK: %[[IMAG_SQRT:.*]] = llvm.intr.sqrt(%[[IMAG_SQ_PLUS_ONE]]) : (f32) -> f32
// CHECK: %[[REAL_ABS:.*]] = llvm.intr.fabs(%[[REAL]]) : (f32) -> f32
// CHECK: %[[ABS_IMAG:.*]] = llvm.fmul %[[IMAG_SQRT]], %[[REAL_ABS]] : f32

// CHECK: %[[REAL_DIV_IMAG:.*]] = llvm.fdiv %[[REAL]], %[[IMAG]] : f32
// CHECK: %[[REAL_SQ:.*]] = llvm.fmul %[[REAL_DIV_IMAG]], %[[REAL_DIV_IMAG]] : f32
// CHECK: %[[REAL_SQ_PLUS_ONE:.*]] = llvm.fadd %[[REAL_SQ]], %[[ONE]]  : f32
// CHECK: %[[REAL_SQRT:.*]] = llvm.intr.sqrt(%[[REAL_SQ_PLUS_ONE]])  : (f32) -> f32
// CHECK: %[[IMAG_ABS:.*]] = llvm.intr.fabs(%[[IMAG]]) : (f32) -> f32
// CHECK: %[[ABS_REAL:.*]] = llvm.fmul %[[REAL_SQRT]], %[[IMAG_ABS]]  : f32

// CHECK: %[[REAL_GT_IMAG:.*]] = llvm.fcmp "ogt" %[[REAL]], %[[IMAG]] : f32
// CHECK: %[[ABS1:.*]] = llvm.select %[[REAL_GT_IMAG]], %[[ABS_IMAG]], %[[ABS_REAL]] : i1, f32
// CHECK: %[[ABS2:.*]] = llvm.select %[[IMAG_IS_ZERO]], %[[REAL_ABS]], %[[ABS1]] : i1, f32
// CHECK: %[[NORM:.*]] = llvm.select %[[REAL_IS_ZERO]], %[[IMAG_ABS]], %[[ABS2]] : i1, f32
// CHECK: llvm.return %[[NORM]] : f32

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
