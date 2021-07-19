// RUN: mlir-opt %s -convert-complex-to-standard -convert-complex-to-llvm -convert-math-to-llvm -convert-std-to-llvm | FileCheck %s

// CHECK-LABEL: llvm.func @complex_abs
// CHECK-SAME: %[[ARG:.*]]: ![[C_TY:.*]])
func @complex_abs(%arg: complex<f32>) -> f32 {
  %abs = complex.abs %arg: complex<f32>
  return %abs : f32
}
// CHECK: %[[REAL:.*]] = llvm.extractvalue %[[ARG]][0] : ![[C_TY]]
// CHECK: %[[IMAG:.*]] = llvm.extractvalue %[[ARG]][1] : ![[C_TY]]
// CHECK-DAG: %[[REAL_SQ:.*]] = llvm.fmul %[[REAL]], %[[REAL]]  : f32
// CHECK-DAG: %[[IMAG_SQ:.*]] = llvm.fmul %[[IMAG]], %[[IMAG]]  : f32
// CHECK: %[[SQ_NORM:.*]] = llvm.fadd %[[REAL_SQ]], %[[IMAG_SQ]]  : f32
// CHECK: %[[NORM:.*]] = "llvm.intr.sqrt"(%[[SQ_NORM]]) : (f32) -> f32
// CHECK: llvm.return %[[NORM]] : f32

// CHECK-LABEL: llvm.func @complex_eq
// CHECK-SAME: %[[LHS:.*]]: ![[C_TY:.*]], %[[RHS:.*]]: ![[C_TY:.*]])
func @complex_eq(%lhs: complex<f32>, %rhs: complex<f32>) -> i1 {
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
func @complex_neq(%lhs: complex<f32>, %rhs: complex<f32>) -> i1 {
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
