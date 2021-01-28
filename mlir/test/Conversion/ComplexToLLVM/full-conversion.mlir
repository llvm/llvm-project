// RUN: mlir-opt %s -convert-complex-to-llvm -convert-std-to-llvm | FileCheck %s

// CHECK-LABEL: llvm.func @complex_div
// CHECK-SAME:    %[[LHS:.*]]: ![[C_TY:.*>]], %[[RHS:.*]]: ![[C_TY]]) -> ![[C_TY]]
func @complex_div(%lhs: complex<f32>, %rhs: complex<f32>) -> complex<f32> {
  %div = complex.div %lhs, %rhs : complex<f32>
  return %div : complex<f32>
}
// CHECK: %[[LHS_RE:.*]] = llvm.extractvalue %[[LHS]][0] : ![[C_TY]]
// CHECK: %[[LHS_IM:.*]] = llvm.extractvalue %[[LHS]][1] : ![[C_TY]]
// CHECK: %[[RHS_RE:.*]] = llvm.extractvalue %[[RHS]][0] : ![[C_TY]]
// CHECK: %[[RHS_IM:.*]] = llvm.extractvalue %[[RHS]][1] : ![[C_TY]]

// CHECK: %[[RESULT_0:.*]] = llvm.mlir.undef : ![[C_TY]]

// CHECK-DAG: %[[RHS_RE_SQ:.*]] = llvm.fmul %[[RHS_RE]], %[[RHS_RE]]  : f32
// CHECK-DAG: %[[RHS_IM_SQ:.*]] = llvm.fmul %[[RHS_IM]], %[[RHS_IM]]  : f32
// CHECK: %[[SQ_NORM:.*]] = llvm.fadd %[[RHS_RE_SQ]], %[[RHS_IM_SQ]]  : f32

// CHECK-DAG: %[[REAL_TMP_0:.*]] = llvm.fmul %[[LHS_RE]], %[[RHS_RE]]  : f32
// CHECK-DAG: %[[REAL_TMP_1:.*]] = llvm.fmul %[[LHS_IM]], %[[RHS_IM]]  : f32
// CHECK: %[[REAL_TMP_2:.*]] = llvm.fadd %[[REAL_TMP_0]], %[[REAL_TMP_1]]  : f32

// CHECK-DAG: %[[IMAG_TMP_0:.*]] = llvm.fmul %[[LHS_IM]], %[[RHS_RE]]  : f32
// CHECK-DAG: %[[IMAG_TMP_1:.*]] = llvm.fmul %[[LHS_RE]], %[[RHS_IM]]  : f32
// CHECK: %[[IMAG_TMP_2:.*]] = llvm.fsub %[[IMAG_TMP_0]], %[[IMAG_TMP_1]]  : f32

// CHECK: %[[REAL:.*]] = llvm.fdiv %[[REAL_TMP_2]], %[[SQ_NORM]]  : f32
// CHECK: %[[RESULT_1:.*]] = llvm.insertvalue %[[REAL]], %[[RESULT_0]][0] : ![[C_TY]]
// CHECK: %[[IMAG:.*]] = llvm.fdiv %[[IMAG_TMP_2]], %[[SQ_NORM]]  : f32
// CHECK: %[[RESULT_2:.*]] = llvm.insertvalue %[[IMAG]], %[[RESULT_1]][1] : ![[C_TY]]
// CHECK: llvm.return %[[RESULT_2]] : ![[C_TY]]

// CHECK-LABEL: llvm.func @complex_mul
// CHECK-SAME:    %[[LHS:.*]]: ![[C_TY:.*>]], %[[RHS:.*]]: ![[C_TY]]) -> ![[C_TY]]
func @complex_mul(%lhs: complex<f32>, %rhs: complex<f32>) -> complex<f32> {
  %mul = complex.mul %lhs, %rhs : complex<f32>
  return %mul : complex<f32>
}
// CHECK: %[[LHS_RE:.*]] = llvm.extractvalue %[[LHS]][0] : ![[C_TY]]
// CHECK: %[[LHS_IM:.*]] = llvm.extractvalue %[[LHS]][1] : ![[C_TY]]
// CHECK: %[[RHS_RE:.*]] = llvm.extractvalue %[[RHS]][0] : ![[C_TY]]
// CHECK: %[[RHS_IM:.*]] = llvm.extractvalue %[[RHS]][1] : ![[C_TY]]
// CHECK: %[[RESULT_0:.*]] = llvm.mlir.undef : ![[C_TY]]

// CHECK-DAG: %[[REAL_TMP_0:.*]] = llvm.fmul %[[RHS_RE]], %[[LHS_RE]]  : f32
// CHECK-DAG: %[[REAL_TMP_1:.*]] = llvm.fmul %[[RHS_IM]], %[[LHS_IM]]  : f32
// CHECK: %[[REAL:.*]] = llvm.fsub %[[REAL_TMP_0]], %[[REAL_TMP_1]]  : f32

// CHECK-DAG: %[[IMAG_TMP_0:.*]] = llvm.fmul %[[LHS_IM]], %[[RHS_RE]]  : f32
// CHECK-DAG: %[[IMAG_TMP_1:.*]] = llvm.fmul %[[LHS_RE]], %[[RHS_IM]]  : f32
// CHECK: %[[IMAG:.*]] = llvm.fadd %[[IMAG_TMP_0]], %[[IMAG_TMP_1]]  : f32

// CHECK: %[[RESULT_1:.*]] = llvm.insertvalue %[[REAL]], %[[RESULT_0]][0]
// CHECK: %[[RESULT_2:.*]] = llvm.insertvalue %[[IMAG]], %[[RESULT_1]][1]
// CHECK: llvm.return %[[RESULT_2]] : ![[C_TY]]

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

