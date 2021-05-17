// RUN: mlir-opt %s -convert-complex-to-standard -convert-complex-to-llvm -convert-std-to-llvm | FileCheck %s

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

