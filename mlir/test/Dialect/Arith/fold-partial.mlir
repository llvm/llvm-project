// RUN: mlir-opt %s -test-single-fold -split-input-file | FileCheck %s --check-prefixes=CHECK,FOLD
// RUN: mlir-opt %s -sccp -split-input-file | FileCheck %s --check-prefixes=CHECK,SCCP
// RUN: mlir-opt %s -canonicalize -split-input-file | FileCheck %s --check-prefix=CANON

// The -test-single-fold and -sccp runs check the mulsi_extended fold alone.
// -canonicalize also runs MulSIExtendedRHSOne, which would hide a broken fold,
// so the CANON run checks only the constant case.

// The fold replaces the low half of mulsi_extended(x, 1) with x and keeps the
// high half. SCCP joins the low half with the lattice of x. That lattice is not
// a constant, so SCCP replaces neither half.

// CHECK-LABEL: func @mulsi_extended_one_rhs(
//  CHECK-SAME:   %[[X:.+]]: i32) -> (i32, i32)
//   SCCP-NOT:    arith.constant 0
//       CHECK:   %[[LOW:.+]], %[[HIGH:.+]] = arith.mulsi_extended %[[X]], %{{.+}} : i32
//        FOLD:   return %[[X]], %[[HIGH]] : i32, i32
//        SCCP:   return %[[LOW]], %[[HIGH]] : i32, i32
func.func @mulsi_extended_one_rhs(%x: i32) -> (i32, i32) {
  %c1 = arith.constant 1 : i32
  %low, %high = arith.mulsi_extended %x, %c1 : i32
  return %low, %high : i32, i32
}

// -----

// CHECK-LABEL: func @mulsi_extended_one_rhs_splat(
//  CHECK-SAME:   %[[X:.+]]: vector<3xi32>) -> (vector<3xi32>, vector<3xi32>)
//   SCCP-NOT:    arith.constant dense<0>
//       CHECK:   %[[LOW:.+]], %[[HIGH:.+]] = arith.mulsi_extended %[[X]], %{{.+}} : vector<3xi32>
//        FOLD:   return %[[X]], %[[HIGH]] : vector<3xi32>, vector<3xi32>
//        SCCP:   return %[[LOW]], %[[HIGH]] : vector<3xi32>, vector<3xi32>
func.func @mulsi_extended_one_rhs_splat(%x: vector<3xi32>) -> (vector<3xi32>, vector<3xi32>) {
  %one = arith.constant dense<1> : vector<3xi32>
  %low, %high = arith.mulsi_extended %x, %one : vector<3xi32>
  return %low, %high : vector<3xi32>, vector<3xi32>
}

// -----

// For i1, true is -1 as a signed value. The low half of x * -1 is still x.

// CHECK-LABEL: func @mulsi_extended_true_rhs_i1(
//  CHECK-SAME:   %[[X:.+]]: i1) -> (i1, i1)
//       CHECK:   %[[LOW:.+]], %[[HIGH:.+]] = arith.mulsi_extended %[[X]], %{{.+}} : i1
//        FOLD:   return %[[X]], %[[HIGH]] : i1, i1
//        SCCP:   return %[[LOW]], %[[HIGH]] : i1, i1
func.func @mulsi_extended_true_rhs_i1(%x: i1) -> (i1, i1) {
  %true = arith.constant true
  %low, %high = arith.mulsi_extended %x, %true : i1
  return %low, %high : i1, i1
}

// -----

// The constant fold comes before the x * 1 fold, so both halves fold.

// CHECK-LABEL: func @mulsi_extended_const_one_rhs(
//   CHECK-DAG:   %[[C5:.+]] = arith.constant 5 : i32
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : i32
//   CHECK-NOT:   arith.mulsi_extended
//       CHECK:   return %[[C5]], %[[C0]] : i32, i32
// CANON-LABEL: func @mulsi_extended_const_one_rhs(
//   CANON-DAG:   %[[C5:.+]] = arith.constant 5 : i32
//   CANON-DAG:   %[[C0:.+]] = arith.constant 0 : i32
//   CANON-NOT:   arith.mulsi_extended
//       CANON:   return %[[C5]], %[[C0]] : i32, i32
func.func @mulsi_extended_const_one_rhs() -> (i32, i32) {
  %c5 = arith.constant 5 : i32
  %c1 = arith.constant 1 : i32
  %low, %high = arith.mulsi_extended %c5, %c1 : i32
  return %low, %high : i32, i32
}
