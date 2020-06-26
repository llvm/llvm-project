// RUN: mlir-opt %s -pass-pipeline='func(canonicalize)' | FileCheck %s

func @single_iteration(%A: memref<?x?x?xi32>) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %c3 = constant 3 : index
  %c6 = constant 6 : index
  %c7 = constant 7 : index
  %c10 = constant 10 : index
  scf.parallel (%i0, %i1, %i2) = (%c0, %c3, %c7) to (%c1, %c6, %c10) step (%c1, %c2, %c3) {
    %c42 = constant 42 : i32
    store %c42, %A[%i0, %i1, %i2] : memref<?x?x?xi32>
    scf.yield
  }
  return
}

// CHECK-LABEL:   func @single_iteration(
// CHECK-SAME:                        [[ARG0:%.*]]: memref<?x?x?xi32>) {
// CHECK:           [[C0:%.*]] = constant 0 : index
// CHECK:           [[C2:%.*]] = constant 2 : index
// CHECK:           [[C3:%.*]] = constant 3 : index
// CHECK:           [[C6:%.*]] = constant 6 : index
// CHECK:           [[C7:%.*]] = constant 7 : index
// CHECK:           [[C42:%.*]] = constant 42 : i32
// CHECK:           scf.parallel ([[V0:%.*]]) = ([[C3]]) to ([[C6]]) step ([[C2]]) {
// CHECK:             store [[C42]], [[ARG0]]{{\[}}[[C0]], [[V0]], [[C7]]] : memref<?x?x?xi32>
// CHECK:             scf.yield
// CHECK:           }
// CHECK:           return

// -----

func @no_iteration(%A: memref<?x?xi32>) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  scf.parallel (%i0, %i1) = (%c0, %c0) to (%c1, %c0) step (%c1, %c1) {
    %c42 = constant 42 : i32
    store %c42, %A[%i0, %i1] : memref<?x?xi32>
    scf.yield
  }
  return
}

// CHECK-LABEL:   func @no_iteration(
// CHECK-SAME:                        [[ARG0:%.*]]: memref<?x?xi32>) {
// CHECK:           [[C0:%.*]] = constant 0 : index
// CHECK:           [[C1:%.*]] = constant 1 : index
// CHECK:           [[C42:%.*]] = constant 42 : i32
// CHECK:           scf.parallel ([[V1:%.*]]) = ([[C0]]) to ([[C0]]) step ([[C1]]) {
// CHECK:             store [[C42]], [[ARG0]]{{\[}}[[C0]], [[V1]]] : memref<?x?xi32>
// CHECK:             scf.yield
// CHECK:           }
// CHECK:           return
