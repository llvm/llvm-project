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

// -----

func @one_unused() -> (index) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %true = constant true
  %0, %1 = scf.if %true -> (index, index) {
    scf.yield %c0, %c1 : index, index
  } else {
    scf.yield %c0, %c1 : index, index
  }
  return %1 : index
}

// CHECK-LABEL:   func @one_unused
// CHECK:           [[C0:%.*]] = constant 1 : index
// CHECK:           [[C1:%.*]] = constant true
// CHECK:           [[V0:%.*]] = scf.if [[C1]] -> (index) {
// CHECK:             scf.yield [[C0]] : index
// CHECK:           } else
// CHECK:             scf.yield [[C0]] : index
// CHECK:           }
// CHECK:           return [[V0]] : index

// -----

func @nested_unused() -> (index) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %true = constant true
  %0, %1 = scf.if %true -> (index, index) {
    %2, %3 = scf.if %true -> (index, index) {
      scf.yield %c0, %c1 : index, index
    } else {
      scf.yield %c0, %c1 : index, index
    }
    scf.yield %2, %3 : index, index
  } else {
    scf.yield %c0, %c1 : index, index
  }
  return %1 : index
}

// CHECK-LABEL:   func @nested_unused
// CHECK:           [[C0:%.*]] = constant 1 : index
// CHECK:           [[C1:%.*]] = constant true
// CHECK:           [[V0:%.*]] = scf.if [[C1]] -> (index) {
// CHECK:             [[V1:%.*]] = scf.if [[C1]] -> (index) {
// CHECK:               scf.yield [[C0]] : index
// CHECK:             } else
// CHECK:               scf.yield [[C0]] : index
// CHECK:             }
// CHECK:             scf.yield [[V1]] : index
// CHECK:           } else
// CHECK:             scf.yield [[C0]] : index
// CHECK:           }
// CHECK:           return [[V0]] : index

// -----

func private @side_effect() {}
func @all_unused() {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %true = constant true
  %0, %1 = scf.if %true -> (index, index) {
    call @side_effect() : () -> ()
    scf.yield %c0, %c1 : index, index
  } else {
    call @side_effect() : () -> ()
    scf.yield %c0, %c1 : index, index
  }
  return
}

// CHECK-LABEL:   func @all_unused
// CHECK:           [[C1:%.*]] = constant true
// CHECK:           scf.if [[C1]] {
// CHECK:             call @side_effect() : () -> ()
// CHECK:           } else
// CHECK:             call @side_effect() : () -> ()
// CHECK:           }
// CHECK:           return

// -----

func private @make_i32() -> i32

func @for_yields_2(%lb : index, %ub : index, %step : index) -> i32 {
  %a = call @make_i32() : () -> (i32)
  %b = scf.for %i = %lb to %ub step %step iter_args(%0 = %a) -> i32 {
    scf.yield %0 : i32
  }
  return %b : i32
}

// CHECK-LABEL:   func @for_yields_2
//  CHECK-NEXT:     %[[R:.*]] = call @make_i32() : () -> i32
//  CHECK-NEXT:     return %[[R]] : i32

func @for_yields_3(%lb : index, %ub : index, %step : index) -> (i32, i32, i32) {
  %a = call @make_i32() : () -> (i32)
  %b = call @make_i32() : () -> (i32)
  %r:3 = scf.for %i = %lb to %ub step %step iter_args(%0 = %a, %1 = %a, %2 = %b) -> (i32, i32, i32) {
    %c = call @make_i32() : () -> (i32)
    scf.yield %0, %c, %2 : i32, i32, i32
  }
  return %r#0, %r#1, %r#2 : i32, i32, i32
}

// CHECK-LABEL:   func @for_yields_3
//  CHECK-NEXT:     %[[a:.*]] = call @make_i32() : () -> i32
//  CHECK-NEXT:     %[[b:.*]] = call @make_i32() : () -> i32
//  CHECK-NEXT:     %[[r1:.*]] = scf.for {{.*}} iter_args(%arg4 = %[[a]]) -> (i32) {
//  CHECK-NEXT:       %[[c:.*]] = call @make_i32() : () -> i32
//  CHECK-NEXT:       scf.yield %[[c]] : i32
//  CHECK-NEXT:     }
//  CHECK-NEXT:     return %[[a]], %[[r1]], %[[b]] : i32, i32, i32
