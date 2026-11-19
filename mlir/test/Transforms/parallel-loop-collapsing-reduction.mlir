// RUN: mlir-opt -pass-pipeline='builtin.module(func.func(test-scf-parallel-loop-collapsing{collapsed-indices-0=0,1}, canonicalize))' --mlir-print-local-scope %s | FileCheck %s

// Collapsing a parallel loop keeps its reductions and rewires their results.

// CHECK-LABEL: func @collapse_reduction(%
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C10:.*]] = arith.constant 10 : index
// CHECK-DAG: %[[C100:.*]] = arith.constant 100 : index
// CHECK: %[[SUM:.*]] = scf.parallel (%[[IV:.*]]) = (%[[C0]]) to (%[[C100]]) step (%[[C1]]) init (%[[ZERO]]) -> f32 {
// CHECK: %[[J:.*]] = arith.remsi %[[IV]], %[[C10]] : index
// CHECK: %[[I:.*]] = arith.divsi %[[IV]], %[[C10]] : index
// CHECK: %[[VAL:.*]] = memref.load {{%.*}}[%[[I]], %[[J]]] : memref<10x10xf32>
// CHECK: scf.reduce(%[[VAL]] : f32) {
// CHECK: arith.addf
// CHECK: memref.store %[[SUM]], {{%.*}}[] : memref<f32>

func.func @collapse_reduction(%arg0: memref<10x10xf32>, %arg1: memref<f32>) {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %0 = scf.parallel (%i, %j) = (%c0, %c0) to (%c10, %c10) step (%c1, %c1) init (%cst) -> f32 {
    %1 = memref.load %arg0[%i, %j] : memref<10x10xf32>
    scf.reduce(%1 : f32) {
    ^bb0(%arg2: f32, %arg3: f32):
      %2 = arith.addf %arg2, %arg3 : f32
      scf.reduce.return %2 : f32
    }
  }
  memref.store %0, %arg1[] : memref<f32>
  return
}

// Each reduction keeps its init, its region, and its result.

// CHECK-LABEL: func @collapse_reductions(%
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : i32
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : i32
// CHECK-DAG: %[[C0_IDX:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1_IDX:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C6:.*]] = arith.constant 6 : index
// CHECK-DAG: %[[C24:.*]] = arith.constant 24 : index
// CHECK: %[[RES:.*]]:2 = scf.parallel (%[[IV:.*]]) = (%[[C0_IDX]]) to (%[[C24]]) step (%[[C1_IDX]]) init (%[[C0]], %[[C1]]) -> (i32, i32) {
// CHECK: %[[J:.*]] = arith.remsi %[[IV]], %[[C6]] : index
// CHECK: %[[I:.*]] = arith.divsi %[[IV]], %[[C6]] : index
// CHECK: %[[VAL:.*]] = memref.load {{%.*}}[%[[I]], %[[J]]] : memref<4x6xi32>
// CHECK: scf.reduce(%[[VAL]], %[[VAL]] : i32, i32) {
// CHECK: arith.addi
// CHECK: }, {
// CHECK: arith.muli
// CHECK: return %[[RES]]#0, %[[RES]]#1 : i32, i32

func.func @collapse_reductions(%m: memref<4x6xi32>) -> (i32, i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c0_idx = arith.constant 0 : index
  %c1_idx = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c6 = arith.constant 6 : index
  %sum, %prod = scf.parallel (%i, %j) = (%c0_idx, %c0_idx) to (%c4, %c6)
      step (%c1_idx, %c1_idx) init (%c0, %c1) -> (i32, i32) {
    %v = memref.load %m[%i, %j] : memref<4x6xi32>
    scf.reduce(%v, %v : i32, i32) {
    ^bb0(%a: i32, %b: i32):
      %s = arith.addi %a, %b : i32
      scf.reduce.return %s : i32
    }, {
    ^bb0(%a: i32, %b: i32):
      %p = arith.muli %a, %b : i32
      scf.reduce.return %p : i32
    }
  }
  return %sum, %prod : i32, i32
}

// A value from the original induction variables, including one used inside a
// reduction region, is recovered from the collapsed induction variable.

// CHECK-LABEL: func @collapse_reduction_uses_iv()
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C8:.*]] = arith.constant 8 : index
// CHECK-DAG: %[[C32:.*]] = arith.constant 32 : index
// CHECK: %[[R:.*]] = scf.parallel (%[[IV:.*]]) = (%[[C0]]) to (%[[C32]]) step (%[[C1]]) init (%[[C0]]) -> index {
// CHECK: %[[J:.*]] = arith.remsi %[[IV]], %[[C8]] : index
// CHECK: %[[I:.*]] = arith.divsi %[[IV]], %[[C8]] : index
// CHECK: scf.reduce(%[[I]] : index) {
// CHECK: arith.addi {{%.*}}, %[[J]] : index
// CHECK: return %[[R]] : index

func.func @collapse_reduction_uses_iv() -> index {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %r = scf.parallel (%i, %j) = (%c0, %c0) to (%c4, %c8) step (%c1, %c1) init (%c0) -> index {
    scf.reduce(%i : index) {
    ^bb0(%a: index, %b: index):
      %s = arith.addi %a, %j : index
      scf.reduce.return %s : index
    }
  }
  return %r : index
}

// Non-zero lower bounds and non-unit steps are denormalized before the value
// is reduced.

// CHECK-LABEL: func @collapse_reduction_normalized(
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C6:.*]] = arith.constant 6 : index
// CHECK-DAG: %[[C18:.*]] = arith.constant 18 : index
// CHECK: %[[R:.*]] = scf.parallel (%[[IV:.*]]) = (%[[C0]]) to (%[[C18]]) step (%[[C1]]) init (%[[INIT:arg[0-9]+]]) -> index {
// CHECK: %[[N1:.*]] = arith.remsi %[[IV]], %[[C6]] : index
// CHECK: %[[N0:.*]] = arith.divsi %[[IV]], %[[C6]] : index
// CHECK: %[[I1:.*]] = affine.apply affine_map<(d0) -> (d0 * 4 + 7)>(%[[N1]])
// CHECK: %[[I0:.*]] = affine.apply affine_map<(d0) -> (d0 * 3 + 3)>(%[[N0]])
// CHECK: %[[SUM:.*]] = arith.addi %[[I0]], %[[I1]] : index
// CHECK: scf.reduce(%[[SUM]] : index) {
// CHECK: return %[[R]] : index

func.func @collapse_reduction_normalized(%init: index) -> index {
  %lb0 = arith.constant 3 : index
  %lb1 = arith.constant 7 : index
  %ub0 = arith.constant 11 : index
  %ub1 = arith.constant 29 : index
  %step0 = arith.constant 3 : index
  %step1 = arith.constant 4 : index
  %r = scf.parallel (%i0, %i1) = (%lb0, %lb1) to (%ub0, %ub1) step (%step0, %step1) init (%init) -> index {
    %sum = arith.addi %i0, %i1 : index
    scf.reduce(%sum : index) {
    ^bb0(%a: index, %b: index):
      %s = arith.addi %a, %b : index
      scf.reduce.return %s : index
    }
  }
  return %r : index
}
