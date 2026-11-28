// RUN: mlir-opt -allow-unregistered-dialect -pass-pipeline='builtin.module(func.func(test-scf-parallel-loop-collapsing{collapsed-indices-0=0,1}, canonicalize))' --mlir-print-local-scope %s | FileCheck %s

func.func @collapse_to_single() {
  %c0 = arith.constant 3 : index
  %c1 = arith.constant 7 : index
  %c2 = arith.constant 11 : index
  %c3 = arith.constant 29 : index
  %c4 = arith.constant 3 : index
  %c5 = arith.constant 4 : index
  scf.parallel (%i0, %i1) = (%c0, %c1) to (%c2, %c3) step (%c4, %c5) {
    %result = "magic.op"(%i0, %i1): (index, index) -> index
  }
  return
}

// CHECK: func @collapse_to_single() {
// CHECK-DAG:         %[[C6:.*]] = arith.constant 6 : index
// CHECK-DAG:         %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:         %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:         %[[C18:.*]] = arith.constant 18 : index
// CHECK:         scf.parallel (%[[NEW_I:.*]]) = (%[[C0]]) to (%[[C18]]) step (%[[C1]]) {
// CHECK:           %[[I0_COUNT:.*]] = arith.remsi %[[NEW_I]], %[[C6]] : index
// CHECK:           %[[I1_COUNT:.*]] = arith.divsi %[[NEW_I]], %[[C6]] : index
// CHECK:           %[[I1:.*]] = affine.apply affine_map<(d0) -> (d0 * 4 + 7)>(%[[I0_COUNT]])
// CHECK:           %[[I0:.*]] = affine.apply affine_map<(d0) -> (d0 * 3 + 3)>(%[[I1_COUNT]])
// CHECK:           "magic.op"(%[[I0]], %[[I1]]) : (index, index) -> index
// CHECK:           scf.reduce
// CHECK-NEXT:    }
// CHECK-NEXT:    return

// Collapsing keeps the reductions and rewires their results.

// CHECK-LABEL: func @collapse_reduction(
// CHECK:         %[[SUM:.*]] = scf.parallel (%{{.*}}) = {{.*}} init ({{.*}}) -> f32 {
// CHECK:           scf.reduce(%{{.*}} : f32) {
// CHECK:             arith.addf
// CHECK:         memref.store %[[SUM]]
func.func @collapse_reduction(%arg0: memref<10x10xf32>, %arg1: memref<f32>) {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %0 = scf.parallel (%i, %j) = (%c0, %c0) to (%c10, %c10) step (%c1, %c1) init (%cst) -> f32 {
    %1 = memref.load %arg0[%i, %j] : memref<10x10xf32>
    scf.reduce(%1 : f32) {
    ^bb0(%a: f32, %b: f32):
      %2 = arith.addf %a, %b : f32
      scf.reduce.return %2 : f32
    }
  }
  memref.store %0, %arg1[] : memref<f32>
  return
}

// Every reduction keeps its init, its region, and its result.

// CHECK-LABEL: func @collapse_reductions(
// CHECK:         %[[RES:.*]]:2 = scf.parallel (%{{.*}}) = {{.*}} init ({{.*}}, {{.*}}) -> (i32, i32) {
// CHECK:           scf.reduce(%{{.*}}, %{{.*}} : i32, i32) {
// CHECK:             arith.addi
// CHECK:           }, {
// CHECK:             arith.muli
// CHECK:         return %[[RES]]#0, %[[RES]]#1
func.func @collapse_reductions(%m: memref<4x6xi32>) -> (i32, i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c0_idx = arith.constant 0 : index
  %c1_idx = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c6 = arith.constant 6 : index
  %sum, %prod = scf.parallel (%i, %j) = (%c0_idx, %c0_idx) to (%c4, %c6) step (%c1_idx, %c1_idx) init (%c0, %c1) -> (i32, i32) {
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

// An original induction variable used inside a reduction region is recovered
// from the collapsed one.

// CHECK-LABEL: func @collapse_reduction_uses_iv(
// CHECK:         scf.parallel (%[[IV:.*]]) = {{.*}} -> index {
// CHECK:           %[[J:.*]] = arith.remsi %[[IV]], {{.*}} : index
// CHECK:           scf.reduce(
// CHECK:             arith.addi {{.*}}, %[[J]] : index
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
