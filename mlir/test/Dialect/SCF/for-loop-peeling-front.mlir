// RUN: mlir-opt %s -scf-for-loop-peeling=peel-front=true -split-input-file | FileCheck %s

//  CHECK-DAG: #[[MAP:.*]] = affine_map<(d0, d1)[s0] -> (4, d0 - d1)>
//      CHECK: func @fully_static_bounds(
//  CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : index
//  CHECK-DAG:   %[[C0_I32:.*]] = arith.constant 0 : i32
//  CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C17:.*]] = arith.constant 17 : index
//      CHECK:   %[[FIRST:.*]] = scf.for %[[IV:.*]] = %[[C0]] to %[[C4]]
// CHECK-SAME:       step %[[C4]] iter_args(%[[ACC:.*]] = %[[C0_I32]]) -> (i32) {
//      CHECK:     %[[MIN:.*]] = affine.min #[[MAP]](%[[C4]], %[[IV]])[%[[C4]]]
//      CHECK:     %[[CAST:.*]] = arith.index_cast %[[MIN]] : index to i32
//      CHECK:     %[[INIT:.*]] = arith.addi %[[ACC]], %[[CAST]] : i32
//      CHECK:     scf.yield %[[INIT]]
//      CHECK:   }
//      CHECK:   %[[RESULT:.*]] = scf.for %[[IV:.*]] = %[[C4]] to %[[C17]]
// CHECK-SAME:       step %[[C4]] iter_args(%[[ACC:.*]] = %[[FIRST]]) -> (i32) {
//      CHECK:     %[[MIN2:.*]] = affine.min #[[MAP]](%[[C17]], %[[IV]])[%[[C4]]]
//      CHECK:     %[[CAST2:.*]] = arith.index_cast %[[MIN2]] : index to i32
//      CHECK:     %[[ADD:.*]] = arith.addi %[[ACC]], %[[CAST2]] : i32
//      CHECK:     scf.yield %[[ADD]]
//      CHECK:   }
//      CHECK:   return %[[RESULT]]
#map = affine_map<(d0, d1)[s0] -> (s0, d0 - d1)>
func.func @fully_static_bounds() -> i32 {
  %c0_i32 = arith.constant 0 : i32
  %lb = arith.constant 0 : index
  %step = arith.constant 4 : index
  %ub = arith.constant 17 : index
  %r = scf.for %iv = %lb to %ub step %step iter_args(%arg = %c0_i32) -> i32 {
    %s = affine.min #map(%ub, %iv)[%step]
    %casted = arith.index_cast %s : index to i32
    %0 = arith.addi %arg, %casted : i32
    scf.yield %0 : i32
  }
  return %r : i32
}

// -----

//  CHECK-DAG: #[[MAP:.*]] = affine_map<(d0, d1)[s0] -> (4, d0 - d1)>
//      CHECK: func @no_loop_results(
// CHECK-SAME:     %[[UB:.*]]: index, %[[MEMREF:.*]]: memref<i32>
//  CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : index
//  CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//      CHECK:   scf.for %[[IV:.*]] = %[[C0]] to %[[C4]] step %[[C4]] {
//      CHECK:     %[[MIN:.*]] = affine.min #[[MAP]](%[[C4]], %[[IV]])[%[[C4]]]
//      CHECK:     %[[LOAD:.*]] = memref.load %[[MEMREF]][]
//      CHECK:     %[[CAST:.*]] = arith.index_cast %[[MIN]]
//      CHECK:     %[[ADD:.*]] = arith.addi %[[LOAD]], %[[CAST]] : i32
//      CHECK:     memref.store %[[ADD]], %[[MEMREF]]
//      CHECK:   }
//      CHECK:   scf.for %[[IV2:.*]] = %[[C4]] to %[[UB]] step %[[C4]] {
//      CHECK:     %[[REM:.*]] = affine.min #[[MAP]](%[[UB]], %[[IV2]])[%[[C4]]]
//      CHECK:     %[[LOAD2:.*]] = memref.load %[[MEMREF]][]
//      CHECK:     %[[CAST2:.*]] = arith.index_cast %[[REM]]
//      CHECK:     %[[ADD2:.*]] = arith.addi %[[LOAD2]], %[[CAST2]]
//      CHECK:     memref.store %[[ADD2]], %[[MEMREF]]
//      CHECK:   }
//      CHECK:   return
#map = affine_map<(d0, d1)[s0] -> (s0, d0 - d1)>
func.func @no_loop_results(%ub : index, %d : memref<i32>) {
  %c0_i32 = arith.constant 0 : i32
  %lb = arith.constant 0 : index
  %step = arith.constant 4 : index
  scf.for %iv = %lb to %ub step %step {
    %s = affine.min #map(%ub, %iv)[%step]
    %r = memref.load %d[] : memref<i32>
    %casted = arith.index_cast %s : index to i32
    %0 = arith.addi %r, %casted : i32
    memref.store %0, %d[] : memref<i32>
  }
  return
}

// -----

//  CHECK-DAG: #[[MAP0:.*]] = affine_map<()[s0, s1] -> (s0 + s1)>
//  CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0, d1)[s0] -> (s0, d0 - d1)>
//      CHECK: func @fully_dynamic_bounds(
// CHECK-SAME:     %[[LB:.*]]: index, %[[UB:.*]]: index, %[[STEP:.*]]: index
//      CHECK:   %[[C0_I32:.*]] = arith.constant 0 : i32
//      CHECK:   %[[NEW_UB:.*]] = affine.apply #[[MAP0]]()[%[[LB]], %[[STEP]]]
//      CHECK:   %[[FIRST:.*]] = scf.for %[[IV:.*]] = %[[LB]] to %[[NEW_UB]]
// CHECK-SAME:       step %[[STEP]] iter_args(%[[ACC:.*]] = %[[C0_I32]]) -> (i32) {
//      CHECK:     %[[MIN:.*]] = affine.min #[[MAP1]](%[[NEW_UB]], %[[IV]])[%[[STEP]]]
//      CHECK:     %[[CAST:.*]] = arith.index_cast %[[MIN]] : index to i32
//      CHECK:     %[[ADD:.*]] = arith.addi %[[ACC]], %[[CAST]] : i32
//      CHECK:     scf.yield %[[ADD]]
//      CHECK:   }
//      CHECK:   %[[RESULT:.*]] = scf.for %[[IV2:.*]] = %[[NEW_UB]] to %[[UB]]
// CHECK-SAME:       step %[[STEP]] iter_args(%[[ACC2:.*]] = %[[FIRST]]) -> (i32) {
//      CHECK:     %[[REM:.*]] = affine.min #[[MAP1]](%[[UB]], %[[IV2]])[%[[STEP]]]
//      CHECK:     %[[CAST2:.*]] = arith.index_cast %[[REM]]
//      CHECK:     %[[ADD2:.*]] = arith.addi %[[ACC2]], %[[CAST2]]
//      CHECK:     scf.yield %[[ADD2]]
//      CHECK:   }
//      CHECK:   return %[[RESULT]]
#map = affine_map<(d0, d1)[s0] -> (s0, d0 - d1)>
func.func @fully_dynamic_bounds(%lb : index, %ub: index, %step: index) -> i32 {
  %c0 = arith.constant 0 : i32
  %r = scf.for %iv = %lb to %ub step %step iter_args(%arg = %c0) -> i32 {
    %s = affine.min #map(%ub, %iv)[%step]
    %casted = arith.index_cast %s : index to i32
    %0 = arith.addi %arg, %casted : i32
    scf.yield %0 : i32
  }
  return %r : i32
}

// -----

//  CHECK-DAG: #[[MAP:.*]] = affine_map<(d0, d1)[s0] -> (4, d0 - d1)>
//      CHECK: func @no_peeling_front(
//  CHECK-DAG:   %[[C0_I32:.*]] = arith.constant 0 : i32
//  CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : index
//      CHECK:   %[[RESULT:.*]] = scf.for %[[IV:.*]] = %[[C0]] to %[[C4]]
// CHECK-SAME:       step %[[C4]] iter_args(%[[ACC:.*]] = %[[C0_I32]]) -> (i32) {
//      CHECK:     %[[MIN:.*]] = affine.min #[[MAP]](%[[C4]], %[[IV]])[%[[C4]]]
//      CHECK:     %[[CAST:.*]] = arith.index_cast %[[MIN]] : index to i32
//      CHECK:     %[[ADD:.*]] = arith.addi %[[ACC]], %[[CAST]] : i32
//      CHECK:     scf.yield %[[ADD]]
//      CHECK:   }
//      CHECK:   return %[[RESULT]]
#map = affine_map<(d0, d1)[s0] -> (s0, d0 - d1)>
func.func @no_peeling_front() -> i32 {
  %c0_i32 = arith.constant 0 : i32
  %lb = arith.constant 0 : index
  %step = arith.constant 4 : index
  %ub = arith.constant 4 : index
  %r = scf.for %iv = %lb to %ub step %step iter_args(%arg = %c0_i32) -> i32 {
    %s = affine.min #map(%ub, %iv)[%step]
    %casted = arith.index_cast %s : index to i32
    %0 = arith.addi %arg, %casted : i32
    scf.yield %0 : i32
  }
  return %r : i32
}
