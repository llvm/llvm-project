// RUN: mlir-opt %s -for-loop-peeling -canonicalize -split-input-file | FileCheck %s

//  CHECK-DAG: #[[MAP0:.*]] = affine_map<()[s0, s1, s2] -> (s1 - (s1 - s0) mod s2)>
//  CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0)[s0, s1] -> (s0, -d0 + s1)>
//  CHECK-DAG: #[[MAP2:.*]] = affine_map<()[s0, s1, s2] -> (s0, s2 - (s2 - (s2 - s1) mod s0))>
//      CHECK: func @fully_dynamic_bounds(
// CHECK-SAME:     %[[LB:.*]]: index, %[[UB:.*]]: index, %[[STEP:.*]]: index
//      CHECK:   %[[C0_I32:.*]] = constant 0 : i32
//      CHECK:   %[[NEW_UB:.*]] = affine.apply #[[MAP0]]()[%[[LB]], %[[UB]], %[[STEP]]]
//      CHECK:   %[[LOOP:.*]] = scf.for %[[IV:.*]] = %[[LB]] to %[[NEW_UB]]
// CHECK-SAME:       step %[[STEP]] iter_args(%[[ACC:.*]] = %[[C0_I32]]) -> (i32) {
//      CHECK:     %[[MINOP:.*]] = affine.min #[[MAP1]](%[[IV]])[%[[STEP]], %[[UB]]]
//      CHECK:     %[[CAST:.*]] = index_cast %[[MINOP]] : index to i32
//      CHECK:     %[[ADD:.*]] = addi %[[ACC]], %[[CAST]] : i32
//      CHECK:     scf.yield %[[ADD]]
//      CHECK:   }
//      CHECK:   %[[HAS_MORE:.*]] = cmpi slt, %[[NEW_UB]], %[[UB]]
//      CHECK:   %[[RESULT:.*]] = scf.if %[[HAS_MORE]] -> (i32) {
//      CHECK:     %[[REM:.*]] = affine.min #[[MAP2]]()[%[[STEP]], %[[LB]], %[[UB]]]
//      CHECK:     %[[CAST2:.*]] = index_cast %[[REM]]
//      CHECK:     %[[ADD2:.*]] = addi %[[LOOP]], %[[CAST2]]
//      CHECK:     scf.yield %[[ADD2]]
//      CHECK:   } else {
//      CHECK:     scf.yield %[[LOOP]]
//      CHECK:   }
//      CHECK:   return %[[RESULT]]
#map = affine_map<(d0, d1)[s0] -> (s0, d0 - d1)>
func @fully_dynamic_bounds(%lb : index, %ub: index, %step: index) -> i32 {
  %c0 = constant 0 : i32
  %r = scf.for %iv = %lb to %ub step %step iter_args(%arg = %c0) -> i32 {
    %s = affine.min #map(%ub, %iv)[%step]
    %casted = index_cast %s : index to i32
    %0 = addi %arg, %casted : i32
    scf.yield %0 : i32
  }
  return %r : i32
}

// -----

//  CHECK-DAG: #[[MAP:.*]] = affine_map<(d0) -> (4, -d0 + 17)>
//      CHECK: func @fully_static_bounds(
//  CHECK-DAG:   %[[C0_I32:.*]] = constant 0 : i32
//  CHECK-DAG:   %[[C1_I32:.*]] = constant 1 : i32
//  CHECK-DAG:   %[[C0:.*]] = constant 0 : index
//  CHECK-DAG:   %[[C4:.*]] = constant 4 : index
//  CHECK-DAG:   %[[C16:.*]] = constant 16 : index
//      CHECK:   %[[LOOP:.*]] = scf.for %[[IV:.*]] = %[[C0]] to %[[C16]]
// CHECK-SAME:       step %[[C4]] iter_args(%[[ACC:.*]] = %[[C0_I32]]) -> (i32) {
//      CHECK:     %[[MINOP:.*]] = affine.min #[[MAP]](%[[IV]])
//      CHECK:     %[[CAST:.*]] = index_cast %[[MINOP]] : index to i32
//      CHECK:     %[[ADD:.*]] = addi %[[ACC]], %[[CAST]] : i32
//      CHECK:     scf.yield %[[ADD]]
//      CHECK:   }
//      CHECK:   %[[RESULT:.*]] = addi %[[LOOP]], %[[C1_I32]] : i32
//      CHECK:   return %[[RESULT]]
#map = affine_map<(d0, d1)[s0] -> (s0, d0 - d1)>
func @fully_static_bounds() -> i32 {
  %c0_i32 = constant 0 : i32
  %lb = constant 0 : index
  %step = constant 4 : index
  %ub = constant 17 : index
  %r = scf.for %iv = %lb to %ub step %step
               iter_args(%arg = %c0_i32) -> i32 {
    %s = affine.min #map(%ub, %iv)[%step]
    %casted = index_cast %s : index to i32
    %0 = addi %arg, %casted : i32
    scf.yield %0 : i32
  }
  return %r : i32
}

// -----

//  CHECK-DAG: #[[MAP0:.*]] = affine_map<()[s0] -> ((s0 floordiv 4) * 4)>
//  CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0)[s0] -> (4, -d0 + s0)>
//  CHECK-DAG: #[[MAP2:.*]] = affine_map<()[s0] -> (4, s0 mod 4)>
//      CHECK: func @dynamic_upper_bound(
// CHECK-SAME:     %[[UB:.*]]: index
//  CHECK-DAG:   %[[C0_I32:.*]] = constant 0 : i32
//  CHECK-DAG:   %[[C0:.*]] = constant 0 : index
//  CHECK-DAG:   %[[C4:.*]] = constant 4 : index
//      CHECK:   %[[NEW_UB:.*]] = affine.apply #[[MAP0]]()[%[[UB]]]
//      CHECK:   %[[LOOP:.*]] = scf.for %[[IV:.*]] = %[[C0]] to %[[NEW_UB]]
// CHECK-SAME:       step %[[C4]] iter_args(%[[ACC:.*]] = %[[C0_I32]]) -> (i32) {
//      CHECK:     %[[MINOP:.*]] = affine.min #[[MAP1]](%[[IV]])[%[[UB]]]
//      CHECK:     %[[CAST:.*]] = index_cast %[[MINOP]] : index to i32
//      CHECK:     %[[ADD:.*]] = addi %[[ACC]], %[[CAST]] : i32
//      CHECK:     scf.yield %[[ADD]]
//      CHECK:   }
//      CHECK:   %[[HAS_MORE:.*]] = cmpi slt, %[[NEW_UB]], %[[UB]]
//      CHECK:   %[[RESULT:.*]] = scf.if %[[HAS_MORE]] -> (i32) {
//      CHECK:     %[[REM:.*]] = affine.min #[[MAP2]]()[%[[UB]]]
//      CHECK:     %[[CAST2:.*]] = index_cast %[[REM]]
//      CHECK:     %[[ADD2:.*]] = addi %[[LOOP]], %[[CAST2]]
//      CHECK:     scf.yield %[[ADD2]]
//      CHECK:   } else {
//      CHECK:     scf.yield %[[LOOP]]
//      CHECK:   }
//      CHECK:   return %[[RESULT]]
#map = affine_map<(d0, d1)[s0] -> (s0, d0 - d1)>
func @dynamic_upper_bound(%ub : index) -> i32 {
  %c0_i32 = constant 0 : i32
  %lb = constant 0 : index
  %step = constant 4 : index
  %r = scf.for %iv = %lb to %ub step %step
               iter_args(%arg = %c0_i32) -> i32 {
    %s = affine.min #map(%ub, %iv)[%step]
    %casted = index_cast %s : index to i32
    %0 = addi %arg, %casted : i32
    scf.yield %0 : i32
  }
  return %r : i32
}

// -----

//  CHECK-DAG: #[[MAP0:.*]] = affine_map<()[s0] -> ((s0 floordiv 4) * 4)>
//  CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0)[s0] -> (4, -d0 + s0)>
//  CHECK-DAG: #[[MAP2:.*]] = affine_map<()[s0] -> (4, s0 mod 4)>
//      CHECK: func @no_loop_results(
// CHECK-SAME:     %[[UB:.*]]: index, %[[MEMREF:.*]]: memref<i32>
//  CHECK-DAG:   %[[C0:.*]] = constant 0 : index
//  CHECK-DAG:   %[[C4:.*]] = constant 4 : index
//      CHECK:   %[[NEW_UB:.*]] = affine.apply #[[MAP0]]()[%[[UB]]]
//      CHECK:   scf.for %[[IV:.*]] = %[[C0]] to %[[NEW_UB]] step %[[C4]] {
//      CHECK:     %[[MINOP:.*]] = affine.min #[[MAP1]](%[[IV]])[%[[UB]]]
//      CHECK:     %[[LOAD:.*]] = memref.load %[[MEMREF]][]
//      CHECK:     %[[CAST:.*]] = index_cast %[[MINOP]] : index to i32
//      CHECK:     %[[ADD:.*]] = addi %[[LOAD]], %[[CAST]] : i32
//      CHECK:     memref.store %[[ADD]], %[[MEMREF]]
//      CHECK:   }
//      CHECK:   %[[HAS_MORE:.*]] = cmpi slt, %[[NEW_UB]], %[[UB]]
//      CHECK:   scf.if %[[HAS_MORE]] {
//      CHECK:     %[[REM:.*]] = affine.min #[[MAP2]]()[%[[UB]]]
//      CHECK:     %[[LOAD2:.*]] = memref.load %[[MEMREF]][]
//      CHECK:     %[[CAST2:.*]] = index_cast %[[REM]]
//      CHECK:     %[[ADD2:.*]] = addi %[[LOAD2]], %[[CAST2]]
//      CHECK:     memref.store %[[ADD2]], %[[MEMREF]]
//      CHECK:   }
//      CHECK:   return
#map = affine_map<(d0, d1)[s0] -> (s0, d0 - d1)>
func @no_loop_results(%ub : index, %d : memref<i32>) {
  %c0_i32 = constant 0 : i32
  %lb = constant 0 : index
  %step = constant 4 : index
  scf.for %iv = %lb to %ub step %step {
    %s = affine.min #map(%ub, %iv)[%step]
    %r = memref.load %d[] : memref<i32>
    %casted = index_cast %s : index to i32
    %0 = addi %r, %casted : i32
    memref.store %0, %d[] : memref<i32>
  }
  return
}
