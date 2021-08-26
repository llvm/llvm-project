// RUN: mlir-opt %s -for-loop-peeling -canonicalize -split-input-file | FileCheck %s

//  CHECK-DAG: #[[MAP0:.*]] = affine_map<()[s0, s1, s2] -> (s1 - (s1 - s0) mod s2)>
//  CHECK-DAG: #[[MAP1:.*]] = affine_map<()[s0, s1, s2] -> (-(s0 - (s0 - s1) mod s2) + s0)>
//      CHECK: func @fully_dynamic_bounds(
// CHECK-SAME:     %[[LB:.*]]: index, %[[UB:.*]]: index, %[[STEP:.*]]: index
//      CHECK:   %[[C0_I32:.*]] = constant 0 : i32
//      CHECK:   %[[NEW_UB:.*]] = affine.apply #[[MAP0]]()[%[[LB]], %[[UB]], %[[STEP]]]
//      CHECK:   %[[LOOP:.*]] = scf.for %[[IV:.*]] = %[[LB]] to %[[NEW_UB]]
// CHECK-SAME:       step %[[STEP]] iter_args(%[[ACC:.*]] = %[[C0_I32]]) -> (i32) {
//      CHECK:     %[[CAST:.*]] = index_cast %[[STEP]] : index to i32
//      CHECK:     %[[ADD:.*]] = addi %[[ACC]], %[[CAST]] : i32
//      CHECK:     scf.yield %[[ADD]]
//      CHECK:   }
//      CHECK:   %[[HAS_MORE:.*]] = cmpi slt, %[[NEW_UB]], %[[UB]]
//      CHECK:   %[[RESULT:.*]] = scf.if %[[HAS_MORE]] -> (i32) {
//      CHECK:     %[[REM:.*]] = affine.apply #[[MAP1]]()[%[[UB]], %[[LB]], %[[STEP]]]
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

//      CHECK: func @fully_static_bounds(
//  CHECK-DAG:   %[[C0_I32:.*]] = constant 0 : i32
//  CHECK-DAG:   %[[C1_I32:.*]] = constant 1 : i32
//  CHECK-DAG:   %[[C4_I32:.*]] = constant 4 : i32
//  CHECK-DAG:   %[[C0:.*]] = constant 0 : index
//  CHECK-DAG:   %[[C4:.*]] = constant 4 : index
//  CHECK-DAG:   %[[C16:.*]] = constant 16 : index
//      CHECK:   %[[LOOP:.*]] = scf.for %[[IV:.*]] = %[[C0]] to %[[C16]]
// CHECK-SAME:       step %[[C4]] iter_args(%[[ACC:.*]] = %[[C0_I32]]) -> (i32) {
//      CHECK:     %[[ADD:.*]] = addi %[[ACC]], %[[C4_I32]] : i32
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
//  CHECK-DAG: #[[MAP1:.*]] = affine_map<()[s0] -> (s0 mod 4)>
//      CHECK: func @dynamic_upper_bound(
// CHECK-SAME:     %[[UB:.*]]: index
//  CHECK-DAG:   %[[C0_I32:.*]] = constant 0 : i32
//  CHECK-DAG:   %[[C4_I32:.*]] = constant 4 : i32
//  CHECK-DAG:   %[[C0:.*]] = constant 0 : index
//  CHECK-DAG:   %[[C4:.*]] = constant 4 : index
//      CHECK:   %[[NEW_UB:.*]] = affine.apply #[[MAP0]]()[%[[UB]]]
//      CHECK:   %[[LOOP:.*]] = scf.for %[[IV:.*]] = %[[C0]] to %[[NEW_UB]]
// CHECK-SAME:       step %[[C4]] iter_args(%[[ACC:.*]] = %[[C0_I32]]) -> (i32) {
//      CHECK:     %[[ADD:.*]] = addi %[[ACC]], %[[C4_I32]] : i32
//      CHECK:     scf.yield %[[ADD]]
//      CHECK:   }
//      CHECK:   %[[HAS_MORE:.*]] = cmpi slt, %[[NEW_UB]], %[[UB]]
//      CHECK:   %[[RESULT:.*]] = scf.if %[[HAS_MORE]] -> (i32) {
//      CHECK:     %[[REM:.*]] = affine.apply #[[MAP1]]()[%[[UB]]]
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
//  CHECK-DAG: #[[MAP1:.*]] = affine_map<()[s0] -> (s0 mod 4)>
//      CHECK: func @no_loop_results(
// CHECK-SAME:     %[[UB:.*]]: index, %[[MEMREF:.*]]: memref<i32>
//  CHECK-DAG:   %[[C4_I32:.*]] = constant 4 : i32
//  CHECK-DAG:   %[[C0:.*]] = constant 0 : index
//  CHECK-DAG:   %[[C4:.*]] = constant 4 : index
//      CHECK:   %[[NEW_UB:.*]] = affine.apply #[[MAP0]]()[%[[UB]]]
//      CHECK:   scf.for %[[IV:.*]] = %[[C0]] to %[[NEW_UB]] step %[[C4]] {
//      CHECK:     %[[LOAD:.*]] = memref.load %[[MEMREF]][]
//      CHECK:     %[[ADD:.*]] = addi %[[LOAD]], %[[C4_I32]] : i32
//      CHECK:     memref.store %[[ADD]], %[[MEMREF]]
//      CHECK:   }
//      CHECK:   %[[HAS_MORE:.*]] = cmpi slt, %[[NEW_UB]], %[[UB]]
//      CHECK:   scf.if %[[HAS_MORE]] {
//      CHECK:     %[[REM:.*]] = affine.apply #[[MAP1]]()[%[[UB]]]
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

// -----

// Test rewriting of affine.min ops. Make sure that more general cases than
// the ones above are successfully rewritten. Also make sure that the pattern
// does not rewrite affine.min ops that should not be rewritten.

//  CHECK-DAG: #[[MAP1:.*]] = affine_map<()[s0] -> (s0 + 1)>
//  CHECK-DAG: #[[MAP2:.*]] = affine_map<(d0)[s0, s1] -> (s0, -d0 + s1 - 1)>
//  CHECK-DAG: #[[MAP3:.*]] = affine_map<(d0)[s0, s1, s2] -> (s0, -d0 + s1, s2)>
//  CHECK-DAG: #[[MAP4:.*]] = affine_map<()[s0, s1, s2] -> (-(s0 - (s0 - s1) mod s2) + s0)>
//  CHECK-DAG: #[[MAP5:.*]] = affine_map<()[s0, s1, s2] -> (-(s0 - (s0 - s1) mod s2) + s0 + 1)>
//  CHECK-DAG: #[[MAP6:.*]] = affine_map<()[s0, s1, s2] -> (-(s0 - (s0 - s1) mod s2) + s0 - 1)>
//  CHECK-DAG: #[[MAP7:.*]] = affine_map<()[s0, s1, s2, s3] -> (s0, s2 - (s2 - (s2 - s1) mod s0), s3)>
//      CHECK: func @test_affine_min_rewrite(
// CHECK-SAME:     %[[LB:.*]]: index, %[[UB:.*]]: index, %[[STEP:.*]]: index,
// CHECK-SAME:     %[[MEMREF:.*]]: memref<?xindex>, %[[SOME_VAL:.*]]: index
//      CHECK:   scf.for %[[IV:.*]] = %[[LB]] to %{{.*}} step %[[STEP]] {
//                 (affine.min folded away)
//      CHECK:     memref.store %[[STEP]]
//                 (affine.min folded away)
//      CHECK:     memref.store %[[STEP]]
//      CHECK:     %[[RES2:.*]] = affine.apply #[[MAP1]]()[%[[STEP]]]
//      CHECK:     memref.store %[[RES2]]
//      CHECK:     %[[RES3:.*]] = affine.min #[[MAP2]](%[[IV]])[%[[STEP]], %[[UB]]]
//      CHECK:     memref.store %[[RES3]]
//      CHECK:     %[[RES4:.*]] = affine.min #[[MAP3]](%[[IV]])[%[[STEP]], %[[UB]], %[[SOME_VAL]]]
//      CHECK:     memref.store %[[RES4]]
//      CHECK:   }
//      CHECK:   scf.if {{.*}} {
//      CHECK:     %[[RES_IF_0:.*]] = affine.apply #[[MAP4]]()[%[[UB]], %[[LB]], %[[STEP]]]
//      CHECK:     memref.store %[[RES_IF_0]]
//      CHECK:     %[[RES_IF_1:.*]] = affine.apply #[[MAP5]]()[%[[UB]], %[[LB]], %[[STEP]]]
//      CHECK:     memref.store %[[RES_IF_1]]
//      CHECK:     %[[RES_IF_2:.*]] = affine.apply #[[MAP5]]()[%[[UB]], %[[LB]], %[[STEP]]]
//      CHECK:     memref.store %[[RES_IF_2]]
//      CHECK:     %[[RES_IF_3:.*]] = affine.apply #[[MAP6]]()[%[[UB]], %[[LB]], %[[STEP]]]
//      CHECK:     memref.store %[[RES_IF_3]]
//      CHECK:     %[[RES_IF_4:.*]] = affine.min #[[MAP7]]()[%[[STEP]], %[[LB]], %[[UB]], %[[SOME_VAL]]]
//      CHECK:     memref.store %[[RES_IF_4]]
#map0 = affine_map<(d0, d1)[s0] -> (s0, d0 - d1)>
#map1 = affine_map<(d0, d1)[s0] -> (d0 - d1 + 1, s0)>
#map2 = affine_map<(d0, d1)[s0] -> (s0 + 1, d0 - d1 + 1)>
#map3 = affine_map<(d0, d1)[s0] -> (s0, d0 - d1 - 1)>
#map4 = affine_map<(d0, d1, d2)[s0] -> (s0, d0 - d1, d2)>
func @test_affine_min_rewrite(%lb : index, %ub: index,
                              %step: index, %d : memref<?xindex>,
                              %some_val: index) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %c3 = constant 3 : index
  %c4 = constant 4 : index
  scf.for %iv = %lb to %ub step %step {
    // Most common case: Rewrite min(%ub - %iv, %step) to %step.
    %m0 = affine.min #map0(%ub, %iv)[%step]
    memref.store %m0, %d[%c0] : memref<?xindex>

    // Increase %ub - %iv a little bit, pattern should still apply.
    %m1 = affine.min #map1(%ub, %iv)[%step]
    memref.store %m1, %d[%c1] : memref<?xindex>

    // Rewrite min(%ub - %iv + 1, %step + 1) to %step + 1.
    %m2 = affine.min #map2(%ub, %iv)[%step]
    memref.store %m2, %d[%c2] : memref<?xindex>

    // min(%ub - %iv - 1, %step) cannot be simplified because %ub - %iv - 1
    // can be smaller than %step. (Can be simplified in if-statement.)
    %m3 = affine.min #map3(%ub, %iv)[%step]
    memref.store %m3, %d[%c3] : memref<?xindex>

    // min(%ub - %iv, %step, %some_val) cannot be simplified because the range
    // of %some_val is unknown.
    %m4 = affine.min #map4(%ub, %iv, %some_val)[%step]
    memref.store %m4, %d[%c4] : memref<?xindex>
  }
  return
}
