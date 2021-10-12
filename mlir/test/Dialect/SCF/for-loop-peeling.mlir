// RUN: mlir-opt %s -for-loop-peeling -canonicalize -split-input-file | FileCheck %s
// RUN: mlir-opt %s -for-loop-peeling=skip-partial=false -canonicalize -split-input-file | FileCheck %s -check-prefix=CHECK-NO-SKIP

//  CHECK-DAG: #[[MAP0:.*]] = affine_map<()[s0, s1, s2] -> (s1 - (s1 - s0) mod s2)>
//  CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0)[s0] -> (-d0 + s0)>
//      CHECK: func @fully_dynamic_bounds(
// CHECK-SAME:     %[[LB:.*]]: index, %[[UB:.*]]: index, %[[STEP:.*]]: index
//      CHECK:   %[[C0_I32:.*]] = arith.constant 0 : i32
//      CHECK:   %[[NEW_UB:.*]] = affine.apply #[[MAP0]]()[%[[LB]], %[[UB]], %[[STEP]]]
//      CHECK:   %[[LOOP:.*]] = scf.for %[[IV:.*]] = %[[LB]] to %[[NEW_UB]]
// CHECK-SAME:       step %[[STEP]] iter_args(%[[ACC:.*]] = %[[C0_I32]]) -> (i32) {
//      CHECK:     %[[CAST:.*]] = arith.index_cast %[[STEP]] : index to i32
//      CHECK:     %[[ADD:.*]] = arith.addi %[[ACC]], %[[CAST]] : i32
//      CHECK:     scf.yield %[[ADD]]
//      CHECK:   }
//      CHECK:   %[[RESULT:.*]] = scf.for %[[IV2:.*]] = %[[NEW_UB]] to %[[UB]]
// CHECK-SAME:       step %[[STEP]] iter_args(%[[ACC2:.*]] = %[[LOOP]]) -> (i32) {
//      CHECK:     %[[REM:.*]] = affine.apply #[[MAP1]](%[[IV2]])[%[[UB]]]
//      CHECK:     %[[CAST2:.*]] = arith.index_cast %[[REM]]
//      CHECK:     %[[ADD2:.*]] = arith.addi %[[ACC2]], %[[CAST2]]
//      CHECK:     scf.yield %[[ADD2]]
//      CHECK:   }
//      CHECK:   return %[[RESULT]]
#map = affine_map<(d0, d1)[s0] -> (s0, d0 - d1)>
func @fully_dynamic_bounds(%lb : index, %ub: index, %step: index) -> i32 {
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

//      CHECK: func @fully_static_bounds(
//  CHECK-DAG:   %[[C0_I32:.*]] = arith.constant 0 : i32
//  CHECK-DAG:   %[[C1_I32:.*]] = arith.constant 1 : i32
//  CHECK-DAG:   %[[C4_I32:.*]] = arith.constant 4 : i32
//  CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : index
//  CHECK-DAG:   %[[C16:.*]] = arith.constant 16 : index
//      CHECK:   %[[LOOP:.*]] = scf.for %[[IV:.*]] = %[[C0]] to %[[C16]]
// CHECK-SAME:       step %[[C4]] iter_args(%[[ACC:.*]] = %[[C0_I32]]) -> (i32) {
//      CHECK:     %[[ADD:.*]] = arith.addi %[[ACC]], %[[C4_I32]] : i32
//      CHECK:     scf.yield %[[ADD]]
//      CHECK:   }
//      CHECK:   %[[RESULT:.*]] = arith.addi %[[LOOP]], %[[C1_I32]] : i32
//      CHECK:   return %[[RESULT]]
#map = affine_map<(d0, d1)[s0] -> (s0, d0 - d1)>
func @fully_static_bounds() -> i32 {
  %c0_i32 = arith.constant 0 : i32
  %lb = arith.constant 0 : index
  %step = arith.constant 4 : index
  %ub = arith.constant 17 : index
  %r = scf.for %iv = %lb to %ub step %step
               iter_args(%arg = %c0_i32) -> i32 {
    %s = affine.min #map(%ub, %iv)[%step]
    %casted = arith.index_cast %s : index to i32
    %0 = arith.addi %arg, %casted : i32
    scf.yield %0 : i32
  }
  return %r : i32
}

// -----

//  CHECK-DAG: #[[MAP0:.*]] = affine_map<()[s0] -> ((s0 floordiv 4) * 4)>
//  CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0)[s0] -> (-d0 + s0)>
//      CHECK: func @dynamic_upper_bound(
// CHECK-SAME:     %[[UB:.*]]: index
//  CHECK-DAG:   %[[C0_I32:.*]] = arith.constant 0 : i32
//  CHECK-DAG:   %[[C4_I32:.*]] = arith.constant 4 : i32
//  CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : index
//      CHECK:   %[[NEW_UB:.*]] = affine.apply #[[MAP0]]()[%[[UB]]]
//      CHECK:   %[[LOOP:.*]] = scf.for %[[IV:.*]] = %[[C0]] to %[[NEW_UB]]
// CHECK-SAME:       step %[[C4]] iter_args(%[[ACC:.*]] = %[[C0_I32]]) -> (i32) {
//      CHECK:     %[[ADD:.*]] = arith.addi %[[ACC]], %[[C4_I32]] : i32
//      CHECK:     scf.yield %[[ADD]]
//      CHECK:   }
//      CHECK:   %[[RESULT:.*]] = scf.for %[[IV2:.*]] = %[[NEW_UB]] to %[[UB]]
// CHECK-SAME:       step %[[C4]] iter_args(%[[ACC2:.*]] = %[[LOOP]]) -> (i32) {
//      CHECK:     %[[REM:.*]] = affine.apply #[[MAP1]](%[[IV2]])[%[[UB]]]
//      CHECK:     %[[CAST2:.*]] = arith.index_cast %[[REM]]
//      CHECK:     %[[ADD2:.*]] = arith.addi %[[ACC2]], %[[CAST2]]
//      CHECK:     scf.yield %[[ADD2]]
//      CHECK:   }
//      CHECK:   return %[[RESULT]]
#map = affine_map<(d0, d1)[s0] -> (s0, d0 - d1)>
func @dynamic_upper_bound(%ub : index) -> i32 {
  %c0_i32 = arith.constant 0 : i32
  %lb = arith.constant 0 : index
  %step = arith.constant 4 : index
  %r = scf.for %iv = %lb to %ub step %step
               iter_args(%arg = %c0_i32) -> i32 {
    %s = affine.min #map(%ub, %iv)[%step]
    %casted = arith.index_cast %s : index to i32
    %0 = arith.addi %arg, %casted : i32
    scf.yield %0 : i32
  }
  return %r : i32
}

// -----

//  CHECK-DAG: #[[MAP0:.*]] = affine_map<()[s0] -> ((s0 floordiv 4) * 4)>
//  CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0)[s0] -> (-d0 + s0)>
//      CHECK: func @no_loop_results(
// CHECK-SAME:     %[[UB:.*]]: index, %[[MEMREF:.*]]: memref<i32>
//  CHECK-DAG:   %[[C4_I32:.*]] = arith.constant 4 : i32
//  CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : index
//      CHECK:   %[[NEW_UB:.*]] = affine.apply #[[MAP0]]()[%[[UB]]]
//      CHECK:   scf.for %[[IV:.*]] = %[[C0]] to %[[NEW_UB]] step %[[C4]] {
//      CHECK:     %[[LOAD:.*]] = memref.load %[[MEMREF]][]
//      CHECK:     %[[ADD:.*]] = arith.addi %[[LOAD]], %[[C4_I32]] : i32
//      CHECK:     memref.store %[[ADD]], %[[MEMREF]]
//      CHECK:   }
//      CHECK:   scf.for %[[IV2:.*]] = %[[NEW_UB]] to %[[UB]] step %[[C4]] {
//      CHECK:     %[[REM:.*]] = affine.apply #[[MAP1]](%[[IV2]])[%[[UB]]]
//      CHECK:     %[[LOAD2:.*]] = memref.load %[[MEMREF]][]
//      CHECK:     %[[CAST2:.*]] = arith.index_cast %[[REM]]
//      CHECK:     %[[ADD2:.*]] = arith.addi %[[LOAD2]], %[[CAST2]]
//      CHECK:     memref.store %[[ADD2]], %[[MEMREF]]
//      CHECK:   }
//      CHECK:   return
#map = affine_map<(d0, d1)[s0] -> (s0, d0 - d1)>
func @no_loop_results(%ub : index, %d : memref<i32>) {
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

// Test rewriting of affine.min/max ops. Make sure that more general cases than
// the ones above are successfully rewritten. Also make sure that the pattern
// does not rewrite ops that should not be rewritten.

//  CHECK-DAG: #[[MAP1:.*]] = affine_map<()[s0] -> (s0 + 1)>
//  CHECK-DAG: #[[MAP2:.*]] = affine_map<(d0)[s0, s1] -> (s0, -d0 + s1 - 1)>
//  CHECK-DAG: #[[MAP3:.*]] = affine_map<(d0)[s0, s1, s2] -> (s0, -d0 + s1, s2)>
//  CHECK-DAG: #[[MAP4:.*]] = affine_map<()[s0] -> (-s0)>
//  CHECK-DAG: #[[MAP5:.*]] = affine_map<(d0)[s0] -> (-d0 + s0)>
//  CHECK-DAG: #[[MAP6:.*]] = affine_map<(d0)[s0] -> (-d0 + s0 + 1)>
//  CHECK-DAG: #[[MAP7:.*]] = affine_map<(d0)[s0] -> (-d0 + s0 - 1)>
//  CHECK-DAG: #[[MAP8:.*]] = affine_map<(d0)[s0] -> (d0 - s0)>
//      CHECK: func @test_affine_op_rewrite(
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
//      CHECK:     %[[RES5:.*]] = affine.apply #[[MAP4]]()[%[[STEP]]]
//      CHECK:     memref.store %[[RES5]]
//      CHECK:   }
//      CHECK:   scf.for %[[IV2:.*]] = {{.*}} to %[[UB]] step %[[STEP]] {
//      CHECK:     %[[RES_IF_0:.*]] = affine.apply #[[MAP5]](%[[IV2]])[%[[UB]]]
//      CHECK:     memref.store %[[RES_IF_0]]
//      CHECK:     %[[RES_IF_1:.*]] = affine.apply #[[MAP6]](%[[IV2]])[%[[UB]]]
//      CHECK:     memref.store %[[RES_IF_1]]
//      CHECK:     %[[RES_IF_2:.*]] = affine.apply #[[MAP6]](%[[IV2]])[%[[UB]]]
//      CHECK:     memref.store %[[RES_IF_2]]
//      CHECK:     %[[RES_IF_3:.*]] = affine.apply #[[MAP7]](%[[IV2]])[%[[UB]]]
//      CHECK:     memref.store %[[RES_IF_3]]
//      CHECK:     %[[RES_IF_4:.*]] = affine.min #[[MAP3]](%[[IV2]])[%[[STEP]], %[[UB]], %[[SOME_VAL]]]
//      CHECK:     memref.store %[[RES_IF_4]]
//      CHECK:     %[[RES_IF_5:.*]] = affine.apply #[[MAP8]](%[[IV2]])[%[[UB]]]
//      CHECK:     memref.store %[[RES_IF_5]]
#map0 = affine_map<(d0, d1)[s0] -> (s0, d0 - d1)>
#map1 = affine_map<(d0, d1)[s0] -> (d0 - d1 + 1, s0)>
#map2 = affine_map<(d0, d1)[s0] -> (s0 + 1, d0 - d1 + 1)>
#map3 = affine_map<(d0, d1)[s0] -> (s0, d0 - d1 - 1)>
#map4 = affine_map<(d0, d1, d2)[s0] -> (s0, d0 - d1, d2)>
#map5 = affine_map<(d0, d1)[s0] -> (-s0, -d0 + d1)>
func @test_affine_op_rewrite(%lb : index, %ub: index,
                             %step: index, %d : memref<?xindex>,
                             %some_val: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
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

    // Rewrite max(-%ub + %iv, -%step) to -%ub + %iv (and -%step in the scf.if).
    %m5 = affine.max #map5(%ub, %iv)[%step]
    memref.store %m5, %d[%c5] : memref<?xindex>
  }
  return
}

// -----

//     CHECK: func @nested_loops
//     CHECK:   scf.for {{.*}} {
//     CHECK:     scf.for {{.*}} {
//     CHECK:     }
//     CHECK:     scf.for {{.*}} {
//     CHECK:     }
//     CHECK:   }
//     CHECK:   scf.for {{.*}} {
//     CHECK:     scf.for {{.*}} {
//     CHECK:     }
// CHECK-NOT:     scf.for
//     CHECK:   }

//     CHECK-NO-SKIP: func @nested_loops
//     CHECK-NO-SKIP:   scf.for {{.*}} {
//     CHECK-NO-SKIP:     scf.for {{.*}} {
//     CHECK-NO-SKIP:     }
//     CHECK-NO-SKIP:     scf.for {{.*}} {
//     CHECK-NO-SKIP:     }
//     CHECK-NO-SKIP:   }
//     CHECK-NO-SKIP:   scf.for {{.*}} {
//     CHECK-NO-SKIP:     scf.for {{.*}} {
//     CHECK-NO-SKIP:     }
//     CHECK-NO-SKIP:     scf.for {{.*}} {
//     CHECK-NO-SKIP:     }
//     CHECK-NO-SKIP:   }
#map = affine_map<(d0, d1)[s0] -> (s0, d0 - d1)>
func @nested_loops(%lb0: index, %lb1 : index, %ub0: index, %ub1: index,
                   %step: index) -> i32 {
  %c0 = arith.constant 0 : i32
  %r0 = scf.for %iv0 = %lb0 to %ub0 step %step iter_args(%arg0 = %c0) -> i32 {
    %r1 = scf.for %iv1 = %lb1 to %ub1 step %step iter_args(%arg1 = %arg0) -> i32 {
      %s = affine.min #map(%ub1, %iv1)[%step]
      %casted = arith.index_cast %s : index to i32
      %0 = arith.addi %arg1, %casted : i32
      scf.yield %0 : i32
    }
    %1 = arith.addi %arg0, %r1 : i32
    scf.yield %1 : i32
  }
  return %r0 : i32
}
