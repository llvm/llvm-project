// RUN: mlir-opt %s -scf-for-loop-peeling -canonicalize -split-input-file -verify-diagnostics | FileCheck %s
// RUN: mlir-opt %s -scf-for-loop-peeling=skip-partial=false -canonicalize -split-input-file -verify-diagnostics | FileCheck %s -check-prefix=CHECK-NO-SKIP


//  CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0)[s0] -> (-d0 + s0)>
//      CHECK: func @fully_dynamic_bounds(
// CHECK-SAME:     %[[LB:.*]]: index, %[[UB:.*]]: index, %[[STEP:.*]]: index
//      CHECK:   %[[C0_I32:.*]] = arith.constant 0 : i32
//      CHECK:   %[[LB_EFF:.*]] = arith.minsi %[[LB]], %[[UB]] : index
//      CHECK:   %[[DIFF:.*]] = arith.subi %[[UB]], %[[LB_EFF]] : index
//      CHECK:   %[[SPLIT_REM:.*]] = arith.remui %[[DIFF]], %[[STEP]] : index
//      CHECK:   %[[NEW_UB:.*]] = arith.subi %[[UB]], %[[SPLIT_REM]] : index
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

// CHECK-LABEL:   func.func @static_two_iterations_ub_used_in_loop(
// CHECK-SAME:                                                         %[[IFM1:.*]]: memref<1xi32>) -> i32 {
// CHECK:           %[[C7_I32:.*]] = arith.constant 7 : i32
// CHECK:           %[[C7_IDX:.*]] = arith.constant 7 : index
// CHECK:           %[[LOAD1:.*]] = memref.load %[[IFM1]]{{\[}}%[[C7_IDX]]] : memref<1xi32>
// CHECK:           %[[ADDI1:.*]] = arith.addi %[[LOAD1]], %[[C7_I32]] : i32
// CHECK:           %[[LOAD2:.*]] = memref.load %[[IFM1]]{{\[}}%[[C7_IDX]]] : memref<1xi32>
// CHECK:           %[[ADDI2:.*]] = arith.addi %[[ADDI1]], %[[LOAD2]] : i32
// CHECK:           return %[[ADDI2]] : i32
// CHECK:         }

#map = affine_map<(d0, d1)[s0] -> (s0, d0 - d1)>
func.func @static_two_iterations_ub_used_in_loop(%arg0: memref<1xi32>) -> i32 {
  %c0_i32 = arith.constant 0 : i32
  %lb = arith.constant 0 : index
  %step = arith.constant 4 : index
  %ub = arith.constant 7 : index
  %r = scf.for %iv = %lb to %ub step %step iter_args(%arg = %c0_i32) -> i32 {
    %s = affine.min #map(%ub, %iv)[%step]
    %casted = arith.index_cast %s : index to i32
    %0 = arith.addi %arg, %casted : i32
    %1 = memref.load %arg0[%ub] : memref<1xi32>
    %result = arith.addi %0, %1 : i32
    scf.yield %result : i32
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
func.func @fully_static_bounds() -> i32 {
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

//      CHECK: func @fully_static_bounds_integers(
//  CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : i32
//  CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : i32
//  CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : i32
//  CHECK-DAG:   %[[C16:.*]] = arith.constant 16 : i32
//      CHECK:   %[[LOOP:.*]] = scf.for %[[IV:.*]] = %[[C0]] to %[[C16]]
// CHECK-SAME:       step %[[C4]] iter_args(%[[ACC:.*]] = %[[C0]]) -> (i32)
//      CHECK:     %[[MAP:.*]] = affine.min
//      CHECK:     %[[MAP_CAST:.*]] = arith.index_cast %[[MAP]]
//      CHECK:     %[[ADD:.*]] = arith.addi %[[ACC]], %[[MAP_CAST]] : i32
//      CHECK:     scf.yield %[[ADD]]
//      CHECK:   }
//      CHECK:   %[[RESULT:.*]] = arith.addi %[[LOOP]], %[[C1]] : i32
//      CHECK:   return %[[RESULT]]
#map = affine_map<(d0, d1)[s0] -> (s0, d0 - d1)>
func.func @fully_static_bounds_integers() -> i32 {
  %c0_i32 = arith.constant 0 : i32
  %lb = arith.constant 0 : i32
  %step = arith.constant 4 : i32
  %ub = arith.constant 17 : i32
  %r = scf.for %iv = %lb to %ub step %step
               iter_args(%arg = %c0_i32) -> i32 : i32 {
    %ub_index = arith.index_cast %ub : i32 to index
    %iv_index = arith.index_cast %iv : i32 to index
    %step_index = arith.index_cast %step : i32 to index
    %s = affine.min #map(%ub_index, %iv_index)[%step_index]
    %casted = arith.index_cast %s : index to i32
    %0 = arith.addi %arg, %casted : i32
    scf.yield %0 : i32
  }
  return %r : i32
}

// -----

//  CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0)[s0] -> (-d0 + s0)>
//      CHECK: func @dynamic_upper_bound(
// CHECK-SAME:     %[[UB:.*]]: index
//  CHECK-DAG:   %[[C0_I32:.*]] = arith.constant 0 : i32
//  CHECK-DAG:   %[[C4_I32:.*]] = arith.constant 4 : i32
//  CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : index
//      CHECK:   %[[LB_EFF:.*]] = arith.minsi %[[UB]], %[[C0]] : index
//      CHECK:   %[[DIFF:.*]] = arith.subi %[[UB]], %[[LB_EFF]] : index
//      CHECK:   %[[SPLIT_REM:.*]] = arith.remui %[[DIFF]], %[[C4]] : index
//      CHECK:   %[[NEW_UB:.*]] = arith.subi %[[UB]], %[[SPLIT_REM]] : index
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
func.func @dynamic_upper_bound(%ub : index) -> i32 {
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

//  CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0)[s0] -> (-d0 + s0)>
//      CHECK: func @no_loop_results(
// CHECK-SAME:     %[[UB:.*]]: index, %[[MEMREF:.*]]: memref<i32>
//  CHECK-DAG:   %[[C4_I32:.*]] = arith.constant 4 : i32
//  CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : index
//      CHECK:   %[[LB_EFF:.*]] = arith.minsi %[[UB]], %[[C0]] : index
//      CHECK:   %[[DIFF:.*]] = arith.subi %[[UB]], %[[LB_EFF]] : index
//      CHECK:   %[[SPLIT_REM:.*]] = arith.remui %[[DIFF]], %[[C4]] : index
//      CHECK:   %[[NEW_UB:.*]] = arith.subi %[[UB]], %[[SPLIT_REM]] : index
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

// Test rewriting of affine.min/max ops. Make sure that more general cases than
// the ones above are successfully rewritten. Also make sure that the pattern
// does not rewrite ops that should not be rewritten.

//  CHECK-DAG: #[[MAP1:.*]] = affine_map<()[s0] -> (s0 + 1)>
//  CHECK-DAG: #[[MAP2:.*]] = affine_map<(d0)[s0, s1] -> (-d0 + s1 - 1, s0)>
//  CHECK-DAG: #[[MAP3:.*]] = affine_map<(d0)[s0, s1, s2] -> (-d0 + s1, s2, s0)>
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
func.func @test_affine_op_rewrite(%lb : index, %ub: index,
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
func.func @nested_loops(%lb0: index, %lb1 : index, %ub0: index, %ub1: index,
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

// -----

// CHECK-LABEL: func @regression
func.func @regression(%arg0: memref<i64>, %arg1: index) {
  %c0 = arith.constant 0 : index
  %0 = affine.apply affine_map<()[s0] -> (s0 * s0)>()[%arg1]
  scf.for %arg2 = %c0 to %0 step %arg1 {
    %1 = affine.min affine_map<(d0)[s0] -> (s0, -d0 + s0 * s0)>(%arg2)[%arg1]
    %2 = arith.index_cast %0 : index to i64
    memref.store %2, %arg0[] : memref<i64>
  }
  return
}

// -----

// Regression test: Make sure that we do not crash.

// CHECK-LABEL: func @zero_step(
//       CHECK:   scf.for
//       CHECK:     arith.index_cast
//       CHECK:   scf.for
//       CHECK:     arith.index_cast
func.func @zero_step(%arg0: memref<i64>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %foldto0 = arith.subi %c1, %c1 : index
  scf.for %arg2 = %c0 to %c1 step %foldto0 {
    %2 = arith.index_cast %arg2 : index to i64
    memref.store %2, %arg0[] : memref<i64>
  }
  return
}

// -----

// CHECK-LABEL:   func.func @non_index_loop_bounds(
// CHECK-SAME:                                         %[[INIT:.*]]: i32,
// CHECK-SAME:                                         %[[N:.*]]: i32) -> i32 {
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : i32
// CHECK-DAG:       %[[C2:.*]] = arith.constant 2 : i32
// CHECK:           %[[LB_EFF:.*]] = arith.minsi %[[N]], %[[C0]] : i32
// CHECK:           %[[DIFF:.*]] = arith.subi %[[N]], %[[LB_EFF]] : i32
// CHECK:           %[[SPLIT_REM:.*]] = arith.remui %[[DIFF]], %[[C2]] : i32
// CHECK:           %[[NEW_UB:.*]] = arith.subi %[[N]], %[[SPLIT_REM]] : i32
// CHECK:           %[[FIRST:.*]] = scf.for %[[IV:.*]] = %[[C0]] to %[[NEW_UB]] step %[[C2]] iter_args(%[[ACC:.*]] = %[[INIT]]) -> (i32) : i32 {
// CHECK:           %[[RESULT:.*]] = scf.for %[[IV2:.*]] = %[[NEW_UB]] to %[[N]] step %[[C2]] iter_args(%[[ACC2:.*]] = %[[FIRST]]) -> (i32) : i32 {
// CHECK:           return %[[RESULT]] : i32
func.func @non_index_loop_bounds(%init: i32, %n: i32) -> i32 {
  %c0 = arith.constant 0 : i32
  %c2 = arith.constant 2 : i32
  %r = scf.for %i = %c0 to %n step %c2 iter_args(%a = %init) -> (i32) : i32 {
    %t = arith.addi %a, %i : i32
    scf.yield %t : i32
  }
  return %r : i32
}

// -----

// Fully dynamic i32 bounds: nothing can fold. Unsigned min/sub/remui chain,
// iter_args threaded from the main loop into the partial loop.

// CHECK-LABEL: func @unsigned_back_peel_dynamic(
// CHECK-SAME:      %[[LB:.*]]: i32, %[[UB:.*]]: i32, %[[STEP:.*]]: i32
//      CHECK:   %[[C0:.*]] = arith.constant 0 : i32
//      CHECK:   %[[LBEFF:.*]] = arith.minui %[[LB]], %[[UB]] : i32
//      CHECK:   %[[DIFF:.*]] = arith.subi %[[UB]], %[[LBEFF]] : i32
//      CHECK:   %[[REM:.*]] = arith.remui %[[DIFF]], %[[STEP]] : i32
//      CHECK:   %[[SPLIT:.*]] = arith.subi %[[UB]], %[[REM]] : i32
//      CHECK:   %[[MAIN:.*]] = scf.for unsigned %[[IV:.*]] = %[[LB]] to %[[SPLIT]] step %[[STEP]] iter_args(%[[ACC:.*]] = %[[C0]]) -> (i32) : i32 {
//      CHECK:     %[[ADD:.*]] = arith.addi %[[ACC]], %[[IV]] : i32
//      CHECK:     scf.yield %[[ADD]] : i32
//      CHECK:   }
//      CHECK:   %[[RES:.*]] = scf.for unsigned %[[IV2:.*]] = %[[SPLIT]] to %[[UB]] step %[[STEP]] iter_args(%[[ACC2:.*]] = %[[MAIN]]) -> (i32) : i32 {
//      CHECK:     %[[ADD2:.*]] = arith.addi %[[ACC2]], %[[IV2]] : i32
//      CHECK:     scf.yield %[[ADD2]] : i32
//      CHECK:   }
//      CHECK:   return %[[RES]] : i32
func.func @unsigned_back_peel_dynamic(%lb: i32, %ub: i32, %step: i32) -> i32 {
  %c0 = arith.constant 0 : i32
  %r = scf.for unsigned %iv = %lb to %ub step %step iter_args(%acc = %c0) -> (i32) : i32 {
    %0 = arith.addi %acc, %iv : i32
    scf.yield %0 : i32
  }
  return %r : i32
}

// -----

// Constants that only make sense unsigned: lb = 0x7FFFFFF8, ub = 0xFFFFFFF9
// (printed -7), step 4. (ub - lb) = 2147483649, remainder 1, so split =
// 0xFFFFFFF8 (-8). The main loop runs 536870912 iterations. The partial loop
// is exactly one iteration, so the canonicalizer inlines it with iv = split.

func.func private @effect(i32)

// CHECK-LABEL: func @unsigned_back_peel_above_signed_max
//   CHECK-DAG:   %[[LB:.*]] = arith.constant 2147483640 : i32
//   CHECK-DAG:   %[[STEP:.*]] = arith.constant 4 : i32
//   CHECK-DAG:   %[[SPLIT:.*]] = arith.constant -8 : i32
//       CHECK:   scf.for unsigned %{{.*}} = %[[LB]] to %[[SPLIT]] step %[[STEP]] : i32 {
//       CHECK:     call @effect(%{{.*}}) : (i32) -> ()
//       CHECK:   }
//       CHECK:   call @effect(%[[SPLIT]]) : (i32) -> ()
func.func @unsigned_back_peel_above_signed_max() {
  %lb = arith.constant 2147483640 : i32
  %ub = arith.constant -7 : i32
  %step = arith.constant 4 : i32
  scf.for unsigned %iv = %lb to %ub step %step : i32 {
    func.call @effect(%iv) : (i32) -> ()
  }
  return
}

// -----

// Empty unsigned loop: lb = 0xFFFFFFFF, ub = 0 (ub <= lb). Signed code would
// read this as -1 to 0, which has one iteration. After peeling and
// canonicalization no iteration may remain.

func.func private @effect(i32)

// CHECK-LABEL: func @unsigned_back_peel_empty_loop
//   CHECK-NOT:   call @effect
//       CHECK:   return
func.func @unsigned_back_peel_empty_loop() {
  %lb = arith.constant -1 : i32
  %ub = arith.constant 0 : i32
  %step = arith.constant 2 : i32
  scf.for unsigned %iv = %lb to %ub step %step : i32 {
    func.call @effect(%iv) : (i32) -> ()
  }
  return
}

// -----

// Step divides the range (0x80000000 - 0 = 2^29 * 4): nothing to peel, so one
// loop stays. 0x80000000 prints as -2147483648, which signed code would read
// as an empty loop.

func.func private @effect(i32)

// CHECK-LABEL: func @unsigned_back_peel_step_divides_range
//       CHECK:   scf.for unsigned
//       CHECK:     call @effect
//       CHECK:   }
//   CHECK-NOT:   scf.for
//   CHECK-NOT:   call @effect
func.func @unsigned_back_peel_step_divides_range() {
  %lb = arith.constant 0 : i32
  %ub = arith.constant -2147483648 : i32
  %step = arith.constant 4 : i32
  scf.for unsigned %iv = %lb to %ub step %step : i32 {
    func.call @effect(%iv) : (i32) -> ()
  }
  return
}

// -----

// Constant step of 1 with dynamic bounds: step 1 always divides the range,
// so there is nothing to peel. One loop stays and no split bound is computed.

func.func private @effect(i32)

// CHECK-LABEL: func @unsigned_back_peel_step_one_dynamic_bounds(
// CHECK-SAME:      %[[LB:.*]]: i32, %[[UB:.*]]: i32
//      CHECK:   %[[STEP:.*]] = arith.constant 1 : i32
//      CHECK:   scf.for unsigned %{{.*}} = %[[LB]] to %[[UB]] step %[[STEP]] : i32 {
//      CHECK:     call @effect
//      CHECK:   }
//  CHECK-NOT:   scf.for
//  CHECK-NOT:   arith.remui
func.func @unsigned_back_peel_step_one_dynamic_bounds(%lb: i32, %ub: i32) {
  %step = arith.constant 1 : i32
  scf.for unsigned %iv = %lb to %ub step %step : i32 {
    func.call @effect(%iv) : (i32) -> ()
  }
  return
}

// -----

// Unsigned loop with an affine.min in the body: the loop is peeled, but the
// affine.min is not simplified in either loop (the simplification is signed).

#map = affine_map<(d0, d1)[s0] -> (s0, d0 - d1)>
func.func private @use(index)

// CHECK-LABEL: func @unsigned_no_bound_simplification
//       CHECK:   scf.for unsigned
//       CHECK:     affine.min
//       CHECK:   scf.for unsigned
//       CHECK:     affine.min
func.func @unsigned_no_bound_simplification(%lb: index, %ub: index, %step: index) {
  scf.for unsigned %iv = %lb to %ub step %step {
    %s = affine.min #map(%ub, %iv)[%step]
    func.call @use(%s) : (index) -> ()
  }
  return
}

// -----

// Signed empty loop whose range is not a multiple of the step: lb=10, ub=5,
// step=4. Without the clamp to lb, split was 2 and the partial loop ran iv=2.

func.func private @effect(index)

// CHECK-LABEL: func @signed_empty_loop_not_executed
//   CHECK-NOT:   call @effect
//       CHECK:   return
func.func @signed_empty_loop_not_executed() {
  %lb = arith.constant 10 : index
  %ub = arith.constant 5 : index
  %step = arith.constant 4 : index
  scf.for %iv = %lb to %ub step %step {
    func.call @effect(%iv) : (index) -> ()
  }
  return
}

// -----

// Signed i32 loop whose range (INT_MIN to INT_MAX - 4) exceeds the signed max.
// ub - lb = 4294967291 only makes sense as an unsigned value, so remui is
// needed for signed loops too. remainder 3, so split = 2147483640. The
// partial loop is exactly one iteration, so canonicalization inlines it.

func.func private @effect(i32)

// CHECK-LABEL: func @signed_range_above_signed_max
//   CHECK-DAG:   %[[LB:.*]] = arith.constant -2147483648 : i32
//   CHECK-DAG:   %[[STEP:.*]] = arith.constant 4 : i32
//   CHECK-DAG:   %[[SPLIT:.*]] = arith.constant 2147483640 : i32
//       CHECK:   scf.for %{{.*}} = %[[LB]] to %[[SPLIT]] step %[[STEP]] : i32 {
//       CHECK:     call @effect(%{{.*}}) : (i32) -> ()
//       CHECK:   }
//       CHECK:   call @effect(%[[SPLIT]]) : (i32) -> ()
func.func @signed_range_above_signed_max() {
  %lb = arith.constant -2147483648 : i32
  %ub = arith.constant 2147483643 : i32
  %step = arith.constant 4 : i32
  scf.for %iv = %lb to %ub step %step : i32 {
    func.call @effect(%iv) : (i32) -> ()
  }
  return
}
