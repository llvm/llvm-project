// RUN: mlir-opt %s -scf-for-loop-peeling=peel-front=true -split-input-file | FileCheck %s

//  CHECK-DAG: #[[MAP:.*]] = affine_map<(d0, d1)[s0] -> (4, d0 - d1)>
//      CHECK: func @fully_static_bounds(
//  CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : index
//  CHECK-DAG:   %[[C0_I32:.*]] = arith.constant 0 : i32
//  CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C17:.*]] = arith.constant 17 : index
//      CHECK:   %[[FIRST:.*]] = scf.for %[[IV:.*]] = %[[C0]] to %[[C4]]
// CHECK-SAME:       step %[[C4]] iter_args(%[[ACC:.*]] = %[[C0_I32]]) -> (i32) {
//      CHECK:     %[[MIN:.*]] = affine.min #[[MAP]](%[[C17]], %[[IV]])[%[[C4]]]
//      CHECK:     %[[CAST:.*]] = arith.index_cast %[[MIN]] : index to i32
//      CHECK:     %[[INIT:.*]] = arith.addi %[[ACC]], %[[CAST]] : i32
//      CHECK:     scf.yield %[[INIT]]
//      CHECK:   }
//      CHECK:   %[[RESULT:.*]] = scf.for %[[IV2:.*]] = %[[C4]] to %[[C17]]
// CHECK-SAME:       step %[[C4]] iter_args(%[[ACC2:.*]] = %[[FIRST]]) -> (i32) {
//      CHECK:     %[[MIN2:.*]] = affine.min #[[MAP]](%[[C17]], %[[IV2]])[%[[C4]]]
//      CHECK:     %[[CAST2:.*]] = arith.index_cast %[[MIN2]] : index to i32
//      CHECK:     %[[ADD:.*]] = arith.addi %[[ACC2]], %[[CAST2]] : i32
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

// CHECK-LABEL:   func.func @static_two_iterations_ub_used_in_loop(
// CHECK-SAME:                                                    %[[IFM1:.*]]: memref<1xi32>) -> i32 {
// CHECK:           %[[C0_I32:.*]] = arith.constant 0 : i32
// CHECK:           %[[C0_IDX:.*]] = arith.constant 0 : index
// CHECK:           %[[C4_IDX:.*]] = arith.constant 4 : index
// CHECK:           %[[C7_IDX:.*]] = arith.constant 7 : index
// CHECK:           %[[FOR1:.*]] = scf.for %[[IV1:.*]] = %[[C0_IDX]] to %[[C4_IDX]] step %[[C4_IDX]] iter_args(%[[ARG1:.*]] = %[[C0_I32]]) -> (i32) {
// CHECK:             %[[AFFINE_MIN1:.*]] = affine.min #map(%[[C7_IDX]], %[[IV1]]){{\[}}%[[C4_IDX]]]
// CHECK:             %[[CAST1:.*]] = arith.index_cast %[[AFFINE_MIN1]] : index to i32
// CHECK:             %[[ADDI1:.*]] = arith.addi %[[ARG1]], %[[CAST1]] : i32
// CHECK:             %[[LOAD1:.*]] = memref.load %[[IFM1]]{{\[}}%[[C7_IDX]]] : memref<1xi32>
// CHECK:             %[[ADDI2:.*]] = arith.addi %[[ADDI1]], %[[LOAD1]] : i32
// CHECK:             scf.yield %[[ADDI2]] : i32
// CHECK:           }
// CHECK:           %[[FOR2:.*]] = scf.for %[[IV2:.*]] = %[[C4_IDX]] to %[[C7_IDX]] step %[[C4_IDX]] iter_args(%[[ARG2:.*]] = %[[RESULT1:.*]]) -> (i32) {
// CHECK:             %[[AFFINE_MIN2:.*]] = affine.min #map(%[[C7_IDX]], %[[IV2]]){{\[}}%[[C4_IDX]]]
// CHECK:             %[[CAST2:.*]] = arith.index_cast %[[AFFINE_MIN2]] : index to i32
// CHECK:             %[[ADDI3:.*]] = arith.addi %[[ARG2]], %[[CAST2]] : i32
// CHECK:             %[[LOAD2:.*]] = memref.load %[[IFM1]]{{\[}}%[[C7_IDX]]] : memref<1xi32>
// CHECK:             %[[ADDI4:.*]] = arith.addi %[[ADDI3]], %[[LOAD2]] : i32
// CHECK:             scf.yield %[[ADDI4]] : i32
// CHECK:           }
// CHECK:           return %[[RESULT2:.*]] : i32
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

//  CHECK-DAG: #[[MAP:.*]] = affine_map<(d0, d1)[s0] -> (4, d0 - d1)>
//      CHECK: func @no_loop_results(
// CHECK-SAME:     %[[UB:.*]]: index, %[[MEMREF:.*]]: memref<i32>
//  CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : index
//  CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//  The peeled first iteration is bounded by %ub as well as by %lb + %step, so
//  it stays empty when the source loop has no iterations.
//      CHECK:   %[[MIN:.*]] = arith.minsi %[[UB]], %[[C4]] : index
//      CHECK:   %[[SPLIT:.*]] = arith.maxsi %[[MIN]], %{{.*}} : index
//      CHECK:   scf.for %[[IV:.*]] = %[[C0]] to %[[SPLIT]] step %[[C4]] {
//      CHECK:     %[[MIN:.*]] = affine.min #[[MAP]](%[[UB]], %[[IV]])[%[[C4]]]
//      CHECK:     %[[LOAD:.*]] = memref.load %[[MEMREF]][]
//      CHECK:     %[[CAST:.*]] = arith.index_cast %[[MIN]]
//      CHECK:     %[[ADD:.*]] = arith.addi %[[LOAD]], %[[CAST]] : i32
//      CHECK:     memref.store %[[ADD]], %[[MEMREF]]
//      CHECK:   }
//      CHECK:   scf.for %[[IV2:.*]] = %[[SPLIT]] to %[[UB]] step %[[C4]] {
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

//  CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0, d1)[s0] -> (s0, d0 - d1)>
//      CHECK: func @fully_dynamic_bounds(
// CHECK-SAME:     %[[LB:.*]]: index, %[[UB:.*]]: index, %[[STEP:.*]]: index
//      CHECK:   %[[C0_I32:.*]] = arith.constant 0 : i32
//      CHECK:   %[[SUM:.*]] = arith.addi %[[LB]], %[[STEP]] : index
//      CHECK:   %[[CAPPED:.*]] = arith.minsi %[[SUM]], %[[UB]] : index
//      CHECK:   %[[SPLIT:.*]] = arith.maxsi %[[LB]], %[[CAPPED]] : index
//      CHECK:   %[[FIRST:.*]] = scf.for %[[IV:.*]] = %[[LB]] to %[[SPLIT]]
// CHECK-SAME:       step %[[STEP]] iter_args(%[[ACC:.*]] = %[[C0_I32]]) -> (i32) {
//      CHECK:     %[[MIN:.*]] = affine.min #[[MAP1]](%[[UB]], %[[IV]])[%[[STEP]]]
//      CHECK:     %[[CAST:.*]] = arith.index_cast %[[MIN]] : index to i32
//      CHECK:     %[[ADD:.*]] = arith.addi %[[ACC]], %[[CAST]] : i32
//      CHECK:     scf.yield %[[ADD]]
//      CHECK:   }
//      CHECK:   %[[RESULT:.*]] = scf.for %[[IV2:.*]] = %[[SPLIT]] to %[[UB]]
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

//      CHECK: func @fully_dynamic_bounds_unsigned(
// CHECK-SAME:     %[[LB:.*]]: i32, %[[UB:.*]]: i32, %[[STEP:.*]]: i32
//      CHECK:   %[[C0:.*]] = arith.constant 0 : i32
//      CHECK:   %[[SUM:.*]] = arith.addi %[[LB]], %[[STEP]] : i32
//      CHECK:   %[[CAPPED:.*]] = arith.minui %[[SUM]], %[[UB]] : i32
//      CHECK:   %[[SPLIT:.*]] = arith.maxui %[[LB]], %[[CAPPED]] : i32
//      CHECK:   %[[FIRST:.*]] = scf.for unsigned %[[IV:.*]] = %[[LB]] to %[[SPLIT]]
//      CHECK:   %[[RESULT:.*]] = scf.for unsigned %[[IV2:.*]] = %[[SPLIT]] to %[[UB]]
//      CHECK:   return %[[RESULT]]
func.func @fully_dynamic_bounds_unsigned(%lb : i32, %ub: i32, %step: i32) -> i32 {
  %c0 = arith.constant 0 : i32
  %r = scf.for unsigned %iv = %lb to %ub step %step iter_args(%arg = %c0) -> (i32) : i32 {
    %0 = arith.addi %arg, %iv : i32
    scf.yield %0 : i32
  }
  return %r : i32
}

// -----

//  CHECK-DAG: #[[MAP:.*]] = affine_map<(d0, d1)[s0] -> (4, d0 - d1)>
//      CHECK: func @two_iteration_example(
//  CHECK-DAG:   %[[C0_I32:.*]] = arith.constant 0 : i32
//  CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
//  CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : index
//  CHECK-DAG:   %[[C8:.*]] = arith.constant 8 : index
//  CHECK-DAG:   %[[C6:.*]] = arith.constant 6 : index
//      CHECK:   %[[FIRST:.*]] = scf.for %[[IV:.*]] = %[[C2]] to %[[C6]]
// CHECK-SAME:       step %[[C4]] iter_args(%[[ACC:.*]] = %[[C0_I32]]) -> (i32) {
//      CHECK:     %[[MIN:.*]] = affine.min #[[MAP]](%[[C8]], %[[IV]])[%[[C4]]]
//      CHECK:     %[[CAST:.*]] = arith.index_cast %[[MIN]] : index to i32
//      CHECK:     %[[INIT:.*]] = arith.addi %[[ACC]], %[[CAST]] : i32
//      CHECK:     scf.yield %[[INIT]]
//      CHECK:   }
//      CHECK:   %[[RESULT:.*]] = scf.for %[[IV2:.*]] = %[[C6]] to %[[C8]]
// CHECK-SAME:       step %[[C4]] iter_args(%[[ACC2:.*]] = %[[FIRST]]) -> (i32) {
//      CHECK:     %[[MIN2:.*]] = affine.min #[[MAP]](%[[C8]], %[[IV2]])[%[[C4]]]
//      CHECK:     %[[CAST2:.*]] = arith.index_cast %[[MIN2]] : index to i32
//      CHECK:     %[[ADD:.*]] = arith.addi %[[ACC2]], %[[CAST2]] : i32
//      CHECK:     scf.yield %[[ADD]]
//      CHECK:   }
//      CHECK:   return %[[RESULT]]
#map = affine_map<(d0, d1)[s0] -> (s0, d0 - d1)>
func.func @two_iteration_example() -> i32 {
  %c0_i32 = arith.constant 0 : i32
  %lb = arith.constant 2 : index
  %step = arith.constant 4 : index
  %ub = arith.constant 8 : index
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

// -----

// CHECK-LABEL:   func.func @non_index_loop_bounds(
// CHECK-SAME:                                         %[[INIT:[^:]*]]: i32,
// CHECK-SAME:                                         %[[LB:[^:]*]]: i32,
// CHECK-SAME:                                         %[[N:[^:]*]]: i32) -> i32 {
// CHECK:           %[[C2:.*]] = arith.constant 2 : i32
// CHECK:           %[[SUM:.*]] = arith.addi %[[LB]], %[[C2]] : i32
// CHECK:           %[[CAPPED:.*]] = arith.minsi %[[SUM]], %[[N]] : i32
// CHECK:           %[[SPLIT:.*]] = arith.maxsi %[[LB]], %[[CAPPED]] : i32
// CHECK:           %[[FIRST:.*]] = scf.for %[[IV:.*]] = %[[LB]] to %[[SPLIT]] step %[[C2]] iter_args(%[[ACC:.*]] = %[[INIT]]) -> (i32) : i32 {
// CHECK:           %[[RESULT:.*]] = scf.for %[[IV2:.*]] = %[[SPLIT]] to %[[N]] step %[[C2]] iter_args(%[[ACC2:.*]] = %[[FIRST]]) -> (i32) : i32 {
// CHECK:           return %[[RESULT]] : i32
func.func @non_index_loop_bounds(%init: i32, %lb: i32, %n: i32) -> i32 {
  %c2 = arith.constant 2 : i32
  %r = scf.for %i = %lb to %n step %c2 iter_args(%a = %init) -> (i32) : i32 {
    %t = arith.addi %a, %i : i32
    scf.yield %t : i32
  }
  return %r : i32
}

// -----

// Tests loops with constant bounds where the step and (upper bound - lower bound) are
// very close, so `ceil(float(ub - lb) / step)` used to round to 1, and the front peeling
// was skipped. See https://github.com/llvm/llvm-project/issues/231062

// CHECK-LABEL:   func.func @peel_front_step_near_ub(
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[C16777216:.*]] = arith.constant 16777216 : index
// CHECK-DAG:       %[[C16777217:.*]] = arith.constant 16777217 : index
// CHECK:           scf.for %[[IV0:.*]] = %[[C0]] to %[[C16777216]] step %[[C16777216]] 
// CHECK:           scf.for %[[IV1:.*]] = %[[C16777216]] to %[[C16777217]] step %[[C16777216]]

func.func @peel_front_step_near_ub(%out: memref<2xindex>) {
  %c0 = arith.constant 0 : index
  %step = arith.constant 16777216 : index
  %ub = arith.constant 16777217 : index

  scf.for %i = %c0 to %ub step %step {
    %slot = arith.divui %i, %step : index
    memref.store %i, %out[%slot] : memref<2xindex>
  }
  return
}

// -----

// dynamic ub >= 2^63 (negative as signed). Split must use minui, clamped by maxui.
// CHECK-LABEL: func @dynamic_ub_high_bit
//  CHECK-SAME:   %[[UB:.*]]: index
//   CHECK-DAG:   %[[C:.*]] = arith.constant 4611686018427387905 : index
//       CHECK:   %[[MIN:.*]] = arith.minui %[[UB]], %[[C]] : index
//       CHECK:   %[[SPLIT:.*]] = arith.maxui %[[MIN]], %{{.*}} : index
//       CHECK:   scf.for unsigned %{{.*}} = %{{.*}} to %[[SPLIT]] step
//       CHECK:   scf.for unsigned %{{.*}} = %[[SPLIT]] to %[[UB]] step
func.func @dynamic_ub_high_bit(%ub: index, %m: memref<4xindex>) {
  %lb = arith.constant 1 : index
  %step = arith.constant 4611686018427387904 : index
  scf.for unsigned %i = %lb to %ub step %step {
    %d = arith.divui %i, %step : index
    memref.store %i, %m[%d] : memref<4xindex>
  }
  return
}

// -----

// constant bounds, ub = 3*2^62 (negative as signed), 3 iterations.
// A signed early-exit check would wrongly refuse.
// CHECK-LABEL: func @const_high_bit_peels
//       CHECK:   scf.for unsigned %{{.*}} = %{{[a-z0-9_]+}} to %[[SPLIT:c4611686018427387905[a-z0-9_]*]] step
//       CHECK:   scf.for unsigned %{{.*}} = %[[SPLIT]] to %{{[-a-z0-9_]+}} step
func.func @const_high_bit_peels(%m: memref<4xindex>) {
  %lb = arith.constant 1 : index
  %ub = arith.constant -4611686018427387904 : index
  %step = arith.constant 4611686018427387904 : index
  scf.for unsigned %i = %lb to %ub step %step {
    %d = arith.divui %i, %step : index
    memref.store %i, %m[%d] : memref<4xindex>
  }
  return
}

// -----

// unsigned, empty loop (lb > ub) whose lb + step wraps to 90.
// Without the clamp the main loop would start at 90. The split must stay at lb.
// CHECK-LABEL: func @unsigned_lb_plus_step_wraps
//   CHECK-NOT:   to %c90
//   CHECK-NOT:   scf.for unsigned %{{.*}} = %c90
//       CHECK:   scf.for unsigned %{{.*}} = %{{[-a-z0-9_]+}} to %[[SPLIT:[-a-z0-9_]+]] step
//       CHECK:   scf.for unsigned %{{.*}} = %[[SPLIT]] to %{{.*}} step
func.func @unsigned_lb_plus_step_wraps(%ub: i64, %m: memref<?xi64>) {
  %c0 = arith.constant 0 : index
  %lb = arith.constant -10 : i64
  %step = arith.constant 100 : i64
  scf.for unsigned %i = %lb to %ub step %step : i64 {
    memref.store %i, %m[%c0] : memref<?xi64>
  }
  return
}

// -----

// signed i64, empty loop (lb > ub) whose lb + step wraps to -3
// The split must stay at lb.
// CHECK-LABEL: func @signed_lb_plus_step_wraps
//   CHECK-NOT:   to %c-3
//   CHECK-NOT:   scf.for %{{.*}} = %c-3
//       CHECK:   scf.for %{{.*}} = %{{[-a-z0-9_]+}} to %[[SPLIT:[-a-z0-9_]+]] step
//       CHECK:   scf.for %{{.*}} = %[[SPLIT]] to %{{.*}} step
func.func @signed_lb_plus_step_wraps(%ub: i64, %m: memref<?xi64>) {
  %c0 = arith.constant 0 : index
  %lb = arith.constant 9223372036854775807 : i64
  %step = arith.constant 9223372036854775806 : i64
  scf.for %i = %lb to %ub step %step : i64 {
    memref.store %i, %m[%c0] : memref<?xi64>
  }
  return
}
