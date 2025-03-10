// RUN: mlir-opt %s --test-loop-unrolling="unroll-factor=3" -split-input-file -canonicalize | FileCheck %s
// RUN: mlir-opt %s --test-loop-unrolling="unroll-factor=1" -split-input-file -canonicalize | FileCheck %s --check-prefix UNROLL-BY-1
// RUN: mlir-opt %s --test-loop-unrolling="unroll-full=true" -split-input-file -canonicalize | FileCheck %s --check-prefix UNROLL-FULL

// CHECK-LABEL: scf_loop_unroll_single
func.func @scf_loop_unroll_single(%arg0 : f32, %arg1 : f32) -> f32 {
  %from = arith.constant 0 : index
  %to = arith.constant 10 : index
  %step = arith.constant 1 : index
  %sum = scf.for %iv = %from to %to step %step iter_args(%sum_iter = %arg0) -> (f32) {
    %next = arith.addf %sum_iter, %arg1 : f32
    scf.yield %next : f32
  }
  // CHECK:      %[[SUM:.*]] = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[V0:.*]] =
  // CHECK-NEXT:   %[[V1:.*]] = arith.addf %[[V0]]
  // CHECK-NEXT:   %[[V2:.*]] = arith.addf %[[V1]]
  // CHECK-NEXT:   %[[V3:.*]] = arith.addf %[[V2]]
  // CHECK-NEXT:   scf.yield %[[V3]]
  // CHECK-NEXT: }
  // CHECK-NEXT: %[[RES:.*]] = arith.addf %[[SUM]],
  // CHECK-NEXT: return %[[RES]]
  return %sum : f32
}

// CHECK-LABEL: scf_loop_unroll_double_symbolic_ub
// CHECK-SAME:     (%{{.*}}: f32, %{{.*}}: f32, %[[N:.*]]: index)
func.func @scf_loop_unroll_double_symbolic_ub(%arg0 : f32, %arg1 : f32, %n : index) -> (f32,f32) {
  %from = arith.constant 0 : index
  %step = arith.constant 1 : index
  %sum:2 = scf.for %iv = %from to %n step %step iter_args(%i0 = %arg0, %i1 = %arg1) -> (f32, f32) {
    %sum0 = arith.addf %i0, %arg0 : f32
    %sum1 = arith.addf %i1, %arg1 : f32
    scf.yield %sum0, %sum1 : f32, f32
  }
  return %sum#0, %sum#1 : f32, f32
  // CHECK-DAG:      %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[C3:.*]] = arith.constant 3 : index
  // CHECK-NEXT: %[[REM:.*]] = arith.remsi %[[N]], %[[C3]]
  // CHECK-NEXT: %[[UB:.*]] = arith.subi %[[N]], %[[REM]]
  // CHECK-NEXT: %[[SUM:.*]]:2 = scf.for {{.*}} = %[[C0]] to %[[UB]] step %[[C3]] iter_args
  // CHECK:      }
  // CHECK-NEXT: %[[SUM1:.*]]:2 = scf.for {{.*}} = %[[UB]] to %[[N]] step %[[C1]] iter_args(%[[V1:.*]] = %[[SUM]]#0, %[[V2:.*]] = %[[SUM]]#1)
  // CHECK:      }
  // CHECK-NEXT: return %[[SUM1]]#0, %[[SUM1]]#1
}

// UNROLL-BY-1-LABEL: scf_loop_unroll_factor_1_promote
func.func @scf_loop_unroll_factor_1_promote() -> () {
  %step = arith.constant 1 : index
  %lo = arith.constant 0 : index
  %hi = arith.constant 1 : index
  scf.for %i = %lo to %hi step %step {
    %x = "test.foo"(%i) : (index) -> i32
  }
  return
  // UNROLL-BY-1-NEXT: %[[C0:.*]] = arith.constant 0 : index
  // UNROLL-BY-1-NEXT: %{{.*}} = "test.foo"(%[[C0]]) : (index) -> i32
}

// UNROLL-FULL-LABEL: func @scf_loop_unroll_full_single
// UNROLL-FULL-SAME:    %[[ARG:.*]]: index)
func.func @scf_loop_unroll_full_single(%arg : index) -> index {
  %0 = arith.constant 0 : index
  %1 = arith.constant 1 : index
  %2 = arith.constant 4 : index
  %4 = scf.for %iv = %0 to %2 step %1 iter_args(%arg1 = %1) -> index {
    %3 = arith.addi %arg1, %arg : index
    scf.yield %3 : index
  }
  return %4 : index
  // UNROLL-FULL: %[[C1:.*]] = arith.constant 1 : index
  // UNROLL-FULL: %[[V0:.*]] = arith.addi %[[ARG]], %[[C1]] : index
  // UNROLL-FULL: %[[V1:.*]] = arith.addi %[[V0]], %[[ARG]] : index
  // UNROLL-FULL: %[[V2:.*]] = arith.addi %[[V1]], %[[ARG]] : index
  // UNROLL-FULL: %[[V3:.*]] = arith.addi %[[V2]], %[[ARG]] : index
  // UNROLL-FULL: return %[[V3]] : index
}

// UNROLL-FULL-LABEL: func @scf_loop_unroll_full_outter_loops
// UNROLL-FULL-SAME:    %[[ARG:.*]]: vector<4x4xindex>)
func.func @scf_loop_unroll_full_outter_loops(%arg0: vector<4x4xindex>) -> index {
  %0 = arith.constant 0 : index
  %1 = arith.constant 1 : index
  %2 = arith.constant 4 : index
  %6 = scf.for %arg1 = %0 to %2 step %1 iter_args(%it0 = %0) -> index {
    %5 = scf.for %arg2 = %0 to %2 step %1 iter_args(%it1 = %it0) -> index {
      %3 = vector.extract %arg0[%arg1, %arg2] : index from vector<4x4xindex>
      %4 = arith.addi %3, %it1 : index
      scf.yield %3 : index
    }
    scf.yield %5 : index
  }
  return %6 : index
  // UNROLL-FULL: %[[C0:.*]] = arith.constant 0 : index
  // UNROLL-FULL: %[[C1:.*]] = arith.constant 1 : index
  // UNROLL-FULL: %[[C4:.*]] = arith.constant 4 : index
  // UNROLL-FULL: %[[SUM0:.*]] = scf.for %[[IV:.*]] = %[[C0]] to %[[C4]] step %[[C1]] iter_args(%{{.*}} = %[[C0]])
  // UNROLL-FULL:   %[[VAL:.*]] = vector.extract %[[ARG]][0, %[[IV]]] : index from vector<4x4xindex>
  // UNROLL-FULL:   scf.yield %[[VAL]] : index
  // UNROLL-FULL: }
  // UNROLL-FULL: %[[SUM1:.*]] = scf.for %[[IV:.*]] = %[[C0]] to %[[C4]] step %[[C1]] iter_args(%{{.*}} = %[[SUM0]])
  // UNROLL-FULL:   %[[VAL:.*]] = vector.extract %[[ARG]][1, %[[IV]]] : index from vector<4x4xindex>
  // UNROLL-FULL:   scf.yield %[[VAL]] : index
  // UNROLL-FULL: }
  // UNROLL-FULL: %[[SUM2:.*]] = scf.for %[[IV:.*]] = %[[C0]] to %[[C4]] step %[[C1]] iter_args(%{{.*}} = %[[SUM1]])
  // UNROLL-FULL:   %[[VAL:.*]] = vector.extract %[[ARG]][2, %[[IV]]] : index from vector<4x4xindex>
  // UNROLL-FULL:   scf.yield %[[VAL]] : index
  // UNROLL-FULL: }
  // UNROLL-FULL: %[[SUM3:.*]] = scf.for %[[IV:.*]] = %[[C0]] to %[[C4]] step %[[C1]] iter_args(%{{.*}} = %[[SUM2]])
  // UNROLL-FULL:   %[[VAL:.*]] = vector.extract %[[ARG]][3, %[[IV]]] : index from vector<4x4xindex>
  // UNROLL-FULL:   scf.yield %[[VAL]] : index
  // UNROLL-FULL: }
  // UNROLL-FULL: return %[[SUM3]] : index
}
