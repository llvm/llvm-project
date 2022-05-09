// RUN: mlir-opt %s --test-loop-unrolling="unroll-factor=3" -split-input-file -canonicalize | FileCheck %s
// RUN: mlir-opt %s --test-loop-unrolling="unroll-factor=1" -split-input-file -canonicalize | FileCheck %s --check-prefix UNROLL-BY-1

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
