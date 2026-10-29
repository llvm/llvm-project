// RUN: mlir-opt %s --mem2reg --split-input-file | FileCheck %s

// Check that a store inside a forall prevents promotion.

// CHECK-LABEL: func.func @forall_store
// CHECK-SAME: (%[[UB:.*]]: index)
// CHECK: %[[C5:.*]] = arith.constant 5 : i32
// CHECK: %[[C7:.*]] = arith.constant 7 : i32
// CHECK: %[[ALLOCA:.*]] = memref.alloca() : memref<i32>
// CHECK: memref.store %[[C5]], %[[ALLOCA]][]
// CHECK: scf.forall (%{{.*}}) in (%[[UB]]) {
// CHECK:   memref.store %[[C7]], %[[ALLOCA]][]
// CHECK: }
// CHECK: %[[LOAD:.*]] = memref.load %[[ALLOCA]][]
// CHECK: return %[[LOAD]] : i32
func.func @forall_store(%ub: index) -> i32 {
  %c5 = arith.constant 5 : i32
  %c7 = arith.constant 7 : i32
  %alloca = memref.alloca() : memref<i32>
  memref.store %c5, %alloca[] : memref<i32>
  scf.forall (%i) in (%ub) {
    memref.store %c7, %alloca[] : memref<i32>
  }
  %load = memref.load %alloca[] : memref<i32>
  return %load : i32
}

// -----

// Check that a store inside an if inside a forall prevents promotion.

// CHECK-LABEL: func.func @forall_if_store
// CHECK-SAME: (%[[UB:.*]]: index, %[[COND:.*]]: i1)
// CHECK: %[[C5:.*]] = arith.constant 5 : i32
// CHECK: %[[C7:.*]] = arith.constant 7 : i32
// CHECK: %[[ALLOCA:.*]] = memref.alloca() : memref<i32>
// CHECK: memref.store %[[C5]], %[[ALLOCA]][]
// CHECK: scf.forall (%{{.*}}) in (%[[UB]]) {
// CHECK:   scf.if %[[COND]] {
// CHECK:     memref.store %[[C7]], %[[ALLOCA]][]
// CHECK:   }
// CHECK: }
// CHECK: %[[LOAD:.*]] = memref.load %[[ALLOCA]][]
// CHECK: return %[[LOAD]] : i32
func.func @forall_if_store(%ub: index, %cond: i1) -> i32 {
  %c5 = arith.constant 5 : i32
  %c7 = arith.constant 7 : i32
  %alloca = memref.alloca() : memref<i32>
  memref.store %c5, %alloca[] : memref<i32>
  scf.forall (%i) in (%ub) {
    scf.if %cond {
      memref.store %c7, %alloca[] : memref<i32>
      scf.yield
    }
  }
  %load = memref.load %alloca[] : memref<i32>
  return %load : i32
}

// -----

// Check that a store inside a parallel prevents promotion.

// CHECK-LABEL: func.func @parallel_store
// CHECK-SAME: (%[[LB:.*]]: index, %[[UB:.*]]: index, %[[STEP:.*]]: index)
// CHECK: %[[C5:.*]] = arith.constant 5 : i32
// CHECK: %[[C7:.*]] = arith.constant 7 : i32
// CHECK: %[[ALLOCA:.*]] = memref.alloca() : memref<i32>
// CHECK: memref.store %[[C5]], %[[ALLOCA]][]
// CHECK: scf.parallel (%{{.*}}) = (%[[LB]]) to (%[[UB]]) step (%[[STEP]]) {
// CHECK:   memref.store %[[C7]], %[[ALLOCA]][]
// CHECK:   scf.reduce
// CHECK: }
// CHECK: %[[LOAD:.*]] = memref.load %[[ALLOCA]][]
// CHECK: return %[[LOAD]] : i32
func.func @parallel_store(%lb: index, %ub: index, %step: index) -> i32 {
  %c5 = arith.constant 5 : i32
  %c7 = arith.constant 7 : i32
  %alloca = memref.alloca() : memref<i32>
  memref.store %c5, %alloca[] : memref<i32>
  scf.parallel (%i) = (%lb) to (%ub) step (%step) {
    memref.store %c7, %alloca[] : memref<i32>
    scf.reduce
  }
  %load = memref.load %alloca[] : memref<i32>
  return %load : i32
}

// -----

// Check that a store inside an if inside a parallel prevents promotion.

// CHECK-LABEL: func.func @parallel_if_store
// CHECK-SAME: (%[[LB:.*]]: index, %[[UB:.*]]: index, %[[STEP:.*]]: index, %[[COND:.*]]: i1)
// CHECK: %[[C5:.*]] = arith.constant 5 : i32
// CHECK: %[[C7:.*]] = arith.constant 7 : i32
// CHECK: %[[ALLOCA:.*]] = memref.alloca() : memref<i32>
// CHECK: memref.store %[[C5]], %[[ALLOCA]][]
// CHECK: scf.parallel (%{{.*}}) = (%[[LB]]) to (%[[UB]]) step (%[[STEP]]) {
// CHECK:   scf.if %[[COND]] {
// CHECK:     memref.store %[[C7]], %[[ALLOCA]][]
// CHECK:   }
// CHECK:   scf.reduce
// CHECK: }
// CHECK: %[[LOAD:.*]] = memref.load %[[ALLOCA]][]
// CHECK: return %[[LOAD]] : i32
func.func @parallel_if_store(%lb: index, %ub: index, %step: index, %cond: i1) -> i32 {
  %c5 = arith.constant 5 : i32
  %c7 = arith.constant 7 : i32
  %alloca = memref.alloca() : memref<i32>
  memref.store %c5, %alloca[] : memref<i32>
  scf.parallel (%i) = (%lb) to (%ub) step (%step) {
    scf.if %cond {
      memref.store %c7, %alloca[] : memref<i32>
      scf.yield
    }
    scf.reduce
  }
  %load = memref.load %alloca[] : memref<i32>
  return %load : i32
}

// -----

// Check that a store inside a reduce region prevents promotion.

// CHECK-LABEL: func.func @parallel_reduce_store
// CHECK-SAME: (%[[LB:.*]]: index, %[[UB:.*]]: index, %[[STEP:.*]]: index)
// CHECK: %[[C5:.*]] = arith.constant 5 : i32
// CHECK: %[[C0:.*]] = arith.constant 0 : i32
// CHECK: %[[ALLOCA:.*]] = memref.alloca() : memref<i32>
// CHECK: memref.store %[[C5]], %[[ALLOCA]][]
// CHECK: %[[RES:.*]] = scf.parallel (%{{.*}}) = (%[[LB]]) to (%[[UB]]) step (%[[STEP]]) init (%[[C0]]) -> i32 {
// CHECK:   %[[C1:.*]] = arith.constant 1 : i32
// CHECK:   scf.reduce(%[[C1]] : i32) {
// CHECK:   ^{{.*}}(%[[LHS:.*]]: i32, %[[RHS:.*]]: i32):
// CHECK:     %[[SUM:.*]] = arith.addi %[[LHS]], %[[RHS]] : i32
// CHECK:     memref.store %[[SUM]], %[[ALLOCA]][]
// CHECK:     scf.reduce.return %[[SUM]] : i32
// CHECK:   }
// CHECK: }
// CHECK: %[[LOAD:.*]] = memref.load %[[ALLOCA]][]
// CHECK: return %[[LOAD]] : i32
func.func @parallel_reduce_store(%lb: index, %ub: index, %step: index) -> i32 {
  %c5 = arith.constant 5 : i32
  %c0 = arith.constant 0 : i32
  %alloca = memref.alloca() : memref<i32>
  memref.store %c5, %alloca[] : memref<i32>
  %res = scf.parallel (%i) = (%lb) to (%ub) step (%step) init (%c0) -> i32 {
    %c1 = arith.constant 1 : i32
    scf.reduce(%c1 : i32) {
    ^bb0(%lhs: i32, %rhs: i32):
      %sum = arith.addi %lhs, %rhs : i32
      memref.store %sum, %alloca[] : memref<i32>
      scf.reduce.return %sum : i32
    }
  }
  %load = memref.load %alloca[] : memref<i32>
  return %load : i32
}
