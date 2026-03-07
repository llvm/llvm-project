// RUN: mlir-opt %s --mem2reg --split-input-file | FileCheck %s \
// RUN:   -implicit-check-not "memref.alloca" \
// RUN:   -implicit-check-not "memref.load" \
// RUN:   -implicit-check-not "memref.store"

// Check regions within if are promoted.

// CHECK-LABEL: func.func @if_load_only
// CHECK-SAME: (%[[COND:.*]]: i1)
// CHECK: %[[C5:.*]] = arith.constant 5 : i32
// CHECK: %[[RES:.*]] = scf.if %[[COND]] -> (i32)
// CHECK:   scf.yield %[[C5]] : i32
// CHECK: } else {
// CHECK:   scf.yield %[[C5]] : i32
// CHECK: }
// CHECK: return %[[RES]] : i32
func.func @if_load_only(%cond: i1) -> i32 {
  %c5 = arith.constant 5 : i32
  %alloca = memref.alloca() : memref<i32>
  memref.store %c5, %alloca[] : memref<i32>
  %res = scf.if %cond -> i32 {
    %load = memref.load %alloca[] : memref<i32>
    scf.yield %load : i32
  } else {
    scf.yield %c5 : i32
  }
  return %res : i32
}

// -----

// Check load promotion through an if with no else branch.

func.func private @use(i32)

// CHECK-LABEL: func.func @if_no_else_load
// CHECK-SAME: (%[[COND:.*]]: i1)
// CHECK: %[[C5:.*]] = arith.constant 5 : i32
// CHECK: scf.if %[[COND]] {
// CHECK:   call @use(%[[C5]])
// CHECK: }
// CHECK: call @use(%[[C5]])
func.func @if_no_else_load(%cond: i1) {
  %c5 = arith.constant 5 : i32
  %alloca = memref.alloca() : memref<i32>
  memref.store %c5, %alloca[] : memref<i32>
  scf.if %cond {
    %load = memref.load %alloca[] : memref<i32>
    func.call @use(%load) : (i32) -> ()
    scf.yield
  }
  %load2 = memref.load %alloca[] : memref<i32>
  func.call @use(%load2) : (i32) -> ()
  return
}

// -----

// Check store promotion through an if with no else branch.

func.func private @use(i32)

// CHECK-LABEL: func.func @if_no_else_store
// CHECK-SAME: (%[[COND:.*]]: i1)
// CHECK-DAG: %[[C5:.*]] = arith.constant 5 : i32
// CHECK-DAG: %[[C7:.*]] = arith.constant 7 : i32
// CHECK: %[[IF:.*]] = scf.if %[[COND]] -> (i32)
// CHECK:   scf.yield %[[C7]] : i32
// CHECK: } else {
// CHECK:   scf.yield %[[C5]] : i32
// CHECK: }
// CHECK: call @use(%[[IF]])
func.func @if_no_else_store(%cond: i1) {
  %c5 = arith.constant 5 : i32
  %c7 = arith.constant 7 : i32
  %alloca = memref.alloca() : memref<i32>
  memref.store %c5, %alloca[] : memref<i32>
  scf.if %cond {
    memref.store %c7, %alloca[] : memref<i32>
    scf.yield
  }
  %load = memref.load %alloca[] : memref<i32>
  func.call @use(%load) : (i32) -> ()
  return
}

// -----

// Check store promotion through nested ifs with no else branches.

func.func private @use(i32)

// CHECK-LABEL: func.func @if_nested_store
// CHECK-SAME: (%[[COND0:.*]]: i1, %[[COND1:.*]]: i1)
// CHECK-DAG: %[[C5:.*]] = arith.constant 5 : i32
// CHECK-DAG: %[[C7:.*]] = arith.constant 7 : i32
// CHECK: %[[OUTER:.*]] = scf.if %[[COND0]] -> (i32)
// CHECK:   %[[INNER:.*]] = scf.if %[[COND1]] -> (i32)
// CHECK:     scf.yield %[[C7]] : i32
// CHECK:   } else {
// CHECK:     scf.yield %[[C5]] : i32
// CHECK:   }
// CHECK:   scf.yield %[[INNER]] : i32
// CHECK: } else {
// CHECK:   scf.yield %[[C5]] : i32
// CHECK: }
// CHECK: call @use(%[[OUTER]])
func.func @if_nested_store(%cond0: i1, %cond1: i1) {
  %c5 = arith.constant 5 : i32
  %c7 = arith.constant 7 : i32
  %alloca = memref.alloca() : memref<i32>
  memref.store %c5, %alloca[] : memref<i32>
  scf.if %cond0 {
    scf.if %cond1 {
      memref.store %c7, %alloca[] : memref<i32>
      scf.yield
    }
    scf.yield
  }
  %load = memref.load %alloca[] : memref<i32>
  func.call @use(%load) : (i32) -> ()
  return
}

// -----

// Check load promotion through execute_region.

func.func private @use(i32)

// CHECK-LABEL: func.func @execute_region_load
// CHECK: %[[C5:.*]] = arith.constant 5 : i32
// CHECK: scf.execute_region {
// CHECK:   call @use(%[[C5]])
// CHECK:   scf.yield
// CHECK: }
func.func @execute_region_load() {
  %c5 = arith.constant 5 : i32
  %alloca = memref.alloca() : memref<i32>
  memref.store %c5, %alloca[] : memref<i32>
  scf.execute_region {
    %load = memref.load %alloca[] : memref<i32>
    func.call @use(%load) : (i32) -> ()
    scf.yield
  }
  return
}

// -----

// Check store promotion through execute_region.

// CHECK-LABEL: func.func @execute_region_store
// CHECK: %[[C7:.*]] = arith.constant 7 : i32
// CHECK: %[[RES:.*]] = scf.execute_region -> i32 {
// CHECK:   scf.yield %[[C7]] : i32
// CHECK: }
// CHECK: return %[[RES]] : i32
func.func @execute_region_store() -> i32 {
  %c5 = arith.constant 5 : i32
  %c7 = arith.constant 7 : i32
  %alloca = memref.alloca() : memref<i32>
  memref.store %c5, %alloca[] : memref<i32>
  scf.execute_region {
    memref.store %c7, %alloca[] : memref<i32>
    scf.yield
  }
  %load = memref.load %alloca[] : memref<i32>
  return %load : i32
}

// -----

// Check promotion through an execute_region with CFG control flow and a
// nested if containing a load. This ensures a block argument is created
// even in blocks with no direct slot use.

func.func private @use(i32)

// CHECK-LABEL: func.func @execute_region_cfg
// CHECK-SAME: (%[[COND0:.*]]: i1, %[[COND1:.*]]: i1)
// CHECK-DAG: %[[C7:.*]] = arith.constant 7 : i32
// CHECK-DAG: %[[C9:.*]] = arith.constant 9 : i32
// CHECK: %[[RES:.*]] = scf.execute_region -> i32 {
// CHECK:   cf.cond_br %[[COND0]], ^[[BB1:.*]], ^[[BB2:.*]]
// CHECK: ^[[BB1]]:
// CHECK:   cf.br ^[[BB3:.*]](%[[C7]] : i32)
// CHECK: ^[[BB2]]:
// CHECK:   cf.br ^[[BB3]](%[[C9]] : i32)
// CHECK: ^[[BB3]](%[[VAL:.*]]: i32):
// CHECK:   scf.if %[[COND1]] {
// CHECK:     call @use(%[[VAL]])
// CHECK:   }
// CHECK:   scf.yield %[[VAL]] : i32
// CHECK: }
// CHECK: call @use(%[[RES]])
func.func @execute_region_cfg(%cond0: i1, %cond1: i1) {
  %c5 = arith.constant 5 : i32
  %c7 = arith.constant 7 : i32
  %c9 = arith.constant 9 : i32
  %alloca = memref.alloca() : memref<i32>
  memref.store %c5, %alloca[] : memref<i32>
  scf.execute_region {
    cf.cond_br %cond0, ^bb1, ^bb2
  ^bb1:
    memref.store %c7, %alloca[] : memref<i32>
    cf.br ^bb3
  ^bb2:
    memref.store %c9, %alloca[] : memref<i32>
    cf.br ^bb3
  ^bb3:
    scf.if %cond1 {
      %load = memref.load %alloca[] : memref<i32>
      func.call @use(%load) : (i32) -> ()
      scf.yield
    }
    scf.yield
  }
  %load2 = memref.load %alloca[] : memref<i32>
  func.call @use(%load2) : (i32) -> ()
  return
}

// CHECK-LABEL: func.func @execute_region_cfg_no_use_at_all
// CHECK-SAME: (%[[COND0:.*]]: i1, %[[COND1:.*]]: i1)
// CHECK-DAG: %[[C7:.*]] = arith.constant 7 : i32
// CHECK-DAG: %[[C9:.*]] = arith.constant 9 : i32
// CHECK: %[[RES:.*]] = scf.execute_region -> i32 {
// CHECK:   cf.cond_br %[[COND0]], ^[[BB1:.*]], ^[[BB2:.*]]
// CHECK: ^[[BB1]]:
// CHECK:   cf.br ^[[BB3:.*]](%[[C7]] : i32)
// CHECK: ^[[BB2]]:
// CHECK:   cf.br ^[[BB3]](%[[C9]] : i32)
// CHECK: ^[[BB3]](%[[VAL:.*]]: i32):
// CHECK:   scf.yield %[[VAL]] : i32
// CHECK: }
// CHECK: return %[[RES]] : i32
func.func @execute_region_cfg_no_use_at_all(%cond0: i1, %cond1: i1) -> i32 {
  %c5 = arith.constant 5 : i32
  %c7 = arith.constant 7 : i32
  %c9 = arith.constant 9 : i32
  %alloca = memref.alloca() : memref<i32>
  memref.store %c5, %alloca[] : memref<i32>
  scf.execute_region {
    cf.cond_br %cond0, ^bb1, ^bb2
  ^bb1:
    memref.store %c7, %alloca[] : memref<i32>
    cf.br ^bb3
  ^bb2:
    memref.store %c9, %alloca[] : memref<i32>
    cf.br ^bb3
  ^bb3:
    scf.yield
  }
  %load2 = memref.load %alloca[] : memref<i32>
  return %load2 : i32
}

// CHECK-LABEL: func.func @execute_region_cfg_with_store
// CHECK-SAME: (%[[COND0:.*]]: i1, %[[COND1:.*]]: i1)
// CHECK-DAG: %[[C7:.*]] = arith.constant 7 : i32
// CHECK-DAG: %[[C9:.*]] = arith.constant 9 : i32
// CHECK-DAG: %[[C11:.*]] = arith.constant 11 : i32
// CHECK: %[[RES:.*]] = scf.execute_region -> i32 {
// CHECK:   cf.cond_br %[[COND0]], ^[[BB1:.*]], ^[[BB2:.*]]
// CHECK: ^[[BB1]]:
// CHECK:   cf.br ^[[BB3:.*]]
// CHECK: ^[[BB2]]:
// CHECK:   cf.br ^[[BB3]]
// CHECK: ^[[BB3]]:
// CHECK:   scf.yield %[[C11]] : i32
// CHECK: }
// CHECK: return %[[RES]] : i32
func.func @execute_region_cfg_with_store(%cond0: i1, %cond1: i1) -> i32 {
  %c5 = arith.constant 5 : i32
  %c7 = arith.constant 7 : i32
  %c9 = arith.constant 9 : i32
  %c11 = arith.constant 11 : i32
  %alloca = memref.alloca() : memref<i32>
  memref.store %c5, %alloca[] : memref<i32>
  scf.execute_region {
    cf.cond_br %cond0, ^bb1, ^bb2
  ^bb1:
    memref.store %c7, %alloca[] : memref<i32>
    cf.br ^bb3
  ^bb2:
    memref.store %c9, %alloca[] : memref<i32>
    cf.br ^bb3
  ^bb3:
    memref.store %c11, %alloca[] : memref<i32>
    scf.yield
  }
  %load2 = memref.load %alloca[] : memref<i32>
  return %load2 : i32
}

// -----

// Check promotion through a for loop with a load and store in the body.

// CHECK-LABEL: func.func @for_load_and_store
// CHECK-SAME: (%[[LB:.*]]: index, %[[UB:.*]]: index, %[[STEP:.*]]: index)
// CHECK-DAG: %[[C5:.*]] = arith.constant 5 : i32
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : i32
// CHECK: %[[RES:.*]] = scf.for %{{.*}} = %[[LB]] to %[[UB]] step %[[STEP]] iter_args(%[[ARG:.*]] = %[[C5]]) -> (i32) {
// CHECK:   %[[NEW:.*]] = arith.addi %[[ARG]], %[[C1]] : i32
// CHECK:   scf.yield %[[NEW]] : i32
// CHECK: }
// CHECK: return %[[RES]] : i32
func.func @for_load_and_store(%lb: index, %ub: index, %step: index) -> i32 {
  %c5 = arith.constant 5 : i32
  %c1 = arith.constant 1 : i32
  %alloca = memref.alloca() : memref<i32>
  memref.store %c5, %alloca[] : memref<i32>
  scf.for %i = %lb to %ub step %step {
    %load = memref.load %alloca[] : memref<i32>
    %new = arith.addi %load, %c1 : i32
    memref.store %new, %alloca[] : memref<i32>
    scf.yield
  }
  %load2 = memref.load %alloca[] : memref<i32>
  return %load2 : i32
}

// -----

// Check promotion adds a second iter_arg when one already exists.

// CHECK-LABEL: func.func @for_existing_iter_arg
// CHECK-SAME: (%[[LB:.*]]: index, %[[UB:.*]]: index, %[[STEP:.*]]: index, %[[INIT:.*]]: i32)
// CHECK-DAG: %[[C5:.*]] = arith.constant 5 : i32
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : i32
// CHECK: %[[RES:.*]]:2 = scf.for %{{.*}} = %[[LB]] to %[[UB]] step %[[STEP]] iter_args(%[[MUL_ARG:.*]] = %[[INIT]], %[[SLOT_ARG:.*]] = %[[C5]]) -> (i32, i32) {
// CHECK:   %[[MUL:.*]] = arith.muli %[[MUL_ARG]], %[[MUL_ARG]] : i32
// CHECK:   %[[NEW:.*]] = arith.addi %[[SLOT_ARG]], %[[C1]] : i32
// CHECK:   scf.yield %[[MUL]], %[[NEW]] : i32, i32
// CHECK: }
// CHECK: return %[[RES]]#1 : i32
func.func @for_existing_iter_arg(%lb: index, %ub: index, %step: index, %init: i32) -> i32 {
  %c5 = arith.constant 5 : i32
  %c1 = arith.constant 1 : i32
  %alloca = memref.alloca() : memref<i32>
  memref.store %c5, %alloca[] : memref<i32>
  %mul_res = scf.for %i = %lb to %ub step %step iter_args(%mul_arg = %init) -> i32 {
    %mul = arith.muli %mul_arg, %mul_arg : i32
    %load = memref.load %alloca[] : memref<i32>
    %new = arith.addi %load, %c1 : i32
    memref.store %new, %alloca[] : memref<i32>
    scf.yield %mul : i32
  }
  %load2 = memref.load %alloca[] : memref<i32>
  return %load2 : i32
}

// -----

// Check load-only promotion through a for loop generates no iter_arg.

func.func private @use(i32)

// CHECK-LABEL: func.func @for_load_only
// CHECK-SAME: (%[[LB:.*]]: index, %[[UB:.*]]: index, %[[STEP:.*]]: index)
// CHECK: %[[C5:.*]] = arith.constant 5 : i32
// CHECK: scf.for %{{.*}} = %[[LB]] to %[[UB]] step %[[STEP]] {
// CHECK:   call @use(%[[C5]])
// CHECK: }
// CHECK: return %[[C5]] : i32
func.func @for_load_only(%lb: index, %ub: index, %step: index) -> i32 {
  %c5 = arith.constant 5 : i32
  %alloca = memref.alloca() : memref<i32>
  memref.store %c5, %alloca[] : memref<i32>
  scf.for %i = %lb to %ub step %step {
    %load = memref.load %alloca[] : memref<i32>
    func.call @use(%load) : (i32) -> ()
    scf.yield
  }
  %load2 = memref.load %alloca[] : memref<i32>
  return %load2 : i32
}

// -----

// Check promotion through a for loop with a store inside an if in the body.

// CHECK-LABEL: func.func @for_if_store
// CHECK-SAME: (%[[LB:.*]]: index, %[[UB:.*]]: index, %[[STEP:.*]]: index, %[[COND:.*]]: i1)
// CHECK-DAG: %[[C5:.*]] = arith.constant 5 : i32
// CHECK-DAG: %[[C7:.*]] = arith.constant 7 : i32
// CHECK: %[[RES:.*]] = scf.for %{{.*}} = %[[LB]] to %[[UB]] step %[[STEP]] iter_args(%[[ARG:.*]] = %[[C5]]) -> (i32) {
// CHECK:   %[[IF:.*]] = scf.if %[[COND]] -> (i32) {
// CHECK:     scf.yield %[[C7]] : i32
// CHECK:   } else {
// CHECK:     scf.yield %[[ARG]] : i32
// CHECK:   }
// CHECK:   scf.yield %[[IF]] : i32
// CHECK: }
// CHECK: return %[[RES]] : i32
func.func @for_if_store(%lb: index, %ub: index, %step: index, %cond: i1) -> i32 {
  %c5 = arith.constant 5 : i32
  %c7 = arith.constant 7 : i32
  %alloca = memref.alloca() : memref<i32>
  memref.store %c5, %alloca[] : memref<i32>
  scf.for %i = %lb to %ub step %step {
    scf.if %cond {
      memref.store %c7, %alloca[] : memref<i32>
      scf.yield
    }
    scf.yield
  }
  %load = memref.load %alloca[] : memref<i32>
  return %load : i32
}

// -----

// Check load promotion through a forall.

func.func private @use(i32)

// CHECK-LABEL: func.func @forall_load
// CHECK-SAME: (%[[UB:.*]]: index)
// CHECK: %[[C5:.*]] = arith.constant 5 : i32
// CHECK: scf.forall (%{{.*}}) in (%[[UB]]) {
// CHECK:   call @use(%[[C5]])
// CHECK: }
func.func @forall_load(%ub: index) {
  %c5 = arith.constant 5 : i32
  %alloca = memref.alloca() : memref<i32>
  memref.store %c5, %alloca[] : memref<i32>
  scf.forall (%i) in (%ub) {
    %load = memref.load %alloca[] : memref<i32>
    func.call @use(%load) : (i32) -> ()
  }
  return
}

// -----

// Check promotion through a forall nested inside an if with a store.

func.func private @use(i32)

// CHECK-LABEL: func.func @forall_in_if
// CHECK-SAME: (%[[UB:.*]]: index, %[[COND:.*]]: i1)
// CHECK-DAG: %[[C5:.*]] = arith.constant 5 : i32
// CHECK-DAG: %[[C7:.*]] = arith.constant 7 : i32
// CHECK: %[[RES:.*]] = scf.if %[[COND]] -> (i32) {
// CHECK:   scf.forall (%{{.*}}) in (%[[UB]]) {
// CHECK:     call @use(%[[C7]])
// CHECK:   }
// CHECK:   scf.yield %[[C7]] : i32
// CHECK: } else {
// CHECK:   scf.yield %[[C5]] : i32
// CHECK: }
// CHECK: return %[[RES]] : i32
func.func @forall_in_if(%ub: index, %cond: i1) -> i32 {
  %c5 = arith.constant 5 : i32
  %c7 = arith.constant 7 : i32
  %alloca = memref.alloca() : memref<i32>
  memref.store %c5, %alloca[] : memref<i32>
  scf.if %cond {
    memref.store %c7, %alloca[] : memref<i32>
    scf.forall (%i) in (%ub) {
      %load = memref.load %alloca[] : memref<i32>
      func.call @use(%load) : (i32) -> ()
    }
    scf.yield
  }
  %load2 = memref.load %alloca[] : memref<i32>
  return %load2 : i32
}

// -----

// Check load promotion through a parallel.

func.func private @use(i32)

// CHECK-LABEL: func.func @parallel_load
// CHECK-SAME: (%[[LB:.*]]: index, %[[UB:.*]]: index, %[[STEP:.*]]: index)
// CHECK: %[[C5:.*]] = arith.constant 5 : i32
// CHECK: scf.parallel (%{{.*}}) = (%[[LB]]) to (%[[UB]]) step (%[[STEP]]) {
// CHECK:   call @use(%[[C5]])
// CHECK:   scf.reduce
// CHECK: }
func.func @parallel_load(%lb: index, %ub: index, %step: index) {
  %c5 = arith.constant 5 : i32
  %alloca = memref.alloca() : memref<i32>
  memref.store %c5, %alloca[] : memref<i32>
  scf.parallel (%i) = (%lb) to (%ub) step (%step) {
    %load = memref.load %alloca[] : memref<i32>
    func.call @use(%load) : (i32) -> ()
    scf.reduce
  }
  return
}

// -----

// Check promotion through a parallel nested inside an if with a store.

func.func private @use(i32)

// CHECK-LABEL: func.func @parallel_in_if
// CHECK-SAME: (%[[LB:.*]]: index, %[[UB:.*]]: index, %[[STEP:.*]]: index, %[[COND:.*]]: i1)
// CHECK-DAG: %[[C5:.*]] = arith.constant 5 : i32
// CHECK-DAG: %[[C7:.*]] = arith.constant 7 : i32
// CHECK: %[[RES:.*]] = scf.if %[[COND]] -> (i32) {
// CHECK:   scf.parallel (%{{.*}}) = (%[[LB]]) to (%[[UB]]) step (%[[STEP]]) {
// CHECK:     call @use(%[[C7]])
// CHECK:     scf.reduce
// CHECK:   }
// CHECK:   scf.yield %[[C7]] : i32
// CHECK: } else {
// CHECK:   scf.yield %[[C5]] : i32
// CHECK: }
// CHECK: return %[[RES]] : i32
func.func @parallel_in_if(%lb: index, %ub: index, %step: index, %cond: i1) -> i32 {
  %c5 = arith.constant 5 : i32
  %c7 = arith.constant 7 : i32
  %alloca = memref.alloca() : memref<i32>
  memref.store %c5, %alloca[] : memref<i32>
  scf.if %cond {
    memref.store %c7, %alloca[] : memref<i32>
    scf.parallel (%i) = (%lb) to (%ub) step (%step) {
      %load = memref.load %alloca[] : memref<i32>
      func.call @use(%load) : (i32) -> ()
      scf.reduce
    }
    scf.yield
  }
  %load2 = memref.load %alloca[] : memref<i32>
  return %load2 : i32
}

// -----

// Check load promotion inside a reduce region.

// CHECK-LABEL: func.func @parallel_reduce_load
// CHECK-SAME: (%[[LB:.*]]: index, %[[UB:.*]]: index, %[[STEP:.*]]: index)
// CHECK-DAG: %[[C5:.*]] = arith.constant 5 : i32
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : i32
// CHECK: %[[RES:.*]] = scf.parallel (%{{.*}}) = (%[[LB]]) to (%[[UB]]) step (%[[STEP]]) init (%[[C0]]) -> i32 {
// CHECK:   %[[C1:.*]] = arith.constant 1 : i32
// CHECK:   scf.reduce(%[[C1]] : i32) {
// CHECK:   ^{{.*}}(%[[LHS:.*]]: i32, %[[RHS:.*]]: i32):
// CHECK:     %[[SUM:.*]] = arith.addi %[[LHS]], %[[RHS]] : i32
// CHECK:     %[[MUL:.*]] = arith.muli %[[SUM]], %[[C5]] : i32
// CHECK:     scf.reduce.return %[[MUL]] : i32
// CHECK:   }
// CHECK: }
// CHECK: return %[[RES]] : i32
func.func @parallel_reduce_load(%lb: index, %ub: index, %step: index) -> i32 {
  %c5 = arith.constant 5 : i32
  %c0 = arith.constant 0 : i32
  %alloca = memref.alloca() : memref<i32>
  memref.store %c5, %alloca[] : memref<i32>
  %res = scf.parallel (%i) = (%lb) to (%ub) step (%step) init (%c0) -> i32 {
    %c1 = arith.constant 1 : i32
    scf.reduce(%c1 : i32) {
    ^bb0(%lhs: i32, %rhs: i32):
      %sum = arith.addi %lhs, %rhs : i32
      %load = memref.load %alloca[] : memref<i32>
      %mul = arith.muli %sum, %load : i32
      scf.reduce.return %mul : i32
    }
  }
  return %res : i32
}

// -----

// Check load promotion in the before region of a while.

func.func private @use(i32)

// CHECK-LABEL: func.func @while_load_before
// CHECK-SAME: (%[[COND:.*]]: i1)
// CHECK: %[[C5:.*]] = arith.constant 5 : i32
// CHECK: scf.while : () -> () {
// CHECK:   call @use(%[[C5]])
// CHECK:   scf.condition(%[[COND]])
// CHECK: } do {
// CHECK:   scf.yield
// CHECK: }
func.func @while_load_before(%cond: i1) {
  %c5 = arith.constant 5 : i32
  %alloca = memref.alloca() : memref<i32>
  memref.store %c5, %alloca[] : memref<i32>
  scf.while : () -> () {
    %load = memref.load %alloca[] : memref<i32>
    func.call @use(%load) : (i32) -> ()
    scf.condition(%cond)
  } do {
    scf.yield
  }
  return
}

// -----

// Check load promotion in the after region of a while.

func.func private @use(i32)

// CHECK-LABEL: func.func @while_load_after
// CHECK-SAME: (%[[COND:.*]]: i1)
// CHECK: %[[C5:.*]] = arith.constant 5 : i32
// CHECK: scf.while : () -> () {
// CHECK:   scf.condition(%[[COND]])
// CHECK: } do {
// CHECK:   call @use(%[[C5]])
// CHECK:   scf.yield
// CHECK: }
func.func @while_load_after(%cond: i1) {
  %c5 = arith.constant 5 : i32
  %alloca = memref.alloca() : memref<i32>
  memref.store %c5, %alloca[] : memref<i32>
  scf.while : () -> () {
    scf.condition(%cond)
  } do {
    %load = memref.load %alloca[] : memref<i32>
    func.call @use(%load) : (i32) -> ()
    scf.yield
  }
  return
}

// -----

// Check promotion with a store in the before region and a load in the after.

func.func private @use(i32)

// CHECK-LABEL: func.func @while_store_before_load_after
// CHECK-SAME: (%[[COND:.*]]: i1)
// CHECK-DAG: %[[C5:.*]] = arith.constant 5 : i32
// CHECK-DAG: %[[C7:.*]] = arith.constant 7 : i32
// CHECK: scf.while (%[[BEFORE:.*]] = %[[C5]]) : (i32) -> i32 {
// CHECK:   scf.condition(%[[COND]]) %[[C7]] : i32
// CHECK: } do {
// CHECK: ^{{.*}}(%[[AFTER:.*]]: i32):
// CHECK:   call @use(%[[AFTER]])
// CHECK:   scf.yield %[[AFTER]] : i32
// CHECK: }
func.func @while_store_before_load_after(%cond: i1) {
  %c5 = arith.constant 5 : i32
  %c7 = arith.constant 7 : i32
  %alloca = memref.alloca() : memref<i32>
  memref.store %c5, %alloca[] : memref<i32>
  scf.while : () -> () {
    memref.store %c7, %alloca[] : memref<i32>
    scf.condition(%cond)
  } do {
    %load = memref.load %alloca[] : memref<i32>
    func.call @use(%load) : (i32) -> ()
    scf.yield
  }
  return
}

// -----

// Check promotion with a store in the before region and a load after the loop.

// CHECK-LABEL: func.func @while_store_before_load_after_loop
// CHECK-SAME: (%[[COND:.*]]: i1)
// CHECK-DAG: %[[C5:.*]] = arith.constant 5 : i32
// CHECK-DAG: %[[C7:.*]] = arith.constant 7 : i32
// CHECK: %[[RES:.*]] = scf.while (%[[BEFORE:.*]] = %[[C5]]) : (i32) -> i32 {
// CHECK:   scf.condition(%[[COND]]) %[[C7]] : i32
// CHECK: } do {
// CHECK: ^{{.*}}(%[[AFTER:.*]]: i32):
// CHECK:   scf.yield %[[AFTER]] : i32
// CHECK: }
// CHECK: return %[[RES]] : i32
func.func @while_store_before_load_after_loop(%cond: i1) -> i32 {
  %c5 = arith.constant 5 : i32
  %c7 = arith.constant 7 : i32
  %alloca = memref.alloca() : memref<i32>
  memref.store %c5, %alloca[] : memref<i32>
  scf.while : () -> () {
    memref.store %c7, %alloca[] : memref<i32>
    scf.condition(%cond)
  } do {
    scf.yield
  }
  %res = memref.load %alloca[] : memref<i32>
  return %res : i32
}

// -----

// Check store promotion through a while implementing a for loop from 0 to 10.

// CHECK-LABEL: func.func @while_store
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : i32
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : i32
// CHECK-DAG: %[[C10:.*]] = arith.constant 10 : i32
// CHECK: %[[RES:.*]] = scf.while (%[[BEFORE:.*]] = %[[C0]]) : (i32) -> i32 {
// CHECK:   %[[COND:.*]] = arith.cmpi slt, %[[BEFORE]], %[[C10]] : i32
// CHECK:   scf.condition(%[[COND]]) %[[BEFORE]] : i32
// CHECK: } do {
// CHECK: ^{{.*}}(%[[AFTER:.*]]: i32):
// CHECK:   %[[NEW:.*]] = arith.addi %[[AFTER]], %[[C1]] : i32
// CHECK:   scf.yield %[[NEW]] : i32
// CHECK: }
// CHECK: return %[[RES]] : i32
func.func @while_store() -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c10 = arith.constant 10 : i32
  %alloca = memref.alloca() : memref<i32>
  memref.store %c0, %alloca[] : memref<i32>
  scf.while : () -> () {
    %val = memref.load %alloca[] : memref<i32>
    %cond = arith.cmpi slt, %val, %c10 : i32
    scf.condition(%cond)
  } do {
    %val = memref.load %alloca[] : memref<i32>
    %new = arith.addi %val, %c1 : i32
    memref.store %new, %alloca[] : memref<i32>
    scf.yield
  }
  %res = memref.load %alloca[] : memref<i32>
  return %res : i32
}

// -----

// Check load promotion through an index_switch default branch.

func.func private @use(i32)

// CHECK-LABEL: func.func @index_switch_load_default
// CHECK-SAME: (%[[IDX:.*]]: index)
// CHECK: %[[C5:.*]] = arith.constant 5 : i32
// CHECK: scf.index_switch %[[IDX]]
// CHECK: default {
// CHECK:   call @use(%[[C5]])
// CHECK: }
func.func @index_switch_load_default(%idx: index) {
  %c5 = arith.constant 5 : i32
  %alloca = memref.alloca() : memref<i32>
  memref.store %c5, %alloca[] : memref<i32>
  scf.index_switch %idx
  default {
    %load = memref.load %alloca[] : memref<i32>
    func.call @use(%load) : (i32) -> ()
    scf.yield
  }
  return
}

// -----

// Check store promotion through an index_switch default branch.

// CHECK-LABEL: func.func @index_switch_store_default
// CHECK-SAME: (%[[IDX:.*]]: index)
// CHECK-DAG: %[[C5:.*]] = arith.constant 5 : i32
// CHECK-DAG: %[[C7:.*]] = arith.constant 7 : i32
// CHECK: %[[RES:.*]] = scf.index_switch %[[IDX]] -> i32
// CHECK: default {
// CHECK:   scf.yield %[[C7]] : i32
// CHECK: }
// CHECK: return %[[RES]] : i32
func.func @index_switch_store_default(%idx: index) -> i32 {
  %c5 = arith.constant 5 : i32
  %c7 = arith.constant 7 : i32
  %alloca = memref.alloca() : memref<i32>
  memref.store %c5, %alloca[] : memref<i32>
  scf.index_switch %idx
  default {
    memref.store %c7, %alloca[] : memref<i32>
    scf.yield
  }
  %load = memref.load %alloca[] : memref<i32>
  return %load : i32
}

// -----

// Check promotion with a store in a case and a load in the default branch.

func.func private @use(i32)

// CHECK-LABEL: func.func @index_switch_store_case_load_default
// CHECK-SAME: (%[[IDX:.*]]: index)
// CHECK-DAG: %[[C5:.*]] = arith.constant 5 : i32
// CHECK-DAG: %[[C7:.*]] = arith.constant 7 : i32
// CHECK: %[[RES:.*]] = scf.index_switch %[[IDX]] -> i32
// CHECK: case 0 {
// CHECK:   scf.yield %[[C7]] : i32
// CHECK: }
// CHECK: default {
// CHECK:   call @use(%[[C5]])
// CHECK:   scf.yield %[[C5]] : i32
// CHECK: }
// CHECK: return %[[RES]] : i32
func.func @index_switch_store_case_load_default(%idx: index) -> i32 {
  %c5 = arith.constant 5 : i32
  %c7 = arith.constant 7 : i32
  %alloca = memref.alloca() : memref<i32>
  memref.store %c5, %alloca[] : memref<i32>
  scf.index_switch %idx
  case 0 {
    memref.store %c7, %alloca[] : memref<i32>
    scf.yield
  }
  default {
    %load = memref.load %alloca[] : memref<i32>
    func.call @use(%load) : (i32) -> ()
    scf.yield
  }
  %load2 = memref.load %alloca[] : memref<i32>
  return %load2 : i32
}
