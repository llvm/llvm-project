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

// CHECK-LABEL: func.func @execute_region_load
// CHECK: %[[C5:.*]] = arith.constant 5 : i32
// CHECK: %[[RES:.*]] = scf.execute_region -> i32 {
// CHECK:   scf.yield %[[C5]] : i32
// CHECK: }
// CHECK: return %[[RES]] : i32
func.func @execute_region_load() -> i32 {
  %c5 = arith.constant 5 : i32
  %alloca = memref.alloca() : memref<i32>
  memref.store %c5, %alloca[] : memref<i32>
  %res = scf.execute_region -> i32 {
    %load = memref.load %alloca[] : memref<i32>
    scf.yield %load : i32
  }
  return %res : i32
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
