// RUN: mlir-opt -pass-pipeline="builtin.module(any(test-cfg-loop-info))" --split-input-file %s 2>&1 | FileCheck %s

// CHECK-LABEL: Testing : "no_loop_single_block"
// CHECK: no loops
func.func @no_loop_single_block() {
  return
}

// -----

// CHECK-LABEL: Testing : "no_loop"
// CHECK: no loops
func.func @no_loop() {
  cf.br ^bb1
^bb1:
  return
}

// -----

// CHECK-LABEL: Testing : "simple_loop"
// CHECK-NEXT: Blocks : ^[[BB0:.*]], ^[[BB1:.*]], ^[[BB2:.*]], ^[[BB3:.*]]
// CHECK: Loop at depth 1 containing:
// CHECK-SAME: ^[[BB1]]<header><exiting>
// CHECK-SAME: ^[[BB2]]<latch>
func.func @simple_loop(%c: i1) {
  cf.br ^bb1
^bb1:
  cf.cond_br %c, ^bb2, ^bb3
^bb2:
  cf.br ^bb1
^bb3:
  return
}

// -----

// CHECK-LABEL: Testing : "single_block_loop"
// CHECK-NEXT: Blocks : ^[[BB0:.*]], ^[[BB1:.*]], ^[[BB2:.*]]
// CHECK: Loop at depth 1 containing:
// CHECK-SAME: ^[[BB1]]<header><latch><exiting>
func.func @single_block_loop(%c: i1) {
  cf.br ^bb1
^bb1:
  cf.cond_br %c, ^bb1, ^bb2
^bb2:
  return
}

// -----

// CHECK-LABEL: Testing : "nested_loop"
// CHECK-NEXT: Blocks : ^[[BB0:.*]], ^[[BB1:.*]], ^[[BB2:.*]], ^[[BB3:.*]], ^[[BB4:.*]]
// CHECK: Loop at depth 1
// CHECK-SAME: ^[[BB1]]<header><exiting>
// CHECK-SAME: ^[[BB2]]<latch>
// CHECK-SAME: ^[[BB3]]
// CHECK: Loop at depth 2
// CHECK-SAME: ^[[BB2]]<header><exiting>
// CHECK-SAME: ^[[BB3]]<latch>
func.func @nested_loop(%c: i1) {
  cf.br ^bb1
^bb1:
  cf.cond_br %c, ^bb2, ^bb4
^bb2:
  cf.cond_br %c, ^bb1, ^bb3
^bb3:
  cf.br ^bb2
^bb4:
  return
}

// -----

// CHECK-LABEL: Testing : "multi_latch"
// CHECK-NEXT: Blocks : ^[[BB0:.*]], ^[[BB1:.*]], ^[[BB2:.*]], ^[[BB3:.*]], ^[[BB4:.*]]
// CHECK: Loop at depth 1
// CHECK-SAME: ^[[BB1]]<header><exiting>
// CHECK-SAME: ^[[BB2]]<latch>
// CHECK-SAME: ^[[BB3]]<latch>
func.func @multi_latch(%c: i1) {
  cf.br ^bb1
^bb1:
  cf.cond_br %c, ^bb4, ^bb2
^bb2:
  cf.cond_br %c, ^bb1, ^bb3
^bb3:
  cf.br ^bb1
^bb4:
  return
}

// -----

// CHECK-LABEL: Testing : "multiple_loops"
// CHECK-NEXT: Blocks : ^[[BB0:.*]], ^[[BB1:.*]], ^[[BB2:.*]], ^[[BB3:.*]], ^[[BB4:.*]], ^[[BB5:.*]]
// CHECK: Loop at depth 1
// CHECK-SAME: ^[[BB3]]<header><exiting>
// CHECK-SAME: ^[[BB4]]<latch>
// CHECK: Loop at depth 1
// CHECK-SAME: ^[[BB1]]<header>
// CHECK-SAME: ^[[BB2]]<latch><exiting>
func.func @multiple_loops(%c: i1) {
  cf.br ^bb1
^bb1:
  cf.br ^bb2
^bb2:
  cf.cond_br %c, ^bb3, ^bb1
^bb3:
  cf.cond_br %c, ^bb5, ^bb4
^bb4:
  cf.br ^bb3
^bb5:
  return
}
