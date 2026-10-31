// RUN: mlir-opt --print-region-branch-op-interface %s --split-input-file | FileCheck %s
// RUN: mlir-opt %s --mlir-print-debuginfo --mlir-print-op-generic --split-input-file  | mlir-opt --print-region-branch-op-interface --split-input-file  | FileCheck %s


func.func @loop_break(%cond : i1) {
  // CHECK: Found RegionBranchOpInterface operation: scf.loop {{.*}} {...} loc("loop1")
  // CHECK:  - Successor is region #0
  // CHECK:  - Found 2 predecessor(s)
  // CHECK:    - Predecessor is scf.break {{.*}} loc("break1")
  // CHECK:    - Predecessor is scf.continue
   scf.loop token(%loop) {
     scf.if %cond {
       scf.break [%loop] loc("break1")
     }
   } loc("loop1")
   return
}

// -----

func.func @loop_continue(%cond1 : i1, %cond2 : i1) {
  // CHECK: Found RegionBranchOpInterface operation: scf.loop {{.*}} {...} loc("loop2")
  // CHECK:  - Successor is region #0
  // CHECK:  - Found 2 predecessor(s)
  // CHECK:    - Predecessor is scf.break {{.*}} loc("break2")
  // CHECK:    - Predecessor is scf.continue
   scf.loop token(%outer) {
    // CHECK: Found RegionBranchOpInterface operation: scf.loop {{.*}} {...} loc("loop3")
    // CHECK:  - Successor is region #0
    // CHECK:  - Found 2 predecessor(s)
    // CHECK:    - Predecessor is scf.continue {{.*}} loc("continue1")
    // CHECK:    - Predecessor is scf.continue
     scf.loop token(%inner) {
       scf.if %cond1 {
         scf.continue [%inner] loc("continue1")
       }
       scf.if %cond2 {
         scf.break [%outer] loc("break2")
       }
     } loc("loop3")
   } loc("loop2")
   return
}

// -----

// CHECK-LABEL: func @loop_with_results(
// CHECK-SAME: %[[VALUE:.*]]: f32
// CHECK: %[[RESULT:.*]] = scf.loop token(%[[TOKEN:.*]]) -> f32 {
// CHECK: scf.break [%[[TOKEN]]] %[[VALUE]] : f32
// CHECK: return %[[RESULT]] : f32
func.func @loop_with_results(%value : f32) -> f32 {
 %result = scf.loop token(%loop) -> f32 {
   scf.break [%loop] %value : f32
 }
 return %result : f32
}

// -----

// CHECK-LABEL: func @loop_continue_iterargs(
// CHECK-SAME: %[[INIT:.*]]: i32
// CHECK: scf.loop token(%[[TOKEN:.*]]) iter_args(%[[ARG:.*]] = %[[INIT]]) : i32 {
// CHECK: scf.continue [%[[TOKEN]]] %[[ARG]] : i32
func.func @loop_continue_iterargs(%init : i32) {
 scf.loop token(%loop) iter_args(%next = %init) : i32 {
   scf.continue [%loop] %next : i32
 }
 return
}

// -----

// A single-operand continue that targets an *outer* loop must be printed
// explicitly. If the printer elided it as the trivial implicit terminator, the
// parser would rebuild it as a continue of the inner loop, silently changing
// the target.
// CHECK-LABEL: func @continue_outer_from_inner
func.func @continue_outer_from_inner() {
  // CHECK: scf.loop token(%[[OUTER:.*]]) {
  scf.loop token(%outer) {
    // CHECK: scf.loop token(%[[INNER:.*]]) {
    scf.loop token(%inner) {
      // CHECK: scf.continue [%[[OUTER]]]
      scf.continue [%outer]
    }
  }
  return
}

// -----

// CHECK-LABEL: func @iter_args_and_results(
// CHECK-SAME: %[[INIT:.*]]: i32, %[[COND:.*]]: i1
// CHECK: %[[RESULT:.*]] = scf.loop token(%[[TOKEN:.*]]) iter_args(%[[ARG:.*]] = %[[INIT]]) : i32 -> i32 {
// CHECK: scf.if %[[COND]] {
// CHECK: scf.break [%[[TOKEN]]] %[[ARG]] : i32
// CHECK: scf.continue [%[[TOKEN]]] %[[ARG]] : i32
// CHECK: return %[[RESULT]] : i32
func.func @iter_args_and_results(%init: i32, %cond: i1) -> i32 {
  %r = scf.loop token(%token) iter_args(%arg = %init) : i32 -> i32 {
    scf.if %cond {
      scf.break [%token] %arg : i32
    }
    scf.continue [%token] %arg : i32
  }
  return %r : i32
}

// -----

// Dialect-defined propagators may host SCF loop exits.
// CHECK-LABEL: func @break_through_dialect_op(
// CHECK: scf.loop token(%[[TOKEN:.*]]) {
// CHECK: test.propagate_control_flow_break
// CHECK: scf.break [%[[TOKEN]]]
func.func @break_through_dialect_op() {
  scf.loop token(%token) {
    test.propagate_control_flow_break {
      scf.break [%token]
    }
    scf.break [%token]
  }
  return
}
