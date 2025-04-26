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
func.func @loop_with_results(%value : f32) -> f32 {
 %result = scf.loop token(%loop) -> f32 {
   scf.break [%loop] %value : f32
 }
 return %result : f32
}

// -----

// CHECK-LABEL: func @loop_continue_iterargs(
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
