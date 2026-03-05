// RUN: mlir-opt --print-region-branch-op-interface %s --split-input-file | FileCheck %s
// RUN: mlir-opt %s --mlir-print-debuginfo --mlir-print-op-generic --split-input-file  | mlir-opt --print-region-branch-op-interface --split-input-file  | FileCheck %s


// CHECK-LABEL: func @unregistered_op
func.func @unregistered_op(%cond : i1) {
   "test.some_loop"() ({
      "test.some_if"(%cond) ({
        "test.some_break"() [2] : () -> ()
      }) : (i1) -> ()
      "test.continue"() [1] : () -> ()
    }) : () -> ()   return
}


// -----

func.func @loop_break(%cond : i1) {
  // CHECK: Found RegionBranchOpInterface operation: scf.loop {...} loc("loop1")
  // CHECK:  - Successor is region #0
  // CHECK:  - Found 2 predecessor(s)
  // CHECK:    - Predecessor is scf.break 2 loc("break1")
  // CHECK:    - Predecessor is scf.continue 1
   scf.loop {
     scf.if %cond {
       scf.break 2 loc("break1")
     }
   } loc("loop1")
   return
}

// -----

func.func @loop_continue(%cond1 : i1, %cond2 : i1) {
  // CHECK: Found RegionBranchOpInterface operation: scf.loop {...} loc("loop2")
  // CHECK:  - Successor is region #0
  // CHECK:  - Found 2 predecessor(s)
  // CHECK:    - Predecessor is scf.break 3 loc("break2")
  // CHECK:    - Predecessor is scf.continue 1
   scf.loop {
    // CHECK: Found RegionBranchOpInterface operation: scf.loop {...} loc("loop3")
    // CHECK:  - Successor is region #0
    // CHECK:  - Found 2 predecessor(s)
    // CHECK:    - Predecessor is scf.continue 2 loc("continue1")
    // CHECK:    - Predecessor is scf.continue 1
     scf.loop {
       scf.if %cond1 {
         scf.continue 2 loc("continue1")
       }
       scf.if %cond2 {
         scf.break 3 loc("break2")
       }
     } loc("loop3")
   } loc("loop2")
   return
}

// -----

// CHECK-LABEL: func @loop_with_results(
func.func @loop_with_results(%value : f32) -> f32 {
 %result = scf.loop -> f32 {
   scf.break 1 %value : f32
 }
 return %result : f32
}

// -----

// CHECK-LABEL: func @loop_continue_iterargs(
func.func @loop_continue_iterargs(%init : i32) {
 scf.loop iter_args(%next = %init) : i32 {
   scf.continue 1 %next : i32
 }
 return
}

