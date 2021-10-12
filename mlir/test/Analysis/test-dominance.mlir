// RUN: mlir-opt %s -test-print-dominance -split-input-file 2>&1 | FileCheck %s

// CHECK-LABEL: Testing : func_condBranch
func @func_condBranch(%cond : i1) {
  cond_br %cond, ^bb1, ^bb2
^bb1:
  br ^exit
^bb2:
  br ^exit
^exit:
  return
}
// CHECK-LABEL: --- DominanceInfo ---
// CHECK-NEXT: Nearest(0, 0) = 0
// CHECK-NEXT: Nearest(0, 1) = 0
// CHECK-NEXT: Nearest(0, 2) = 0
// CHECK-NEXT: Nearest(0, 3) = 0
// CHECK: Nearest(1, 0) = 0
// CHECK-NEXT: Nearest(1, 1) = 1
// CHECK-NEXT: Nearest(1, 2) = 0
// CHECK-NEXT: Nearest(1, 3) = 0
// CHECK: Nearest(2, 0) = 0
// CHECK-NEXT: Nearest(2, 1) = 0
// CHECK-NEXT: Nearest(2, 2) = 2
// CHECK-NEXT: Nearest(2, 3) = 0
// CHECK: Nearest(3, 0) = 0
// CHECK-NEXT: Nearest(3, 1) = 0
// CHECK-NEXT: Nearest(3, 2) = 0
// CHECK-NEXT: Nearest(3, 3) = 3
// CHECK-LABEL: --- PostDominanceInfo ---
// CHECK-NEXT: Nearest(0, 0) = 0
// CHECK-NEXT: Nearest(0, 1) = 3
// CHECK-NEXT: Nearest(0, 2) = 3
// CHECK-NEXT: Nearest(0, 3) = 3
// CHECK: Nearest(1, 0) = 3
// CHECK-NEXT: Nearest(1, 1) = 1
// CHECK-NEXT: Nearest(1, 2) = 3
// CHECK-NEXT: Nearest(1, 3) = 3
// CHECK: Nearest(2, 0) = 3
// CHECK-NEXT: Nearest(2, 1) = 3
// CHECK-NEXT: Nearest(2, 2) = 2
// CHECK-NEXT: Nearest(2, 3) = 3
// CHECK: Nearest(3, 0) = 3
// CHECK-NEXT: Nearest(3, 1) = 3
// CHECK-NEXT: Nearest(3, 2) = 3
// CHECK-NEXT: Nearest(3, 3) = 3

// -----

// CHECK-LABEL: Testing : func_loop
func @func_loop(%arg0 : i32, %arg1 : i32) {
  br ^loopHeader(%arg0 : i32)
^loopHeader(%counter : i32):
  %lessThan = arith.cmpi slt, %counter, %arg1 : i32
  cond_br %lessThan, ^loopBody, ^exit
^loopBody:
  %const0 = arith.constant 1 : i32
  %inc = arith.addi %counter, %const0 : i32
  br ^loopHeader(%inc : i32)
^exit:
  return
}
// CHECK-LABEL: --- DominanceInfo ---
// CHECK: Nearest(1, 0) = 0
// CHECK-NEXT: Nearest(1, 1) = 1
// CHECK-NEXT: Nearest(1, 2) = 1
// CHECK-NEXT: Nearest(1, 3) = 1
// CHECK: Nearest(2, 0) = 0
// CHECK-NEXT: Nearest(2, 1) = 1
// CHECK-NEXT: Nearest(2, 2) = 2
// CHECK-NEXT: Nearest(2, 3) = 1
// CHECK: Nearest(3, 0) = 0
// CHECK-NEXT: Nearest(3, 1) = 1
// CHECK-NEXT: Nearest(3, 2) = 1
// CHECK-NEXT: Nearest(3, 3) = 3
// CHECK-LABEL: --- PostDominanceInfo ---
// CHECK: Nearest(1, 0) = 1
// CHECK-NEXT: Nearest(1, 1) = 1
// CHECK-NEXT: Nearest(1, 2) = 1
// CHECK-NEXT: Nearest(1, 3) = 3
// CHECK: Nearest(2, 0) = 1
// CHECK-NEXT: Nearest(2, 1) = 1
// CHECK-NEXT: Nearest(2, 2) = 2
// CHECK-NEXT: Nearest(2, 3) = 3
// CHECK: Nearest(3, 0) = 3
// CHECK-NEXT: Nearest(3, 1) = 3
// CHECK-NEXT: Nearest(3, 2) = 3
// CHECK-NEXT: Nearest(3, 3) = 3

// -----

// CHECK-LABEL: Testing : nested_region
func @nested_region(%arg0 : index, %arg1 : index, %arg2 : index) {
  scf.for %arg3 = %arg0 to %arg1 step %arg2 { }
  return
}

// CHECK-LABEL: --- DominanceInfo ---
// CHECK-NEXT: Nearest(0, 0) = 0
// CHECK-NEXT: Nearest(0, 1) = 1
// CHECK: Nearest(1, 0) = 1
// CHECK-NEXT: Nearest(1, 1) = 1
// CHECK-LABEL: --- PostDominanceInfo ---
// CHECK-NEXT: Nearest(0, 0) = 0
// CHECK-NEXT: Nearest(0, 1) = 1
// CHECK: Nearest(1, 0) = 1
// CHECK-NEXT: Nearest(1, 1) = 1

// -----

// CHECK-LABEL: Testing : nested_region2
func @nested_region2(%arg0 : index, %arg1 : index, %arg2 : index) {
  scf.for %arg3 = %arg0 to %arg1 step %arg2 {
    scf.for %arg4 = %arg0 to %arg1 step %arg2 {
      scf.for %arg5 = %arg0 to %arg1 step %arg2 { }
    }
  }
  return
}
// CHECK-LABEL: --- DominanceInfo ---
// CHECK: Nearest(1, 0) = 1
// CHECK-NEXT: Nearest(1, 1) = 1
// CHECK-NEXT: Nearest(1, 2) = 2
// CHECK-NEXT: Nearest(1, 3) = 3
// CHECK: Nearest(2, 0) = 2
// CHECK-NEXT: Nearest(2, 1) = 2
// CHECK-NEXT: Nearest(2, 2) = 2
// CHECK-NEXT: Nearest(2, 3) = 3
// CHECK: Nearest(3, 0) = 3
// CHECK-NEXT: Nearest(3, 1) = 3
// CHECK-NEXT: Nearest(3, 2) = 3
// CHECK-NEXT: Nearest(3, 3) = 3
// CHECK-LABEL: --- PostDominanceInfo ---
// CHECK-NEXT: Nearest(0, 0) = 0
// CHECK-NEXT: Nearest(0, 1) = 1
// CHECK-NEXT: Nearest(0, 2) = 2
// CHECK-NEXT: Nearest(0, 3) = 3
// CHECK: Nearest(1, 0) = 1
// CHECK-NEXT: Nearest(1, 1) = 1
// CHECK-NEXT: Nearest(1, 2) = 2
// CHECK-NEXT: Nearest(1, 3) = 3
// CHECK: Nearest(2, 0) = 2
// CHECK-NEXT: Nearest(2, 1) = 2
// CHECK-NEXT: Nearest(2, 2) = 2
// CHECK-NEXT: Nearest(2, 3) = 3

// -----

// CHECK-LABEL: Testing : func_loop_nested_region
func @func_loop_nested_region(
  %arg0 : i32,
  %arg1 : i32,
  %arg2 : index,
  %arg3 : index,
  %arg4 : index) {
  br ^loopHeader(%arg0 : i32)
^loopHeader(%counter : i32):
  %lessThan = arith.cmpi slt, %counter, %arg1 : i32
  cond_br %lessThan, ^loopBody, ^exit
^loopBody:
  %const0 = arith.constant 1 : i32
  %inc = arith.addi %counter, %const0 : i32
  scf.for %arg5 = %arg2 to %arg3 step %arg4 {
    scf.for %arg6 = %arg2 to %arg3 step %arg4 { }
  }
  br ^loopHeader(%inc : i32)
^exit:
  return
}
// CHECK-LABEL: --- DominanceInfo ---
// CHECK: Nearest(2, 0) = 0
// CHECK-NEXT: Nearest(2, 1) = 1
// CHECK-NEXT: Nearest(2, 2) = 2
// CHECK-NEXT: Nearest(2, 3) = 2
// CHECK-NEXT: Nearest(2, 4) = 2
// CHECK-NEXT: Nearest(2, 5) = 1
// CHECK: Nearest(3, 0) = 0
// CHECK-NEXT: Nearest(3, 1) = 1
// CHECK-NEXT: Nearest(3, 2) = 2
// CHECK-NEXT: Nearest(3, 3) = 3
// CHECK-NEXT: Nearest(3, 4) = 4
// CHECK-NEXT: Nearest(3, 5) = 1
// CHECK: Nearest(4, 0) = 0
// CHECK-NEXT: Nearest(4, 1) = 1
// CHECK-NEXT: Nearest(4, 2) = 2
// CHECK-NEXT: Nearest(4, 3) = 4
// CHECK-NEXT: Nearest(4, 4) = 4
// CHECK-NEXT: Nearest(4, 5) = 1
// CHECK-LABEL: --- PostDominanceInfo ---
// CHECK: Nearest(2, 0) = 1
// CHECK-NEXT: Nearest(2, 1) = 1
// CHECK-NEXT: Nearest(2, 2) = 2
// CHECK-NEXT: Nearest(2, 3) = 2
// CHECK-NEXT: Nearest(2, 4) = 2
// CHECK-NEXT: Nearest(2, 5) = 5
// CHECK: Nearest(3, 0) = 1
// CHECK-NEXT: Nearest(3, 1) = 1
// CHECK-NEXT: Nearest(3, 2) = 2
// CHECK-NEXT: Nearest(3, 3) = 3
// CHECK-NEXT: Nearest(3, 4) = 4
// CHECK-NEXT: Nearest(3, 5) = 5
// CHECK: Nearest(4, 0) = 1
// CHECK-NEXT: Nearest(4, 1) = 1
// CHECK-NEXT: Nearest(4, 2) = 2
// CHECK-NEXT: Nearest(4, 3) = 4
// CHECK-NEXT: Nearest(4, 4) = 4
// CHECK-NEXT: Nearest(4, 5) = 5
// CHECK-LABEL: --- Block Dominance relationship ---
// CHECK-NEXT: dominates(0, 0) = true
// CHECK-NEXT: dominates(0, 1) = true
// CHECK-NEXT: dominates(0, 2) = true
// CHECK-NEXT: dominates(0, 3) = true
// CHECK-NEXT: dominates(0, 4) = true
// CHECK-NEXT: dominates(0, 5) = true
// CHECK-NEXT: dominates(0, 6) = false
// CHECK-NEXT: dominates(1, 0) = false
// CHECK-NEXT: dominates(1, 1) = true
// CHECK-NEXT: dominates(1, 2) = true
// CHECK-NEXT: dominates(1, 3) = true
// CHECK-NEXT: dominates(1, 4) = true
// CHECK-NEXT: dominates(1, 5) = true
// CHECK-NEXT: dominates(1, 6) = false
// CHECK-NEXT: dominates(2, 0) = false
// CHECK-NEXT: dominates(2, 1) = false
// CHECK-NEXT: dominates(2, 2) = true
// CHECK-NEXT: dominates(2, 3) = true
// CHECK-NEXT: dominates(2, 4) = true
// CHECK-NEXT: dominates(2, 5) = false
// CHECK-NEXT: dominates(2, 6) = false
// CHECK-NEXT: dominates(3, 0) = false
// CHECK-NEXT: dominates(3, 1) = false
// CHECK-NEXT: dominates(3, 2) = false
// CHECK-NEXT: dominates(3, 3) = true
// CHECK-NEXT: dominates(3, 4) = false
// CHECK-NEXT: dominates(3, 5) = false
// CHECK-NEXT: dominates(3, 6) = false
// CHECK-NEXT: dominates(4, 0) = false
// CHECK-NEXT: dominates(4, 1) = false
// CHECK-NEXT: dominates(4, 2) = false
// CHECK-NEXT: dominates(4, 3) = true
// CHECK-NEXT: dominates(4, 4) = true
// CHECK-NEXT: dominates(4, 5) = false
// CHECK-NEXT: dominates(4, 6) = false
// CHECK-NEXT: dominates(5, 0) = false
// CHECK-NEXT: dominates(5, 1) = false
// CHECK-NEXT: dominates(5, 2) = false
// CHECK-NEXT: dominates(5, 3) = false
// CHECK-NEXT: dominates(5, 4) = false
// CHECK-NEXT: dominates(5, 5) = true
// CHECK-NEXT: dominates(5, 6) = false
// CHECK-NEXT: dominates(6, 0) = true
// CHECK-NEXT: dominates(6, 1) = true
// CHECK-NEXT: dominates(6, 2) = true
// CHECK-NEXT: dominates(6, 3) = true
// CHECK-NEXT: dominates(6, 4) = true
// CHECK-NEXT: dominates(6, 5) = true
// CHECK-NEXT: dominates(6, 6) = true
// CHECK-LABEL: --- Block PostDominance relationship ---
// CHECK-NEXT: postdominates(0, 0) = true
// CHECK-NEXT: postdominates(0, 1) = false
// CHECK-NEXT: postdominates(0, 2) = false
// CHECK-NEXT: postdominates(0, 3) = false
// CHECK-NEXT: postdominates(0, 4) = false
// CHECK-NEXT: postdominates(0, 5) = false
// CHECK-NEXT: postdominates(0, 6) = false
// CHECK-NEXT: postdominates(1, 0) = true
// CHECK-NEXT: postdominates(1, 1) = true
// CHECK-NEXT: postdominates(1, 2) = true
// CHECK-NEXT: postdominates(1, 3) = true
// CHECK-NEXT: postdominates(1, 4) = true
// CHECK-NEXT: postdominates(1, 5) = false
// CHECK-NEXT: postdominates(1, 6) = false
// CHECK-NEXT: postdominates(2, 0) = false
// CHECK-NEXT: postdominates(2, 1) = false
// CHECK-NEXT: postdominates(2, 2) = true
// CHECK-NEXT: postdominates(2, 3) = true
// CHECK-NEXT: postdominates(2, 4) = true
// CHECK-NEXT: postdominates(2, 5) = false
// CHECK-NEXT: postdominates(2, 6) = false
// CHECK-NEXT: postdominates(3, 0) = false
// CHECK-NEXT: postdominates(3, 1) = false
// CHECK-NEXT: postdominates(3, 2) = false
// CHECK-NEXT: postdominates(3, 3) = true
// CHECK-NEXT: postdominates(3, 4) = false
// CHECK-NEXT: postdominates(3, 5) = false
// CHECK-NEXT: postdominates(3, 6) = false
// CHECK-NEXT: postdominates(4, 0) = false
// CHECK-NEXT: postdominates(4, 1) = false
// CHECK-NEXT: postdominates(4, 2) = false
// CHECK-NEXT: postdominates(4, 3) = true
// CHECK-NEXT: postdominates(4, 4) = true
// CHECK-NEXT: postdominates(4, 5) = false
// CHECK-NEXT: postdominates(4, 6) = false
// CHECK-NEXT: postdominates(5, 0) = true
// CHECK-NEXT: postdominates(5, 1) = true
// CHECK-NEXT: postdominates(5, 2) = true
// CHECK-NEXT: postdominates(5, 3) = true
// CHECK-NEXT: postdominates(5, 4) = true
// CHECK-NEXT: postdominates(5, 5) = true
// CHECK-NEXT: postdominates(5, 6) = false
// CHECK-NEXT: postdominates(6, 0) = true
// CHECK-NEXT: postdominates(6, 1) = true
// CHECK-NEXT: postdominates(6, 2) = true
// CHECK-NEXT: postdominates(6, 3) = true
// CHECK-NEXT: postdominates(6, 4) = true
// CHECK-NEXT: postdominates(6, 5) = true
// CHECK-NEXT: postdominates(6, 6) = true
