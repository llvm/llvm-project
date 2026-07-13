// RUN: mlir-opt %s -pass-pipeline="builtin.module(func.func(test-print-dominance))" -split-input-file | FileCheck %s

// CHECK-LABEL: Testing : func_condBranch
func.func @func_condBranch(%cond : i1) {
  cf.cond_br %cond, ^bb1, ^bb2
^bb1:
  cf.br ^exit
^bb2:
  cf.br ^exit
^exit:
  return
}

// CHECK: --- DominanceInfo ---
// CHECK: Nearest(0, 0) = 0
// CHECK: Nearest(0, 1) = 0
// CHECK: Nearest(0, 2) = 0
// CHECK: Nearest(0, 3) = 0
// CHECK: Nearest(0, 4) = 4
// CHECK: Nearest(1, 0) = 0
// CHECK: Nearest(1, 1) = 1
// CHECK: Nearest(1, 2) = 0
// CHECK: Nearest(1, 3) = 0
// CHECK: Nearest(1, 4) = 4
// CHECK: Nearest(2, 0) = 0
// CHECK: Nearest(2, 1) = 0
// CHECK: Nearest(2, 2) = 2
// CHECK: Nearest(2, 3) = 0
// CHECK: Nearest(2, 4) = 4
// CHECK: Nearest(3, 0) = 0
// CHECK: Nearest(3, 1) = 0
// CHECK: Nearest(3, 2) = 0
// CHECK: Nearest(3, 3) = 3
// CHECK: Nearest(3, 4) = 4
// CHECK: Nearest(4, 0) = 4
// CHECK: Nearest(4, 1) = 4
// CHECK: Nearest(4, 2) = 4
// CHECK: Nearest(4, 3) = 4
// CHECK: Nearest(4, 4) = 4

// CHECK: --- PostDominanceInfo ---
// CHECK: Nearest(0, 0) = 0
// CHECK: Nearest(0, 1) = 3
// CHECK: Nearest(0, 2) = 3
// CHECK: Nearest(0, 3) = 3
// CHECK: Nearest(0, 4) = 4
// CHECK: Nearest(1, 0) = 3
// CHECK: Nearest(1, 1) = 1
// CHECK: Nearest(1, 2) = 3
// CHECK: Nearest(1, 3) = 3
// CHECK: Nearest(1, 4) = 4
// CHECK: Nearest(2, 0) = 3
// CHECK: Nearest(2, 1) = 3
// CHECK: Nearest(2, 2) = 2
// CHECK: Nearest(2, 3) = 3
// CHECK: Nearest(2, 4) = 4
// CHECK: Nearest(3, 0) = 3
// CHECK: Nearest(3, 1) = 3
// CHECK: Nearest(3, 2) = 3
// CHECK: Nearest(3, 3) = 3
// CHECK: Nearest(3, 4) = 4
// CHECK: Nearest(4, 0) = 4
// CHECK: Nearest(4, 1) = 4
// CHECK: Nearest(4, 2) = 4
// CHECK: Nearest(4, 3) = 4
// CHECK: Nearest(4, 4) = 4

// CHECK: --- Block Dominance relationship ---
// CHECK: dominates(0, 0) = 1 (properly = 0)
// CHECK: dominates(0, 1) = 1 (properly = 1)
// CHECK: dominates(0, 2) = 1 (properly = 1)
// CHECK: dominates(0, 3) = 1 (properly = 1)
// CHECK: dominates(0, 4) = 0 (properly = 0)
// CHECK: dominates(1, 0) = 0 (properly = 0)
// CHECK: dominates(1, 1) = 1 (properly = 0)
// CHECK: dominates(1, 2) = 0 (properly = 0)
// CHECK: dominates(1, 3) = 0 (properly = 0)
// CHECK: dominates(1, 4) = 0 (properly = 0)
// CHECK: dominates(2, 0) = 0 (properly = 0)
// CHECK: dominates(2, 1) = 0 (properly = 0)
// CHECK: dominates(2, 2) = 1 (properly = 0)
// CHECK: dominates(2, 3) = 0 (properly = 0)
// CHECK: dominates(2, 4) = 0 (properly = 0)
// CHECK: dominates(3, 0) = 0 (properly = 0)
// CHECK: dominates(3, 1) = 0 (properly = 0)
// CHECK: dominates(3, 2) = 0 (properly = 0)
// CHECK: dominates(3, 3) = 1 (properly = 0)
// CHECK: dominates(3, 4) = 0 (properly = 0)
// CHECK: dominates(4, 0) = 1 (properly = 1)
// CHECK: dominates(4, 1) = 1 (properly = 1)
// CHECK: dominates(4, 2) = 1 (properly = 1)
// CHECK: dominates(4, 3) = 1 (properly = 1)
// CHECK: dominates(4, 4) = 1 (properly = 1)

// CHECK: --- Block PostDominance relationship ---
// CHECK: postdominates(0, 0) = 1 (properly = 0)
// CHECK: postdominates(0, 1) = 0 (properly = 0)
// CHECK: postdominates(0, 2) = 0 (properly = 0)
// CHECK: postdominates(0, 3) = 0 (properly = 0)
// CHECK: postdominates(0, 4) = 0 (properly = 0)
// CHECK: postdominates(1, 0) = 0 (properly = 0)
// CHECK: postdominates(1, 1) = 1 (properly = 0)
// CHECK: postdominates(1, 2) = 0 (properly = 0)
// CHECK: postdominates(1, 3) = 0 (properly = 0)
// CHECK: postdominates(1, 4) = 0 (properly = 0)
// CHECK: postdominates(2, 0) = 0 (properly = 0)
// CHECK: postdominates(2, 1) = 0 (properly = 0)
// CHECK: postdominates(2, 2) = 1 (properly = 0)
// CHECK: postdominates(2, 3) = 0 (properly = 0)
// CHECK: postdominates(2, 4) = 0 (properly = 0)
// CHECK: postdominates(3, 0) = 1 (properly = 1)
// CHECK: postdominates(3, 1) = 1 (properly = 1)
// CHECK: postdominates(3, 2) = 1 (properly = 1)
// CHECK: postdominates(3, 3) = 1 (properly = 0)
// CHECK: postdominates(3, 4) = 0 (properly = 0)
// CHECK: postdominates(4, 0) = 1 (properly = 1)
// CHECK: postdominates(4, 1) = 1 (properly = 1)
// CHECK: postdominates(4, 2) = 1 (properly = 1)
// CHECK: postdominates(4, 3) = 1 (properly = 1)
// CHECK: postdominates(4, 4) = 1 (properly = 1)

// CHECK: module attributes {test.block_ids = array<i64: 4>}
// CHECK:   func.func @func_condBranch({{.*}}) attributes {test.block_ids = array<i64: 0, 1, 2, 3>}

// -----

// CHECK-LABEL: Testing : func_loop
func.func @func_loop(%arg0 : i32, %arg1 : i32) {
  cf.br ^loopHeader(%arg0 : i32)
^loopHeader(%counter : i32):
  %lessThan = arith.cmpi slt, %counter, %arg1 : i32
  cf.cond_br %lessThan, ^loopBody, ^exit
^loopBody:
  %const0 = arith.constant 1 : i32
  %inc = arith.addi %counter, %const0 : i32
  cf.br ^loopHeader(%inc : i32)
^exit:
  return
}

// CHECK: --- DominanceInfo ---
// CHECK: Nearest(0, 0) = 0
// CHECK: Nearest(0, 1) = 0
// CHECK: Nearest(0, 2) = 0
// CHECK: Nearest(0, 3) = 0
// CHECK: Nearest(0, 4) = 4
// CHECK: Nearest(1, 0) = 0
// CHECK: Nearest(1, 1) = 1
// CHECK: Nearest(1, 2) = 1
// CHECK: Nearest(1, 3) = 1
// CHECK: Nearest(1, 4) = 4
// CHECK: Nearest(2, 0) = 0
// CHECK: Nearest(2, 1) = 1
// CHECK: Nearest(2, 2) = 2
// CHECK: Nearest(2, 3) = 1
// CHECK: Nearest(2, 4) = 4
// CHECK: Nearest(3, 0) = 0
// CHECK: Nearest(3, 1) = 1
// CHECK: Nearest(3, 2) = 1
// CHECK: Nearest(3, 3) = 3
// CHECK: Nearest(3, 4) = 4
// CHECK: Nearest(4, 0) = 4
// CHECK: Nearest(4, 1) = 4
// CHECK: Nearest(4, 2) = 4
// CHECK: Nearest(4, 3) = 4
// CHECK: Nearest(4, 4) = 4

// CHECK: --- PostDominanceInfo ---
// CHECK: Nearest(0, 0) = 0
// CHECK: Nearest(0, 1) = 1
// CHECK: Nearest(0, 2) = 1
// CHECK: Nearest(0, 3) = 3
// CHECK: Nearest(0, 4) = 4
// CHECK: Nearest(1, 0) = 1
// CHECK: Nearest(1, 1) = 1
// CHECK: Nearest(1, 2) = 1
// CHECK: Nearest(1, 3) = 3
// CHECK: Nearest(1, 4) = 4
// CHECK: Nearest(2, 0) = 1
// CHECK: Nearest(2, 1) = 1
// CHECK: Nearest(2, 2) = 2
// CHECK: Nearest(2, 3) = 3
// CHECK: Nearest(2, 4) = 4
// CHECK: Nearest(3, 0) = 3
// CHECK: Nearest(3, 1) = 3
// CHECK: Nearest(3, 2) = 3
// CHECK: Nearest(3, 3) = 3
// CHECK: Nearest(3, 4) = 4
// CHECK: Nearest(4, 0) = 4
// CHECK: Nearest(4, 1) = 4
// CHECK: Nearest(4, 2) = 4
// CHECK: Nearest(4, 3) = 4
// CHECK: Nearest(4, 4) = 4

// CHECK: --- Block Dominance relationship ---
// CHECK: dominates(0, 0) = 1 (properly = 0)
// CHECK: dominates(0, 1) = 1 (properly = 1)
// CHECK: dominates(0, 2) = 1 (properly = 1)
// CHECK: dominates(0, 3) = 1 (properly = 1)
// CHECK: dominates(0, 4) = 0 (properly = 0)
// CHECK: dominates(1, 0) = 0 (properly = 0)
// CHECK: dominates(1, 1) = 1 (properly = 0)
// CHECK: dominates(1, 2) = 1 (properly = 1)
// CHECK: dominates(1, 3) = 1 (properly = 1)
// CHECK: dominates(1, 4) = 0 (properly = 0)
// CHECK: dominates(2, 0) = 0 (properly = 0)
// CHECK: dominates(2, 1) = 0 (properly = 0)
// CHECK: dominates(2, 2) = 1 (properly = 0)
// CHECK: dominates(2, 3) = 0 (properly = 0)
// CHECK: dominates(2, 4) = 0 (properly = 0)
// CHECK: dominates(3, 0) = 0 (properly = 0)
// CHECK: dominates(3, 1) = 0 (properly = 0)
// CHECK: dominates(3, 2) = 0 (properly = 0)
// CHECK: dominates(3, 3) = 1 (properly = 0)
// CHECK: dominates(3, 4) = 0 (properly = 0)
// CHECK: dominates(4, 0) = 1 (properly = 1)
// CHECK: dominates(4, 1) = 1 (properly = 1)
// CHECK: dominates(4, 2) = 1 (properly = 1)
// CHECK: dominates(4, 3) = 1 (properly = 1)
// CHECK: dominates(4, 4) = 1 (properly = 1)

// CHECK: --- Block PostDominance relationship ---
// CHECK: postdominates(0, 0) = 1 (properly = 0)
// CHECK: postdominates(0, 1) = 0 (properly = 0)
// CHECK: postdominates(0, 2) = 0 (properly = 0)
// CHECK: postdominates(0, 3) = 0 (properly = 0)
// CHECK: postdominates(0, 4) = 0 (properly = 0)
// CHECK: postdominates(1, 0) = 1 (properly = 1)
// CHECK: postdominates(1, 1) = 1 (properly = 0)
// CHECK: postdominates(1, 2) = 1 (properly = 1)
// CHECK: postdominates(1, 3) = 0 (properly = 0)
// CHECK: postdominates(1, 4) = 0 (properly = 0)
// CHECK: postdominates(2, 0) = 0 (properly = 0)
// CHECK: postdominates(2, 1) = 0 (properly = 0)
// CHECK: postdominates(2, 2) = 1 (properly = 0)
// CHECK: postdominates(2, 3) = 0 (properly = 0)
// CHECK: postdominates(2, 4) = 0 (properly = 0)
// CHECK: postdominates(3, 0) = 1 (properly = 1)
// CHECK: postdominates(3, 1) = 1 (properly = 1)
// CHECK: postdominates(3, 2) = 1 (properly = 1)
// CHECK: postdominates(3, 3) = 1 (properly = 0)
// CHECK: postdominates(3, 4) = 0 (properly = 0)
// CHECK: postdominates(4, 0) = 1 (properly = 1)
// CHECK: postdominates(4, 1) = 1 (properly = 1)
// CHECK: postdominates(4, 2) = 1 (properly = 1)
// CHECK: postdominates(4, 3) = 1 (properly = 1)
// CHECK: postdominates(4, 4) = 1 (properly = 1)

// CHECK: module attributes {test.block_ids = array<i64: 4>}
// CHECK:   func.func @func_loop({{.*}}) attributes {test.block_ids = array<i64: 0, 1, 2, 3>}

// -----

// CHECK-LABEL: Testing : nested_region
func.func @nested_region(%arg0 : index, %arg1 : index, %arg2 : index) {
  scf.for %arg3 = %arg0 to %arg1 step %arg2 { }
  return
}

// CHECK: --- DominanceInfo ---
// CHECK: Nearest(0, 0) = 0
// CHECK: Nearest(0, 1) = 1
// CHECK: Nearest(0, 2) = 2
// CHECK: Nearest(1, 0) = 1
// CHECK: Nearest(1, 1) = 1
// CHECK: Nearest(1, 2) = 2
// CHECK: Nearest(2, 0) = 2
// CHECK: Nearest(2, 1) = 2
// CHECK: Nearest(2, 2) = 2

// CHECK: --- PostDominanceInfo ---
// CHECK: Nearest(0, 0) = 0
// CHECK: Nearest(0, 1) = 1
// CHECK: Nearest(0, 2) = 2
// CHECK: Nearest(1, 0) = 1
// CHECK: Nearest(1, 1) = 1
// CHECK: Nearest(1, 2) = 2
// CHECK: Nearest(2, 0) = 2
// CHECK: Nearest(2, 1) = 2
// CHECK: Nearest(2, 2) = 2

// CHECK: --- Block Dominance relationship ---
// CHECK: dominates(0, 0) = 1 (properly = 0)
// CHECK: dominates(0, 1) = 0 (properly = 0)
// CHECK: dominates(0, 2) = 0 (properly = 0)
// CHECK: dominates(1, 0) = 1 (properly = 1)
// CHECK: dominates(1, 1) = 1 (properly = 0)
// CHECK: dominates(1, 2) = 0 (properly = 0)
// CHECK: dominates(2, 0) = 1 (properly = 1)
// CHECK: dominates(2, 1) = 1 (properly = 1)
// CHECK: dominates(2, 2) = 1 (properly = 1)

// CHECK: --- Block PostDominance relationship ---
// CHECK: postdominates(0, 0) = 1 (properly = 0)
// CHECK: postdominates(0, 1) = 0 (properly = 0)
// CHECK: postdominates(0, 2) = 0 (properly = 0)
// CHECK: postdominates(1, 0) = 1 (properly = 1)
// CHECK: postdominates(1, 1) = 1 (properly = 0)
// CHECK: postdominates(1, 2) = 0 (properly = 0)
// CHECK: postdominates(2, 0) = 1 (properly = 1)
// CHECK: postdominates(2, 1) = 1 (properly = 1)
// CHECK: postdominates(2, 2) = 1 (properly = 1)

// CHECK: module attributes {test.block_ids = array<i64: 2>} {
// CHECK:   func.func @nested_region({{.*}}) attributes {test.block_ids = array<i64: 1>} {
// CHECK:     scf.for {{.*}} {
// CHECK:     } {test.block_ids = array<i64: 0>}
// CHECK:     return
// CHECK:   }
// CHECK: }

// -----

// CHECK-LABEL: Testing : nested_region2
func.func @nested_region2(%arg0 : index, %arg1 : index, %arg2 : index) {
  scf.for %arg3 = %arg0 to %arg1 step %arg2 {
    scf.for %arg4 = %arg0 to %arg1 step %arg2 {
      scf.for %arg5 = %arg0 to %arg1 step %arg2 { }
    }
  }
  return
}

// CHECK: --- DominanceInfo ---
// CHECK: Nearest(0, 0) = 0
// CHECK: Nearest(0, 1) = 1
// CHECK: Nearest(0, 2) = 2
// CHECK: Nearest(0, 3) = 3
// CHECK: Nearest(0, 4) = 4
// CHECK: Nearest(1, 0) = 1
// CHECK: Nearest(1, 1) = 1
// CHECK: Nearest(1, 2) = 2
// CHECK: Nearest(1, 3) = 3
// CHECK: Nearest(1, 4) = 4
// CHECK: Nearest(2, 0) = 2
// CHECK: Nearest(2, 1) = 2
// CHECK: Nearest(2, 2) = 2
// CHECK: Nearest(2, 3) = 3
// CHECK: Nearest(2, 4) = 4
// CHECK: Nearest(3, 0) = 3
// CHECK: Nearest(3, 1) = 3
// CHECK: Nearest(3, 2) = 3
// CHECK: Nearest(3, 3) = 3
// CHECK: Nearest(3, 4) = 4
// CHECK: Nearest(4, 0) = 4
// CHECK: Nearest(4, 1) = 4
// CHECK: Nearest(4, 2) = 4
// CHECK: Nearest(4, 3) = 4
// CHECK: Nearest(4, 4) = 4

// CHECK: --- PostDominanceInfo ---
// CHECK: Nearest(0, 0) = 0
// CHECK: Nearest(0, 1) = 1
// CHECK: Nearest(0, 2) = 2
// CHECK: Nearest(0, 3) = 3
// CHECK: Nearest(0, 4) = 4
// CHECK: Nearest(1, 0) = 1
// CHECK: Nearest(1, 1) = 1
// CHECK: Nearest(1, 2) = 2
// CHECK: Nearest(1, 3) = 3
// CHECK: Nearest(1, 4) = 4
// CHECK: Nearest(2, 0) = 2
// CHECK: Nearest(2, 1) = 2
// CHECK: Nearest(2, 2) = 2
// CHECK: Nearest(2, 3) = 3
// CHECK: Nearest(2, 4) = 4
// CHECK: Nearest(3, 0) = 3
// CHECK: Nearest(3, 1) = 3
// CHECK: Nearest(3, 2) = 3
// CHECK: Nearest(3, 3) = 3
// CHECK: Nearest(3, 4) = 4
// CHECK: Nearest(4, 0) = 4
// CHECK: Nearest(4, 1) = 4
// CHECK: Nearest(4, 2) = 4
// CHECK: Nearest(4, 3) = 4
// CHECK: Nearest(4, 4) = 4

// CHECK: --- Block Dominance relationship ---
// CHECK: dominates(0, 0) = 1 (properly = 0)
// CHECK: dominates(0, 1) = 0 (properly = 0)
// CHECK: dominates(0, 2) = 0 (properly = 0)
// CHECK: dominates(0, 3) = 0 (properly = 0)
// CHECK: dominates(0, 4) = 0 (properly = 0)
// CHECK: dominates(1, 0) = 1 (properly = 1)
// CHECK: dominates(1, 1) = 1 (properly = 0)
// CHECK: dominates(1, 2) = 0 (properly = 0)
// CHECK: dominates(1, 3) = 0 (properly = 0)
// CHECK: dominates(1, 4) = 0 (properly = 0)
// CHECK: dominates(2, 0) = 1 (properly = 1)
// CHECK: dominates(2, 1) = 1 (properly = 1)
// CHECK: dominates(2, 2) = 1 (properly = 0)
// CHECK: dominates(2, 3) = 0 (properly = 0)
// CHECK: dominates(2, 4) = 0 (properly = 0)
// CHECK: dominates(3, 0) = 1 (properly = 1)
// CHECK: dominates(3, 1) = 1 (properly = 1)
// CHECK: dominates(3, 2) = 1 (properly = 1)
// CHECK: dominates(3, 3) = 1 (properly = 0)
// CHECK: dominates(3, 4) = 0 (properly = 0)
// CHECK: dominates(4, 0) = 1 (properly = 1)
// CHECK: dominates(4, 1) = 1 (properly = 1)
// CHECK: dominates(4, 2) = 1 (properly = 1)
// CHECK: dominates(4, 3) = 1 (properly = 1)
// CHECK: dominates(4, 4) = 1 (properly = 1)

// CHECK: --- Block PostDominance relationship ---
// CHECK: postdominates(0, 0) = 1 (properly = 0)
// CHECK: postdominates(0, 1) = 0 (properly = 0)
// CHECK: postdominates(0, 2) = 0 (properly = 0)
// CHECK: postdominates(0, 3) = 0 (properly = 0)
// CHECK: postdominates(0, 4) = 0 (properly = 0)
// CHECK: postdominates(1, 0) = 1 (properly = 1)
// CHECK: postdominates(1, 1) = 1 (properly = 0)
// CHECK: postdominates(1, 2) = 0 (properly = 0)
// CHECK: postdominates(1, 3) = 0 (properly = 0)
// CHECK: postdominates(1, 4) = 0 (properly = 0)
// CHECK: postdominates(2, 0) = 1 (properly = 1)
// CHECK: postdominates(2, 1) = 1 (properly = 1)
// CHECK: postdominates(2, 2) = 1 (properly = 0)
// CHECK: postdominates(2, 3) = 0 (properly = 0)
// CHECK: postdominates(2, 4) = 0 (properly = 0)
// CHECK: postdominates(3, 0) = 1 (properly = 1)
// CHECK: postdominates(3, 1) = 1 (properly = 1)
// CHECK: postdominates(3, 2) = 1 (properly = 1)
// CHECK: postdominates(3, 3) = 1 (properly = 0)
// CHECK: postdominates(3, 4) = 0 (properly = 0)
// CHECK: postdominates(4, 0) = 1 (properly = 1)
// CHECK: postdominates(4, 1) = 1 (properly = 1)
// CHECK: postdominates(4, 2) = 1 (properly = 1)
// CHECK: postdominates(4, 3) = 1 (properly = 1)
// CHECK: postdominates(4, 4) = 1 (properly = 1)

// CHECK: module attributes {test.block_ids = array<i64: 4>} {
// CHECK:   func.func @nested_region2({{.*}}) attributes {test.block_ids = array<i64: 3>} {
// CHECK:     scf.for {{.*}} {
// CHECK:       scf.for {{.*}} {
// CHECK:         scf.for {{.*}} {
// CHECK:         } {test.block_ids = array<i64: 0>}
// CHECK:       } {test.block_ids = array<i64: 1>}
// CHECK:     } {test.block_ids = array<i64: 2>}
// CHECK:     return
// CHECK:   }
// CHECK: }

// -----

// CHECK-LABEL: Testing : func_loop_nested_region
func.func @func_loop_nested_region(
  %arg0 : i32,
  %arg1 : i32,
  %arg2 : index,
  %arg3 : index,
  %arg4 : index) {
  cf.br ^loopHeader(%arg0 : i32)
^loopHeader(%counter : i32):
  %lessThan = arith.cmpi slt, %counter, %arg1 : i32
  cf.cond_br %lessThan, ^loopBody, ^exit
^loopBody:
  %const0 = arith.constant 1 : i32
  %inc = arith.addi %counter, %const0 : i32
  scf.for %arg5 = %arg2 to %arg3 step %arg4 {
    scf.for %arg6 = %arg2 to %arg3 step %arg4 { }
  }
  cf.br ^loopHeader(%inc : i32)
^exit:
  return
}

// CHECK: --- DominanceInfo ---
// CHECK: Nearest(0, 0) = 0
// CHECK: Nearest(0, 1) = 0
// CHECK: Nearest(0, 2) = 0
// CHECK: Nearest(0, 3) = 0
// CHECK: Nearest(0, 4) = 0
// CHECK: Nearest(0, 5) = 0
// CHECK: Nearest(0, 6) = 6
// CHECK: Nearest(1, 0) = 0
// CHECK: Nearest(1, 1) = 1
// CHECK: Nearest(1, 2) = 1
// CHECK: Nearest(1, 3) = 1
// CHECK: Nearest(1, 4) = 1
// CHECK: Nearest(1, 5) = 1
// CHECK: Nearest(1, 6) = 6
// CHECK: Nearest(2, 0) = 0
// CHECK: Nearest(2, 1) = 1
// CHECK: Nearest(2, 2) = 2
// CHECK: Nearest(2, 3) = 2
// CHECK: Nearest(2, 4) = 2
// CHECK: Nearest(2, 5) = 1
// CHECK: Nearest(2, 6) = 6
// CHECK: Nearest(3, 0) = 0
// CHECK: Nearest(3, 1) = 1
// CHECK: Nearest(3, 2) = 2
// CHECK: Nearest(3, 3) = 3
// CHECK: Nearest(3, 4) = 4
// CHECK: Nearest(3, 5) = 1
// CHECK: Nearest(3, 6) = 6
// CHECK: Nearest(4, 0) = 0
// CHECK: Nearest(4, 1) = 1
// CHECK: Nearest(4, 2) = 2
// CHECK: Nearest(4, 3) = 4
// CHECK: Nearest(4, 4) = 4
// CHECK: Nearest(4, 5) = 1
// CHECK: Nearest(4, 6) = 6
// CHECK: Nearest(5, 0) = 0
// CHECK: Nearest(5, 1) = 1
// CHECK: Nearest(5, 2) = 1
// CHECK: Nearest(5, 3) = 1
// CHECK: Nearest(5, 4) = 1
// CHECK: Nearest(5, 5) = 5
// CHECK: Nearest(5, 6) = 6
// CHECK: Nearest(6, 0) = 6
// CHECK: Nearest(6, 1) = 6
// CHECK: Nearest(6, 2) = 6
// CHECK: Nearest(6, 3) = 6
// CHECK: Nearest(6, 4) = 6
// CHECK: Nearest(6, 5) = 6
// CHECK: Nearest(6, 6) = 6

// CHECK: --- PostDominanceInfo ---
// CHECK: Nearest(0, 0) = 0
// CHECK: Nearest(0, 1) = 1
// CHECK: Nearest(0, 2) = 1
// CHECK: Nearest(0, 3) = 1
// CHECK: Nearest(0, 4) = 1
// CHECK: Nearest(0, 5) = 5
// CHECK: Nearest(0, 6) = 6
// CHECK: Nearest(1, 0) = 1
// CHECK: Nearest(1, 1) = 1
// CHECK: Nearest(1, 2) = 1
// CHECK: Nearest(1, 3) = 1
// CHECK: Nearest(1, 4) = 1
// CHECK: Nearest(1, 5) = 5
// CHECK: Nearest(1, 6) = 6
// CHECK: Nearest(2, 0) = 1
// CHECK: Nearest(2, 1) = 1
// CHECK: Nearest(2, 2) = 2
// CHECK: Nearest(2, 3) = 2
// CHECK: Nearest(2, 4) = 2
// CHECK: Nearest(2, 5) = 5
// CHECK: Nearest(2, 6) = 6
// CHECK: Nearest(3, 0) = 1
// CHECK: Nearest(3, 1) = 1
// CHECK: Nearest(3, 2) = 2
// CHECK: Nearest(3, 3) = 3
// CHECK: Nearest(3, 4) = 4
// CHECK: Nearest(3, 5) = 5
// CHECK: Nearest(3, 6) = 6
// CHECK: Nearest(4, 0) = 1
// CHECK: Nearest(4, 1) = 1
// CHECK: Nearest(4, 2) = 2
// CHECK: Nearest(4, 3) = 4
// CHECK: Nearest(4, 4) = 4
// CHECK: Nearest(4, 5) = 5
// CHECK: Nearest(4, 6) = 6
// CHECK: Nearest(5, 0) = 5
// CHECK: Nearest(5, 1) = 5
// CHECK: Nearest(5, 2) = 5
// CHECK: Nearest(5, 3) = 5
// CHECK: Nearest(5, 4) = 5
// CHECK: Nearest(5, 5) = 5
// CHECK: Nearest(5, 6) = 6
// CHECK: Nearest(6, 0) = 6
// CHECK: Nearest(6, 1) = 6
// CHECK: Nearest(6, 2) = 6
// CHECK: Nearest(6, 3) = 6
// CHECK: Nearest(6, 4) = 6
// CHECK: Nearest(6, 5) = 6
// CHECK: Nearest(6, 6) = 6

// CHECK: --- Block Dominance relationship ---
// CHECK: dominates(0, 0) = 1 (properly = 0)
// CHECK: dominates(0, 1) = 1 (properly = 1)
// CHECK: dominates(0, 2) = 1 (properly = 1)
// CHECK: dominates(0, 3) = 1 (properly = 1)
// CHECK: dominates(0, 4) = 1 (properly = 1)
// CHECK: dominates(0, 5) = 1 (properly = 1)
// CHECK: dominates(0, 6) = 0 (properly = 0)
// CHECK: dominates(1, 0) = 0 (properly = 0)
// CHECK: dominates(1, 1) = 1 (properly = 0)
// CHECK: dominates(1, 2) = 1 (properly = 1)
// CHECK: dominates(1, 3) = 1 (properly = 1)
// CHECK: dominates(1, 4) = 1 (properly = 1)
// CHECK: dominates(1, 5) = 1 (properly = 1)
// CHECK: dominates(1, 6) = 0 (properly = 0)
// CHECK: dominates(2, 0) = 0 (properly = 0)
// CHECK: dominates(2, 1) = 0 (properly = 0)
// CHECK: dominates(2, 2) = 1 (properly = 0)
// CHECK: dominates(2, 3) = 1 (properly = 1)
// CHECK: dominates(2, 4) = 1 (properly = 1)
// CHECK: dominates(2, 5) = 0 (properly = 0)
// CHECK: dominates(2, 6) = 0 (properly = 0)
// CHECK: dominates(3, 0) = 0 (properly = 0)
// CHECK: dominates(3, 1) = 0 (properly = 0)
// CHECK: dominates(3, 2) = 0 (properly = 0)
// CHECK: dominates(3, 3) = 1 (properly = 0)
// CHECK: dominates(3, 4) = 0 (properly = 0)
// CHECK: dominates(3, 5) = 0 (properly = 0)
// CHECK: dominates(3, 6) = 0 (properly = 0)
// CHECK: dominates(4, 0) = 0 (properly = 0)
// CHECK: dominates(4, 1) = 0 (properly = 0)
// CHECK: dominates(4, 2) = 0 (properly = 0)
// CHECK: dominates(4, 3) = 1 (properly = 1)
// CHECK: dominates(4, 4) = 1 (properly = 0)
// CHECK: dominates(4, 5) = 0 (properly = 0)
// CHECK: dominates(4, 6) = 0 (properly = 0)
// CHECK: dominates(5, 0) = 0 (properly = 0)
// CHECK: dominates(5, 1) = 0 (properly = 0)
// CHECK: dominates(5, 2) = 0 (properly = 0)
// CHECK: dominates(5, 3) = 0 (properly = 0)
// CHECK: dominates(5, 4) = 0 (properly = 0)
// CHECK: dominates(5, 5) = 1 (properly = 0)
// CHECK: dominates(5, 6) = 0 (properly = 0)
// CHECK: dominates(6, 0) = 1 (properly = 1)
// CHECK: dominates(6, 1) = 1 (properly = 1)
// CHECK: dominates(6, 2) = 1 (properly = 1)
// CHECK: dominates(6, 3) = 1 (properly = 1)
// CHECK: dominates(6, 4) = 1 (properly = 1)
// CHECK: dominates(6, 5) = 1 (properly = 1)
// CHECK: dominates(6, 6) = 1 (properly = 1)

// CHECK: --- Block PostDominance relationship ---
// CHECK: postdominates(0, 0) = 1 (properly = 0)
// CHECK: postdominates(0, 1) = 0 (properly = 0)
// CHECK: postdominates(0, 2) = 0 (properly = 0)
// CHECK: postdominates(0, 3) = 0 (properly = 0)
// CHECK: postdominates(0, 4) = 0 (properly = 0)
// CHECK: postdominates(0, 5) = 0 (properly = 0)
// CHECK: postdominates(0, 6) = 0 (properly = 0)
// CHECK: postdominates(1, 0) = 1 (properly = 1)
// CHECK: postdominates(1, 1) = 1 (properly = 0)
// CHECK: postdominates(1, 2) = 1 (properly = 1)
// CHECK: postdominates(1, 3) = 1 (properly = 1)
// CHECK: postdominates(1, 4) = 1 (properly = 1)
// CHECK: postdominates(1, 5) = 0 (properly = 0)
// CHECK: postdominates(1, 6) = 0 (properly = 0)
// CHECK: postdominates(2, 0) = 0 (properly = 0)
// CHECK: postdominates(2, 1) = 0 (properly = 0)
// CHECK: postdominates(2, 2) = 1 (properly = 0)
// CHECK: postdominates(2, 3) = 1 (properly = 1)
// CHECK: postdominates(2, 4) = 1 (properly = 1)
// CHECK: postdominates(2, 5) = 0 (properly = 0)
// CHECK: postdominates(2, 6) = 0 (properly = 0)
// CHECK: postdominates(3, 0) = 0 (properly = 0)
// CHECK: postdominates(3, 1) = 0 (properly = 0)
// CHECK: postdominates(3, 2) = 0 (properly = 0)
// CHECK: postdominates(3, 3) = 1 (properly = 0)
// CHECK: postdominates(3, 4) = 0 (properly = 0)
// CHECK: postdominates(3, 5) = 0 (properly = 0)
// CHECK: postdominates(3, 6) = 0 (properly = 0)
// CHECK: postdominates(4, 0) = 0 (properly = 0)
// CHECK: postdominates(4, 1) = 0 (properly = 0)
// CHECK: postdominates(4, 2) = 0 (properly = 0)
// CHECK: postdominates(4, 3) = 1 (properly = 1)
// CHECK: postdominates(4, 4) = 1 (properly = 0)
// CHECK: postdominates(4, 5) = 0 (properly = 0)
// CHECK: postdominates(4, 6) = 0 (properly = 0)
// CHECK: postdominates(5, 0) = 1 (properly = 1)
// CHECK: postdominates(5, 1) = 1 (properly = 1)
// CHECK: postdominates(5, 2) = 1 (properly = 1)
// CHECK: postdominates(5, 3) = 1 (properly = 1)
// CHECK: postdominates(5, 4) = 1 (properly = 1)
// CHECK: postdominates(5, 5) = 1 (properly = 0)
// CHECK: postdominates(5, 6) = 0 (properly = 0)
// CHECK: postdominates(6, 0) = 1 (properly = 1)
// CHECK: postdominates(6, 1) = 1 (properly = 1)
// CHECK: postdominates(6, 2) = 1 (properly = 1)
// CHECK: postdominates(6, 3) = 1 (properly = 1)
// CHECK: postdominates(6, 4) = 1 (properly = 1)
// CHECK: postdominates(6, 5) = 1 (properly = 1)
// CHECK: postdominates(6, 6) = 1 (properly = 1)

// CHECK: module attributes {test.block_ids = array<i64: 6>} {
// CHECK:   func.func @func_loop_nested_region({{.*}}) attributes {test.block_ids = array<i64: 0, 1, 2, 5>} {
// CHECK:   ^{{.*}}
// CHECK:   ^{{.*}}
// CHECK:     scf.for {{.*}} {
// CHECK:       scf.for {{.*}} {
// CHECK:       } {test.block_ids = array<i64: 3>}
// CHECK:     } {test.block_ids = array<i64: 4>}
// CHECK:   ^{{.*}}
// CHECK:   }
// CHECK: }
