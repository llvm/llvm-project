// RUN: mlir-opt -test-dead-code-analysis --split-input-file 2>&1 %s | FileCheck %s

// Tests verifying that DeadCodeAnalysis correctly propagates breaking control
// flow through PropagateControlFlowBreak ops (e.g. scf.if). The fix is in
// visitRegionTerminator: when the immediate parent RegionBranchOpInterface
// returns a propagating successor sentinel for a propagating break, we resolve
// the actual HasBreakingControlFlowOp ancestor and re-dispatch through it, so
// the correct predecessor edge is established at the loop's exit point.

// -----

// A loop whose only exit is scf.break propagated through scf.if.
// The loop's exit point (op_preds) must show the break as a predecessor.

// CHECK-LABEL: loop:
// CHECK:  region #0
// CHECK:   ^bb0 = live
// CHECK: region_preds: (all) predecessors:
// CHECK:   %0 = scf.loop
// CHECK:   scf.continue
// CHECK: op_preds: (all) predecessors:
// CHECK:   scf.break
func.func @test_break_through_if(%cond: i1) -> i32 {
  %result = scf.loop token(%loop) -> i32 {
    scf.if %cond {
      %c42 = arith.constant 42 : i32
      scf.break [%loop] %c42 : i32
    }
    scf.continue [%loop]
  } {tag = "loop"}
  return %result : i32
}

// -----

// For comparison: a loop with a direct scf.break.

// CHECK-LABEL: loop:
// CHECK:  region #0
// CHECK:   ^bb0 = live
// CHECK: region_preds: (all) predecessors:
// CHECK:   %0 = scf.loop
// CHECK: op_preds: (all) predecessors:
// CHECK:   scf.break
func.func @test_direct_break(%cond: i1) -> i32 {
  %result = scf.loop token(%loop) -> i32 {
    scf.if %cond {
    }
    %c42 = arith.constant 42 : i32
    scf.break [%loop] %c42 : i32
  } {tag = "loop"}
  return %result : i32
}

// -----

// A loop whose only exit is scf.break propagated through two nested scf.if
// ops. The loop's op_preds must show the break. The intermediate if ops do not
// appear in op_preds for the break — the break bypasses them without yielding
// values to them, which is correct.

// CHECK-LABEL: inner_if:
// CHECK:  region #0
// CHECK:   ^bb0 = live
// CHECK: region_preds: (all) predecessors:
// CHECK:   scf.if {{.*}} {tag = "inner_if"}
// CHECK:  region #1
// CHECK: op_preds: (all) predecessors:
// CHECK:   scf.if {{.*}} {tag = "inner_if"}

// CHECK-LABEL: outer_if:
// CHECK:  region #0
// CHECK:   ^bb0 = live
// CHECK: region_preds: (all) predecessors:
// CHECK:   scf.if {{.*}} {tag = "outer_if"}
// CHECK:  region #1
// CHECK: op_preds: (all) predecessors:
// CHECK:   scf.if {{.*}} {tag = "outer_if"}

// CHECK-LABEL: loop:
// CHECK:  region #0
// CHECK:   ^bb0 = live
// CHECK: region_preds: (all) predecessors:
// CHECK:   %0 = scf.loop
// CHECK:   scf.continue
// CHECK: op_preds: (all) predecessors:
// CHECK:   scf.break
func.func @test_break_through_nested_ifs(%cond1: i1, %cond2: i1) -> i32 {
  %result = scf.loop token(%loop) -> i32 {
    scf.if %cond1 {
      scf.if %cond2 {
        %c99 = arith.constant 99 : i32
        scf.break [%loop] %c99 : i32
      } {tag = "inner_if"}
      scf.continue [%loop]
    } {tag = "outer_if"}
    scf.continue [%loop]
  } {tag = "loop"}
  return %result : i32
}
