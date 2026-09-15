// RUN: mlir-opt --mlir-disable-threading -pass-pipeline="builtin.module(any(test-control-flow-predecessors))" %s 2>&1 | FileCheck %s

// Test that getControlFlowPredecessors correctly handles values that are not
// successor inputs (issue #175168). Before the fix, these cases would either
// crash (out-of-bounds index) or silently return wrong results.

// Test case 1: scf.for with no iter_args.
// The induction variable (%iv) is block arg #0 and is NOT in
// getSuccessorInputs(body region), since there are no iter_args.
// Before the fix, getControlFlowPredecessors(%iv) would call
// getPredecessorValues with index 0 on an empty successor-inputs range,
// causing an out-of-bounds access.

// CHECK-LABEL: Control flow predecessors for '"test_scf_for_no_iter_args"'
// CHECK:   'scf.for': region #0 block arg #0: no predecessors
func.func @test_scf_for_no_iter_args(%lb: index, %ub: index, %step: index) {
  scf.for %iv = %lb to %ub step %step {
  }
  return
}

// Test case 2: test.loop_with_extra_result.
// - result #0 (extraResult): NOT in getSuccessorInputs(parent) — no predecessors.
// - result #1 (iterResult): IS in getSuccessorInputs(parent) — has predecessors.
// - body block arg #0 (extraArg): NOT in getSuccessorInputs(body) — no predecessors.
// - body block arg #1 (iterArg): IS in getSuccessorInputs(body) — has predecessors.
// Before the fix, querying extraResult/extraArg would use the wrong index into
// getSuccessorInputs, returning a predecessor for the wrong value.

// CHECK-LABEL: Control flow predecessors for '"test_loop_with_extra_result"'
// CHECK:   'test.loop_with_extra_result': result #0: no predecessors
// CHECK:   'test.loop_with_extra_result': result #1: 1 predecessor(s)
// CHECK:   'test.loop_with_extra_result': region #0 block arg #0: no predecessors
// CHECK:   'test.loop_with_extra_result': region #0 block arg #1: 2 predecessor(s)
func.func @test_loop_with_extra_result(%init: i32) {
  %extra, %iter = test.loop_with_extra_result %init : i32 -> (i32, i32) {
  ^bb0(%extra_arg: i32, %iter_arg: i32):
    test.loop_with_extra_result_yield %iter_arg : i32
  }
  return
}
