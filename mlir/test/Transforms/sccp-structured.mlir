// RUN: mlir-opt -allow-unregistered-dialect %s -pass-pipeline="func.func(sccp)" -split-input-file | FileCheck %s

/// Check that a constant is properly propagated when only one edge is taken.

// CHECK-LABEL: func @simple(
func.func @simple(%arg0 : i32) -> i32 {
  // CHECK: %[[CST:.*]] = arith.constant 1 : i32
  // CHECK-NOT: scf.if
  // CHECK: return %[[CST]] : i32

  %cond = arith.constant true
  %res = scf.if %cond -> (i32) {
    %1 = arith.constant 1 : i32
    scf.yield %1 : i32
  } else {
    scf.yield %arg0 : i32
  }
  return %res : i32
}

/// Check that a constant is properly propagated when both edges produce the
/// same value.

// CHECK-LABEL: func @simple_both_same(
func.func @simple_both_same(%cond : i1) -> i32 {
  // CHECK: %[[CST:.*]] = arith.constant 1 : i32
  // CHECK-NOT: scf.if
  // CHECK: return %[[CST]] : i32

  %res = scf.if %cond -> (i32) {
    %1 = arith.constant 1 : i32
    scf.yield %1 : i32
  } else {
    %2 = arith.constant 1 : i32
    scf.yield %2 : i32
  }
  return %res : i32
}

/// Check that the arguments go to overdefined if the branch cannot detect when
/// a specific successor is taken.

// CHECK-LABEL: func @overdefined_unknown_condition(
func.func @overdefined_unknown_condition(%cond : i1, %arg0 : i32) -> i32 {
  // CHECK: %[[RES:.*]] = scf.if
  // CHECK: return %[[RES]] : i32

  %res = scf.if %cond -> (i32) {
    %1 = arith.constant 1 : i32
    scf.yield %1 : i32
  } else {
    scf.yield %arg0 : i32
  }
  return %res : i32
}

/// Check that the arguments go to overdefined if there are conflicting
/// constants.

// CHECK-LABEL: func @overdefined_different_constants(
func.func @overdefined_different_constants(%cond : i1) -> i32 {
  // CHECK: %[[RES:.*]] = scf.if
  // CHECK: return %[[RES]] : i32

  %res = scf.if %cond -> (i32) {
    %1 = arith.constant 1 : i32
    scf.yield %1 : i32
  } else {
    %2 = arith.constant 2 : i32
    scf.yield %2 : i32
  }
  return %res : i32
}

/// Check that arguments are properly merged across loop-like control flow.

// CHECK-LABEL: func @simple_loop(
func.func @simple_loop(%arg0 : index, %arg1 : index, %arg2 : index) -> i32 {
  // CHECK: %[[CST:.*]] = arith.constant 0 : i32
  // CHECK-NOT: scf.for
  // CHECK: return %[[CST]] : i32

  %s0 = arith.constant 0 : i32
  %result = scf.for %i0 = %arg0 to %arg1 step %arg2 iter_args(%si = %s0) -> (i32) {
    %sn = arith.addi %si, %si : i32
    scf.yield %sn : i32
  }
  return %result : i32
}

/// Check that arguments go to overdefined when loop backedges produce a
/// conflicting value.

// CHECK-LABEL: func @loop_overdefined(
func.func @loop_overdefined(%arg0 : index, %arg1 : index, %arg2 : index) -> i32 {
  // CHECK: %[[RES:.*]] = scf.for
  // CHECK: return %[[RES]] : i32

  %s0 = arith.constant 1 : i32
  %result = scf.for %i0 = %arg0 to %arg1 step %arg2 iter_args(%si = %s0) -> (i32) {
    %sn = arith.addi %si, %si : i32
    scf.yield %sn : i32
  }
  return %result : i32
}

/// Test that we can properly propagate within inner control, and in situations
/// where the executable edges within the CFG are sensitive to the current state
/// of the analysis.

// CHECK-LABEL: func @loop_inner_control_flow(
func.func @loop_inner_control_flow(%arg0 : index, %arg1 : index, %arg2 : index) -> i32 {
  // CHECK: %[[CST:.*]] = arith.constant 1 : i32
  // CHECK-NOT: scf.for
  // CHECK-NOT: scf.if
  // CHECK: return %[[CST]] : i32

  %cst_1 = arith.constant 1 : i32
  %result = scf.for %i0 = %arg0 to %arg1 step %arg2 iter_args(%si = %cst_1) -> (i32) {
    %cst_20 = arith.constant 20 : i32
    %cond = arith.cmpi ult, %si, %cst_20 : i32
    %inner_res = scf.if %cond -> (i32) {
      %1 = arith.constant 1 : i32
      scf.yield %1 : i32
    } else {
      %si_inc = arith.addi %si, %cst_1 : i32
      scf.yield %si_inc : i32
    }
    scf.yield %inner_res : i32
  }
  return %result : i32
}

/// Test that we can properly visit region successors when the terminator
/// implements the RegionBranchTerminatorOpInterface.

// CHECK-LABEL: func @loop_region_branch_terminator_op(
func.func @loop_region_branch_terminator_op(%arg1 : i32) {
  // CHECK:      %c2_i32 = arith.constant 2 : i32
  // CHECK-NEXT: return

  %c2_i32 = arith.constant 2 : i32
   %0 = scf.while (%arg2 = %c2_i32) : (i32) -> (i32) {
    %1 = arith.cmpi slt, %arg2, %arg1 : i32
    scf.condition(%1) %arg2 : i32
  } do {
  ^bb0(%arg2: i32):
    scf.yield %arg2 : i32
  }
  return
}

/// Check that propgation happens for affine.for -- tests its region branch op
/// interface as well.

// CHECK-LABEL: func @affine_loop_one_iter(
func.func @affine_loop_one_iter() -> i32 {
  // CHECK: %[[C1:.*]] = arith.constant 1 : i32
  %s0 = arith.constant 0 : i32
  %s1 = arith.constant 1 : i32
  %result = affine.for %i = 0 to 1 iter_args(%si = %s0) -> (i32) {
    %sn = arith.addi %si, %s1 : i32
    affine.yield %sn : i32
  }
  // CHECK: return %[[C1]] : i32
  return %result : i32
}

// CHECK-LABEL: func @affine_loop_zero_iter(
func.func @affine_loop_zero_iter() -> i32 {
  // CHECK: %[[C1:.*]] = arith.constant 1 : i32
  %s1 = arith.constant 1 : i32
  %result = affine.for %i = 0 to 0 iter_args(%si = %s1) -> (i32) {
   %sn = arith.addi %si, %si : i32
   affine.yield %sn : i32
  }
  // CHECK: return %[[C1]] : i32
  return %result : i32
}

// CHECK-LABEL: func @affine_loop_unknown_trip_count(
func.func @affine_loop_unknown_trip_count(%ub: index) -> i32 {
  // CHECK: %[[C0:.*]] = arith.constant 0 : i32
  %s0 = arith.constant 0 : i32
  %result = affine.for %i = 0 to %ub iter_args(%si = %s0) -> (i32) {
   %sn = arith.addi %si, %si : i32
   affine.yield %sn : i32
  }
  // CHECK: return %[[C0]] : i32
  return %result : i32
}

// CHECK-LABEL: func @while_loop_different_arg_count
func.func @while_loop_different_arg_count() -> index {
  // CHECK-DAG: %[[TRUE:.*]] = arith.constant true
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK: %[[WHILE:.*]] = scf.while
  %0 = scf.while (%arg3 = %c0, %arg4 = %c1) : (index, index) -> index {
    %1 = arith.cmpi slt, %arg3, %c1 : index
    // CHECK: scf.condition(%[[TRUE]]) %[[C1]]
    scf.condition(%1) %arg4 : index
  } do {
  ^bb0(%arg3: index):
    %1 = arith.muli %arg3, %c1 : index
    // CHECK: scf.yield %[[C0]], %[[C1]]
    scf.yield %c0, %1 : index, index
  }
  // CHECK: return %[[WHILE]]
  return %0 : index
}

// CHECK-LABEL: func @while_loop_false_condition
func.func @while_loop_false_condition(%arg0 : index) -> index {
  // CHECK: %[[C0:.*]] = arith.constant 0
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = arith.muli %arg0, %c0 : index
  %1 = scf.while (%arg1 = %0) : (index) -> index {
    %2 = arith.cmpi slt, %arg1, %c0 : index
    scf.condition(%2) %arg1 : index
  } do {
  ^bb0(%arg2 : index):
    %3 = arith.addi %arg2, %c1 : index
    scf.yield %3 : index
  }
  // CHECK: return %[[C0]]
  func.return %1 : index
}
