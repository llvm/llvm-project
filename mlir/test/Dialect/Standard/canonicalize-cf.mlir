// RUN: mlir-opt %s -allow-unregistered-dialect -pass-pipeline='func(canonicalize)' -split-input-file | FileCheck %s

// Test the folding of BranchOp.

// CHECK-LABEL: func @br_folding(
func @br_folding() -> i32 {
  // CHECK-NEXT: %[[CST:.*]] = constant 0 : i32
  // CHECK-NEXT: return %[[CST]] : i32
  %c0_i32 = constant 0 : i32
  br ^bb1(%c0_i32 : i32)
^bb1(%x : i32):
  return %x : i32
}

// Test the folding of CondBranchOp with a constant condition.

// CHECK-LABEL: func @cond_br_folding(
func @cond_br_folding(%cond : i1, %a : i32) {
  // CHECK-NEXT: cond_br %{{.*}}, ^bb1, ^bb1

  %false_cond = constant 0 : i1
  %true_cond = constant 1 : i1
  cond_br %cond, ^bb1, ^bb2(%a : i32)

^bb1:
  cond_br %true_cond, ^bb3, ^bb2(%a : i32)

^bb2(%x : i32):
  cond_br %false_cond, ^bb2(%x : i32), ^bb3

^bb3:
  // CHECK: ^bb1:
  // CHECK-NEXT: return

  return
}

// Test the compound folding of BranchOp and CondBranchOp.

// CHECK-LABEL: func @cond_br_and_br_folding(
func @cond_br_and_br_folding(%a : i32) {
  // CHECK-NEXT: return

  %false_cond = constant 0 : i1
  %true_cond = constant 1 : i1
  cond_br %true_cond, ^bb2, ^bb1(%a : i32)

^bb1(%x : i32):
  cond_br %false_cond, ^bb1(%x : i32), ^bb2

^bb2:
  return
}

/// Test that pass-through successors of CondBranchOp get folded.

// CHECK-LABEL: func @cond_br_pass_through(
// CHECK-SAME: %[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32, %[[ARG2:.*]]: i32
func @cond_br_pass_through(%arg0 : i32, %arg1 : i32, %arg2 : i32, %cond : i1) -> (i32, i32) {
  // CHECK: cond_br %{{.*}}, ^bb1(%[[ARG0]], %[[ARG1]] : i32, i32), ^bb1(%[[ARG2]], %[[ARG2]] : i32, i32)

  cond_br %cond, ^bb1(%arg0 : i32), ^bb2(%arg2, %arg2 : i32, i32)

^bb1(%arg3: i32):
  br ^bb2(%arg3, %arg1 : i32, i32)

^bb2(%arg4: i32, %arg5: i32):
  // CHECK: ^bb1(%[[RET0:.*]]: i32, %[[RET1:.*]]: i32):
  // CHECK-NEXT: return %[[RET0]], %[[RET1]]

  return %arg4, %arg5 : i32, i32
}

/// Test the failure modes of collapsing CondBranchOp pass-throughs successors.

// CHECK-LABEL: func @cond_br_pass_through_fail(
func @cond_br_pass_through_fail(%cond : i1) {
  // CHECK: cond_br %{{.*}}, ^bb1, ^bb2

  cond_br %cond, ^bb1, ^bb2

^bb1:
  // CHECK: ^bb1:
  // CHECK: "foo.op"
  // CHECK: br ^bb2

  // Successors can't be collapsed if they contain other operations.
  "foo.op"() : () -> ()
  br ^bb2

^bb2:
  return
}
