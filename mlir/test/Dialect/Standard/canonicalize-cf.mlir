// RUN: mlir-opt %s -allow-unregistered-dialect -pass-pipeline='func(canonicalize)' -split-input-file | FileCheck %s

/// Test the folding of BranchOp.

// CHECK-LABEL: func @br_folding(
func @br_folding() -> i32 {
  // CHECK-NEXT: %[[CST:.*]] = constant 0 : i32
  // CHECK-NEXT: return %[[CST]] : i32
  %c0_i32 = constant 0 : i32
  br ^bb1(%c0_i32 : i32)
^bb1(%x : i32):
  return %x : i32
}

/// Test that pass-through successors of BranchOp get folded.

// CHECK-LABEL: func @br_passthrough(
// CHECK-SAME: %[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32
func @br_passthrough(%arg0 : i32, %arg1 : i32) -> (i32, i32) {
  "foo.switch"() [^bb1, ^bb2, ^bb3] : () -> ()

^bb1:
  // CHECK: ^bb1:
  // CHECK-NEXT: br ^bb3(%[[ARG0]], %[[ARG1]] : i32, i32)

  br ^bb2(%arg0 : i32)

^bb2(%arg2 : i32):
  br ^bb3(%arg2, %arg1 : i32, i32)

^bb3(%arg4 : i32, %arg5 : i32):
  return %arg4, %arg5 : i32, i32
}

/// Test the folding of CondBranchOp with a constant condition.

// CHECK-LABEL: func @cond_br_folding(
func @cond_br_folding(%cond : i1, %a : i32) {
  // CHECK-NEXT: return

  %false_cond = constant false
  %true_cond = constant true
  cond_br %cond, ^bb1, ^bb2(%a : i32)

^bb1:
  cond_br %true_cond, ^bb3, ^bb2(%a : i32)

^bb2(%x : i32):
  cond_br %false_cond, ^bb2(%x : i32), ^bb3

^bb3:
  return
}

/// Test the folding of CondBranchOp when the successors are identical.

// CHECK-LABEL: func @cond_br_same_successor(
func @cond_br_same_successor(%cond : i1, %a : i32) {
  // CHECK-NEXT: return

  cond_br %cond, ^bb1(%a : i32), ^bb1(%a : i32)

^bb1(%result : i32):
  return
}

/// Test the folding of CondBranchOp when the successors are identical, but the
/// arguments are different.

// CHECK-LABEL: func @cond_br_same_successor_insert_select(
// CHECK-SAME: %[[COND:.*]]: i1, %[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32
// CHECK-SAME: %[[ARG2:.*]]: tensor<2xi32>, %[[ARG3:.*]]: tensor<2xi32>
func @cond_br_same_successor_insert_select(
      %cond : i1, %a : i32, %b : i32, %c : tensor<2xi32>, %d : tensor<2xi32>
    ) -> (i32, tensor<2xi32>)  {
  // CHECK: %[[RES:.*]] = select %[[COND]], %[[ARG0]], %[[ARG1]]
  // CHECK: %[[RES2:.*]] = select %[[COND]], %[[ARG2]], %[[ARG3]]
  // CHECK: return %[[RES]], %[[RES2]]

  cond_br %cond, ^bb1(%a, %c : i32, tensor<2xi32>), ^bb1(%b, %d : i32, tensor<2xi32>)

^bb1(%result : i32, %result2 : tensor<2xi32>):
  return %result, %result2 : i32, tensor<2xi32>
}

/// Test the compound folding of BranchOp and CondBranchOp.

// CHECK-LABEL: func @cond_br_and_br_folding(
func @cond_br_and_br_folding(%a : i32) {
  // CHECK-NEXT: return

  %false_cond = constant false
  %true_cond = constant true
  cond_br %true_cond, ^bb2, ^bb1(%a : i32)

^bb1(%x : i32):
  cond_br %false_cond, ^bb1(%x : i32), ^bb2

^bb2:
  return
}

/// Test that pass-through successors of CondBranchOp get folded.

// CHECK-LABEL: func @cond_br_passthrough(
// CHECK-SAME: %[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32, %[[ARG2:.*]]: i32, %[[COND:.*]]: i1
func @cond_br_passthrough(%arg0 : i32, %arg1 : i32, %arg2 : i32, %cond : i1) -> (i32, i32) {
  // CHECK: %[[RES:.*]] = select %[[COND]], %[[ARG0]], %[[ARG2]]
  // CHECK: %[[RES2:.*]] = select %[[COND]], %[[ARG1]], %[[ARG2]]
  // CHECK: return %[[RES]], %[[RES2]]

  cond_br %cond, ^bb1(%arg0 : i32), ^bb2(%arg2, %arg2 : i32, i32)

^bb1(%arg3: i32):
  br ^bb2(%arg3, %arg1 : i32, i32)

^bb2(%arg4: i32, %arg5: i32):
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

// -----

// Erase assertion if condition is known to be true at compile time.
// CHECK-LABEL: @assert_true
func @assert_true() {
  // CHECK-NOT: assert
  %true = constant true
  assert %true, "Computer says no"
  return
}

// -----

// Keep assertion if condition unknown at compile time.
// CHECK-LABEL: @assert
// CHECK-SAME:  (%[[ARG:.*]]: i1)
func @assert(%arg : i1) {
  // CHECK: assert %[[ARG]], "Computer says no"
  assert %arg, "Computer says no"
  return
}

