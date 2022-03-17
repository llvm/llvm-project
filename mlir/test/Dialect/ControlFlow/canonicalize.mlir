// RUN: mlir-opt %s -allow-unregistered-dialect -pass-pipeline='func.func(canonicalize)' -split-input-file | FileCheck --dump-input-context 20 %s

/// Test the folding of BranchOp.

// CHECK-LABEL: func @br_folding(
func @br_folding() -> i32 {
  // CHECK-NEXT: %[[CST:.*]] = arith.constant 0 : i32
  // CHECK-NEXT: return %[[CST]] : i32
  %c0_i32 = arith.constant 0 : i32
  cf.br ^bb1(%c0_i32 : i32)
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
  // CHECK-NEXT: cf.br ^bb3(%[[ARG0]], %[[ARG1]] : i32, i32)

  cf.br ^bb2(%arg0 : i32)

^bb2(%arg2 : i32):
  cf.br ^bb3(%arg2, %arg1 : i32, i32)

^bb3(%arg4 : i32, %arg5 : i32):
  return %arg4, %arg5 : i32, i32
}

/// Test the folding of CondBranchOp with a constant condition.

// CHECK-LABEL: func @cond_br_folding(
func @cond_br_folding(%cond : i1, %a : i32) {
  // CHECK-NEXT: return

  %false_cond = arith.constant false
  %true_cond = arith.constant true
  cf.cond_br %cond, ^bb1, ^bb2(%a : i32)

^bb1:
  cf.cond_br %true_cond, ^bb3, ^bb2(%a : i32)

^bb2(%x : i32):
  cf.cond_br %false_cond, ^bb2(%x : i32), ^bb3

^bb3:
  return
}

/// Test the folding of CondBranchOp when the successors are identical.

// CHECK-LABEL: func @cond_br_same_successor(
func @cond_br_same_successor(%cond : i1, %a : i32) {
  // CHECK-NEXT: return

  cf.cond_br %cond, ^bb1(%a : i32), ^bb1(%a : i32)

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
  // CHECK: %[[RES:.*]] = arith.select %[[COND]], %[[ARG0]], %[[ARG1]]
  // CHECK: %[[RES2:.*]] = arith.select %[[COND]], %[[ARG2]], %[[ARG3]]
  // CHECK: return %[[RES]], %[[RES2]]

  cf.cond_br %cond, ^bb1(%a, %c : i32, tensor<2xi32>), ^bb1(%b, %d : i32, tensor<2xi32>)

^bb1(%result : i32, %result2 : tensor<2xi32>):
  return %result, %result2 : i32, tensor<2xi32>
}

/// Test the compound folding of BranchOp and CondBranchOp.

// CHECK-LABEL: func @cond_br_and_br_folding(
func @cond_br_and_br_folding(%a : i32) {
  // CHECK-NEXT: return

  %false_cond = arith.constant false
  %true_cond = arith.constant true
  cf.cond_br %true_cond, ^bb2, ^bb1(%a : i32)

^bb1(%x : i32):
  cf.cond_br %false_cond, ^bb1(%x : i32), ^bb2

^bb2:
  return
}

/// Test that pass-through successors of CondBranchOp get folded.

// CHECK-LABEL: func @cond_br_passthrough(
// CHECK-SAME: %[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32, %[[ARG2:.*]]: i32, %[[COND:.*]]: i1
func @cond_br_passthrough(%arg0 : i32, %arg1 : i32, %arg2 : i32, %cond : i1) -> (i32, i32) {
  // CHECK: %[[RES:.*]] = arith.select %[[COND]], %[[ARG0]], %[[ARG2]]
  // CHECK: %[[RES2:.*]] = arith.select %[[COND]], %[[ARG1]], %[[ARG2]]
  // CHECK: return %[[RES]], %[[RES2]]

  cf.cond_br %cond, ^bb1(%arg0 : i32), ^bb2(%arg2, %arg2 : i32, i32)

^bb1(%arg3: i32):
  cf.br ^bb2(%arg3, %arg1 : i32, i32)

^bb2(%arg4: i32, %arg5: i32):
  return %arg4, %arg5 : i32, i32
}

/// Test the failure modes of collapsing CondBranchOp pass-throughs successors.

// CHECK-LABEL: func @cond_br_pass_through_fail(
func @cond_br_pass_through_fail(%cond : i1) {
  // CHECK: cf.cond_br %{{.*}}, ^bb1, ^bb2

  cf.cond_br %cond, ^bb1, ^bb2

^bb1:
  // CHECK: ^bb1:
  // CHECK: "foo.op"
  // CHECK: cf.br ^bb2

  // Successors can't be collapsed if they contain other operations.
  "foo.op"() : () -> ()
  cf.br ^bb2

^bb2:
  return
}


/// Test the folding of SwitchOp

// CHECK-LABEL: func @switch_only_default(
// CHECK-SAME: %[[FLAG:[a-zA-Z0-9_]+]]
// CHECK-SAME: %[[CASE_OPERAND_0:[a-zA-Z0-9_]+]]
func @switch_only_default(%flag : i32, %caseOperand0 : f32) {
  // add predecessors for all blocks to avoid other canonicalizations.
  "foo.pred"() [^bb1, ^bb2] : () -> ()
  ^bb1:
    // CHECK-NOT: cf.switch
    // CHECK: cf.br ^[[BB2:[a-zA-Z0-9_]+]](%[[CASE_OPERAND_0]]
    cf.switch %flag : i32, [
      default: ^bb2(%caseOperand0 : f32)
    ]
  // CHECK: ^[[BB2]]({{.*}}):
  ^bb2(%bb2Arg : f32):
    // CHECK-NEXT: "foo.bb2Terminator"
    "foo.bb2Terminator"(%bb2Arg) : (f32) -> ()
}


// CHECK-LABEL: func @switch_case_matching_default(
// CHECK-SAME: %[[FLAG:[a-zA-Z0-9_]+]]
// CHECK-SAME: %[[CASE_OPERAND_0:[a-zA-Z0-9_]+]]
// CHECK-SAME: %[[CASE_OPERAND_1:[a-zA-Z0-9_]+]]
func @switch_case_matching_default(%flag : i32, %caseOperand0 : f32, %caseOperand1 : f32) {
  // add predecessors for all blocks to avoid other canonicalizations.
  "foo.pred"() [^bb1, ^bb2, ^bb3] : () -> ()
  ^bb1:
    // CHECK: cf.switch %[[FLAG]]
    // CHECK-NEXT:   default: ^[[BB1:.+]](%[[CASE_OPERAND_0]] : f32)
    // CHECK-NEXT:   10: ^[[BB2:.+]](%[[CASE_OPERAND_1]] : f32)
    // CHECK-NEXT: ]
    cf.switch %flag : i32, [
      default: ^bb2(%caseOperand0 : f32),
      42: ^bb2(%caseOperand0 : f32),
      10: ^bb3(%caseOperand1 : f32),
      17: ^bb2(%caseOperand0 : f32)
    ]
  ^bb2(%bb2Arg : f32):
    "foo.bb2Terminator"(%bb2Arg) : (f32) -> ()
  ^bb3(%bb3Arg : f32):
    "foo.bb3Terminator"(%bb3Arg) : (f32) -> ()
}


// CHECK-LABEL: func @switch_on_const_no_match(
// CHECK-SAME: %[[CASE_OPERAND_0:[a-zA-Z0-9_]+]]
// CHECK-SAME: %[[CASE_OPERAND_1:[a-zA-Z0-9_]+]]
// CHECK-SAME: %[[CASE_OPERAND_2:[a-zA-Z0-9_]+]]
func @switch_on_const_no_match(%caseOperand0 : f32, %caseOperand1 : f32, %caseOperand2 : f32) {
  // add predecessors for all blocks to avoid other canonicalizations.
  "foo.pred"() [^bb1, ^bb2, ^bb3, ^bb4] : () -> ()
  ^bb1:
    // CHECK-NOT: cf.switch
    // CHECK: cf.br ^[[BB2:[a-zA-Z0-9_]+]](%[[CASE_OPERAND_0]]
    %c0_i32 = arith.constant 0 : i32
    cf.switch %c0_i32 : i32, [
      default: ^bb2(%caseOperand0 : f32),
      -1: ^bb3(%caseOperand1 : f32),
      1: ^bb4(%caseOperand2 : f32)
    ]
  // CHECK: ^[[BB2]]({{.*}}):
  // CHECK-NEXT: "foo.bb2Terminator"
  ^bb2(%bb2Arg : f32):
    "foo.bb2Terminator"(%bb2Arg) : (f32) -> ()
  ^bb3(%bb3Arg : f32):
    "foo.bb3Terminator"(%bb3Arg) : (f32) -> ()
  ^bb4(%bb4Arg : f32):
    "foo.bb4Terminator"(%bb4Arg) : (f32) -> ()
}

// CHECK-LABEL: func @switch_on_const_with_match(
// CHECK-SAME: %[[CASE_OPERAND_0:[a-zA-Z0-9_]+]]
// CHECK-SAME: %[[CASE_OPERAND_1:[a-zA-Z0-9_]+]]
// CHECK-SAME: %[[CASE_OPERAND_2:[a-zA-Z0-9_]+]]
func @switch_on_const_with_match(%caseOperand0 : f32, %caseOperand1 : f32, %caseOperand2 : f32) {
  // add predecessors for all blocks to avoid other canonicalizations.
  "foo.pred"() [^bb1, ^bb2, ^bb3, ^bb4] : () -> ()
  ^bb1:
    // CHECK-NOT: cf.switch
    // CHECK: cf.br ^[[BB4:[a-zA-Z0-9_]+]](%[[CASE_OPERAND_2]]
    %c0_i32 = arith.constant 1 : i32
    cf.switch %c0_i32 : i32, [
      default: ^bb2(%caseOperand0 : f32),
      -1: ^bb3(%caseOperand1 : f32),
      1: ^bb4(%caseOperand2 : f32)
    ]
  ^bb2(%bb2Arg : f32):
    "foo.bb2Terminator"(%bb2Arg) : (f32) -> ()
  ^bb3(%bb3Arg : f32):
    "foo.bb3Terminator"(%bb3Arg) : (f32) -> ()
  // CHECK: ^[[BB4]]({{.*}}):
  // CHECK-NEXT: "foo.bb4Terminator"
  ^bb4(%bb4Arg : f32):
    "foo.bb4Terminator"(%bb4Arg) : (f32) -> ()
}

// CHECK-LABEL: func @switch_passthrough(
// CHECK-SAME: %[[FLAG:[a-zA-Z0-9_]+]]
// CHECK-SAME: %[[CASE_OPERAND_0:[a-zA-Z0-9_]+]]
// CHECK-SAME: %[[CASE_OPERAND_1:[a-zA-Z0-9_]+]]
// CHECK-SAME: %[[CASE_OPERAND_2:[a-zA-Z0-9_]+]]
// CHECK-SAME: %[[CASE_OPERAND_3:[a-zA-Z0-9_]+]]
func @switch_passthrough(%flag : i32,
                         %caseOperand0 : f32,
                         %caseOperand1 : f32,
                         %caseOperand2 : f32,
                         %caseOperand3 : f32) {
  // add predecessors for all blocks to avoid other canonicalizations.
  "foo.pred"() [^bb1, ^bb2, ^bb3, ^bb4, ^bb5, ^bb6] : () -> ()

  ^bb1:
  //      CHECK: cf.switch %[[FLAG]]
  // CHECK-NEXT:   default: ^[[BB5:[a-zA-Z0-9_]+]](%[[CASE_OPERAND_0]]
  // CHECK-NEXT:   43: ^[[BB6:[a-zA-Z0-9_]+]](%[[CASE_OPERAND_1]]
  // CHECK-NEXT:   44: ^[[BB4:[a-zA-Z0-9_]+]](%[[CASE_OPERAND_2]]
  // CHECK-NEXT: ]
    cf.switch %flag : i32, [
      default: ^bb2(%caseOperand0 : f32),
      43: ^bb3(%caseOperand1 : f32),
      44: ^bb4(%caseOperand2 : f32)
    ]
  ^bb2(%bb2Arg : f32):
    cf.br ^bb5(%bb2Arg : f32)
  ^bb3(%bb3Arg : f32):
    cf.br ^bb6(%bb3Arg : f32)
  ^bb4(%bb4Arg : f32):
    "foo.bb4Terminator"(%bb4Arg) : (f32) -> ()

  // CHECK: ^[[BB5]]({{.*}}):
  // CHECK-NEXT: "foo.bb5Terminator"
  ^bb5(%bb5Arg : f32):
    "foo.bb5Terminator"(%bb5Arg) : (f32) -> ()

  // CHECK: ^[[BB6]]({{.*}}):
  // CHECK-NEXT: "foo.bb6Terminator"
  ^bb6(%bb6Arg : f32):
    "foo.bb6Terminator"(%bb6Arg) : (f32) -> ()
}

// CHECK-LABEL: func @switch_from_switch_with_same_value_with_match(
// CHECK-SAME: %[[FLAG:[a-zA-Z0-9_]+]]
// CHECK-SAME: %[[CASE_OPERAND_0:[a-zA-Z0-9_]+]]
// CHECK-SAME: %[[CASE_OPERAND_1:[a-zA-Z0-9_]+]]
func @switch_from_switch_with_same_value_with_match(%flag : i32, %caseOperand0 : f32, %caseOperand1 : f32) {
  // add predecessors for all blocks except ^bb3 to avoid other canonicalizations.
  "foo.pred"() [^bb1, ^bb2, ^bb4, ^bb5] : () -> ()

  ^bb1:
    // CHECK: cf.switch %[[FLAG]]
    cf.switch %flag : i32, [
      default: ^bb2,
      42: ^bb3
    ]

  ^bb2:
    "foo.bb2Terminator"() : () -> ()
  ^bb3:
    // prevent this block from being simplified away
    "foo.op"() : () -> ()
    // CHECK-NOT: cf.switch %[[FLAG]]
    // CHECK: cf.br ^[[BB5:[a-zA-Z0-9_]+]](%[[CASE_OPERAND_1]]
    cf.switch %flag : i32, [
      default: ^bb4(%caseOperand0 : f32),
      42: ^bb5(%caseOperand1 : f32)
    ]

  ^bb4(%bb4Arg : f32):
    "foo.bb4Terminator"(%bb4Arg) : (f32) -> ()

  // CHECK: ^[[BB5]]({{.*}}):
  // CHECK-NEXT: "foo.bb5Terminator"
  ^bb5(%bb5Arg : f32):
    "foo.bb5Terminator"(%bb5Arg) : (f32) -> ()
}

// CHECK-LABEL: func @switch_from_switch_with_same_value_no_match(
// CHECK-SAME: %[[FLAG:[a-zA-Z0-9_]+]]
// CHECK-SAME: %[[CASE_OPERAND_0:[a-zA-Z0-9_]+]]
// CHECK-SAME: %[[CASE_OPERAND_1:[a-zA-Z0-9_]+]]
// CHECK-SAME: %[[CASE_OPERAND_2:[a-zA-Z0-9_]+]]
func @switch_from_switch_with_same_value_no_match(%flag : i32, %caseOperand0 : f32, %caseOperand1 : f32, %caseOperand2 : f32) {
  // add predecessors for all blocks except ^bb3 to avoid other canonicalizations.
  "foo.pred"() [^bb1, ^bb2, ^bb4, ^bb5, ^bb6] : () -> ()

  ^bb1:
    // CHECK: cf.switch %[[FLAG]]
    cf.switch %flag : i32, [
      default: ^bb2,
      42: ^bb3
    ]

  ^bb2:
    "foo.bb2Terminator"() : () -> ()
  ^bb3:
    "foo.op"() : () -> ()
    // CHECK-NOT: cf.switch %[[FLAG]]
    // CHECK: cf.br ^[[BB4:[a-zA-Z0-9_]+]](%[[CASE_OPERAND_0]]
    cf.switch %flag : i32, [
      default: ^bb4(%caseOperand0 : f32),
      0: ^bb5(%caseOperand1 : f32),
      43: ^bb6(%caseOperand2 : f32)
    ]

  // CHECK: ^[[BB4]]({{.*}})
  // CHECK-NEXT: "foo.bb4Terminator"
  ^bb4(%bb4Arg : f32):
    "foo.bb4Terminator"(%bb4Arg) : (f32) -> ()

  ^bb5(%bb5Arg : f32):
    "foo.bb5Terminator"(%bb5Arg) : (f32) -> ()

  ^bb6(%bb6Arg : f32):
    "foo.bb6Terminator"(%bb6Arg) : (f32) -> ()
}

// CHECK-LABEL: func @switch_from_switch_default_with_same_value(
// CHECK-SAME: %[[FLAG:[a-zA-Z0-9_]+]]
// CHECK-SAME: %[[CASE_OPERAND_0:[a-zA-Z0-9_]+]]
// CHECK-SAME: %[[CASE_OPERAND_1:[a-zA-Z0-9_]+]]
// CHECK-SAME: %[[CASE_OPERAND_2:[a-zA-Z0-9_]+]]
func @switch_from_switch_default_with_same_value(%flag : i32, %caseOperand0 : f32, %caseOperand1 : f32, %caseOperand2 : f32) {
  // add predecessors for all blocks except ^bb3 to avoid other canonicalizations.
  "foo.pred"() [^bb1, ^bb2, ^bb4, ^bb5, ^bb6] : () -> ()

  ^bb1:
    // CHECK: cf.switch %[[FLAG]]
    cf.switch %flag : i32, [
      default: ^bb3,
      42: ^bb2
    ]

  ^bb2:
    "foo.bb2Terminator"() : () -> ()
  ^bb3:
    "foo.op"() : () -> ()
    // CHECK: cf.switch %[[FLAG]]
    // CHECK-NEXT: default: ^[[BB4:[a-zA-Z0-9_]+]](%[[CASE_OPERAND_0]]
    // CHECK-NEXT: 43: ^[[BB6:[a-zA-Z0-9_]+]](%[[CASE_OPERAND_2]]
    // CHECK-NOT: 42
    cf.switch %flag : i32, [
      default: ^bb4(%caseOperand0 : f32),
      42: ^bb5(%caseOperand1 : f32),
      43: ^bb6(%caseOperand2 : f32)
    ]

  // CHECK: ^[[BB4]]({{.*}}):
  // CHECK-NEXT: "foo.bb4Terminator"
  ^bb4(%bb4Arg : f32):
    "foo.bb4Terminator"(%bb4Arg) : (f32) -> ()

  ^bb5(%bb5Arg : f32):
    "foo.bb5Terminator"(%bb5Arg) : (f32) -> ()

  // CHECK: ^[[BB6]]({{.*}}):
  // CHECK-NEXT: "foo.bb6Terminator"
  ^bb6(%bb6Arg : f32):
    "foo.bb6Terminator"(%bb6Arg) : (f32) -> ()
}

/// Test folding conditional branches that are successors of conditional
/// branches with the same condition.

// CHECK-LABEL: func @cond_br_from_cond_br_with_same_condition
func @cond_br_from_cond_br_with_same_condition(%cond : i1) {
  // CHECK:   cf.cond_br %{{.*}}, ^bb1, ^bb2
  // CHECK: ^bb1:
  // CHECK:   return

  cf.cond_br %cond, ^bb1, ^bb2

^bb1:
  cf.cond_br %cond, ^bb3, ^bb2

^bb2:
  "foo.terminator"() : () -> ()

^bb3:
  return
}

// -----

// Erase assertion if condition is known to be true at compile time.
// CHECK-LABEL: @assert_true
func @assert_true() {
  // CHECK-NOT: cf.assert
  %true = arith.constant true
  cf.assert %true, "Computer says no"
  return
}

// -----

// Keep assertion if condition unknown at compile time.
// CHECK-LABEL: @cf.assert
// CHECK-SAME:  (%[[ARG:.*]]: i1)
func @cf.assert(%arg : i1) {
  // CHECK: cf.assert %[[ARG]], "Computer says no"
  cf.assert %arg, "Computer says no"
  return
}

// -----

// CHECK-LABEL: @branchCondProp
//       CHECK:       %[[trueval:.+]] = arith.constant true
//       CHECK:       %[[falseval:.+]] = arith.constant false
//       CHECK:       "test.consumer1"(%[[trueval]]) : (i1) -> ()
//       CHECK:       "test.consumer2"(%[[falseval]]) : (i1) -> ()
func @branchCondProp(%arg0: i1) {
  cf.cond_br %arg0, ^trueB, ^falseB

^trueB:
  "test.consumer1"(%arg0) : (i1) -> ()
  cf.br ^exit

^falseB:
  "test.consumer2"(%arg0) : (i1) -> ()
  cf.br ^exit

^exit:
  return
}
