// RUN: mlir-opt %s --canonicalize --split-input-file | FileCheck %s

// Entry and backedge values differ, so live block arguments cannot fold.
// Check operand order, block arguments, and segment sizes on the parent and
// on both terminators. The control operands must remain in place.

// CHECK-LABEL: func.func @remove_from_both_segments(
// CHECK-SAME: %[[CONDITION:[^ ,)]+]]: i1, %[[STOP:[^ ,)]+]]: i1, %[[A:[^ ,)]+]]: i32, %[[B:[^ ,)]+]]: i32, %[[C:[^ ,)]+]]: i32, %[[D:[^ ,)]+]]: i32, %[[E:[^ ,)]+]]: i32, %[[F:[^ ,)]+]]: i32) {
// CHECK: "test.segmented_region_branch"(%[[CONDITION]], %[[A]], %[[C]], %[[E]], %[[STOP]]) <{operandSegmentSizes = array<i32: 1, 2, 1, 1>}> ({
// CHECK: ^bb0(%[[R0_0:[^ ,)]+]]: i32, %[[R0_2:[^ ,)]+]]: i32):
// CHECK: call @use2(%[[R0_0]], %[[R0_2]])
// CHECK: "test.segmented_region_branch_terminator"(%[[CONDITION]], %[[D]], %[[F]], %[[B]], %[[STOP]]) <{operandSegmentSizes = array<i32: 1, 2, 1, 1>}>
// CHECK: ^bb0(%[[R1_1:[^ ,)]+]]: i32):
// CHECK: call @use1(%[[R1_1]])
// CHECK: "test.segmented_region_branch_terminator"(%[[CONDITION]], %[[D]], %[[F]], %[[B]], %[[STOP]]) <{operandSegmentSizes = array<i32: 1, 2, 1, 1>}>
// CHECK: })
func.func @remove_from_both_segments(%condition: i1, %stop: i1, %a: i32, %b: i32, %c: i32, %d: i32, %e: i32, %f: i32) {
  "test.segmented_region_branch"(%condition, %a, %b, %c, %d, %e, %f, %stop) <{operandSegmentSizes = array<i32: 1, 3, 3, 1>}> ({
  ^bb0(%r0_0: i32, %r0_1: i32, %r0_2: i32):
    func.call @use2(%r0_0, %r0_2) : (i32, i32) -> ()
    "test.segmented_region_branch_terminator"(%condition, %d, %e, %f, %a, %b, %c, %stop) <{operandSegmentSizes = array<i32: 1, 3, 3, 1>}> : (i1, i32, i32, i32, i32, i32, i32, i1) -> ()
  }, {
  ^bb0(%r1_0: i32, %r1_1: i32, %r1_2: i32):
    func.call @use1(%r1_1) : (i32) -> ()
    "test.segmented_region_branch_terminator"(%condition, %d, %e, %f, %a, %b, %c, %stop) <{operandSegmentSizes = array<i32: 1, 3, 3, 1>}> : (i1, i32, i32, i32, i32, i32, i32, i1) -> ()
  }) : (i1, i32, i32, i32, i32, i32, i32, i1) -> ()
  return
}
func.func private @use1(i32)
func.func private @use2(i32, i32)

// -----

// CHECK-LABEL: func.func @remove_entire_segment(
// CHECK-SAME: %[[CONDITION:[^ ,)]+]]: i1, %[[STOP:[^ ,)]+]]: i1, %[[A:[^ ,)]+]]: i32, %[[B:[^ ,)]+]]: i32, %[[C:[^ ,)]+]]: i32, %[[D:[^ ,)]+]]: i32, %[[E:[^ ,)]+]]: i32, %[[F:[^ ,)]+]]: i32) {
// CHECK: "test.segmented_region_branch"(%[[CONDITION]], %[[D]], %[[F]], %[[STOP]]) <{operandSegmentSizes = array<i32: 1, 0, 2, 1>}> ({
// CHECK-NOT: ^bb
// CHECK: "test.segmented_region_branch_terminator"(%[[CONDITION]], %[[A]], %[[C]], %[[STOP]]) <{operandSegmentSizes = array<i32: 1, 0, 2, 1>}>
// CHECK: ^bb0(%[[R1_0:[^ ,)]+]]: i32, %[[R1_2:[^ ,)]+]]: i32):
// CHECK: call @use2(%[[R1_0]], %[[R1_2]])
// CHECK: "test.segmented_region_branch_terminator"(%[[CONDITION]], %[[A]], %[[C]], %[[STOP]]) <{operandSegmentSizes = array<i32: 1, 0, 2, 1>}>
// CHECK: })
func.func @remove_entire_segment(%condition: i1, %stop: i1, %a: i32, %b: i32, %c: i32, %d: i32, %e: i32, %f: i32) {
  "test.segmented_region_branch"(%condition, %a, %b, %c, %d, %e, %f, %stop) <{operandSegmentSizes = array<i32: 1, 3, 3, 1>}> ({
  ^bb0(%r0_0: i32, %r0_1: i32, %r0_2: i32):
    "test.segmented_region_branch_terminator"(%condition, %d, %e, %f, %a, %b, %c, %stop) <{operandSegmentSizes = array<i32: 1, 3, 3, 1>}> : (i1, i32, i32, i32, i32, i32, i32, i1) -> ()
  }, {
  ^bb0(%r1_0: i32, %r1_1: i32, %r1_2: i32):
    func.call @use2(%r1_0, %r1_2) : (i32, i32) -> ()
    "test.segmented_region_branch_terminator"(%condition, %d, %e, %f, %a, %b, %c, %stop) <{operandSegmentSizes = array<i32: 1, 3, 3, 1>}> : (i1, i32, i32, i32, i32, i32, i32, i1) -> ()
  }) : (i1, i32, i32, i32, i32, i32, i32, i1) -> ()
  return
}
func.func private @use2(i32, i32)

// -----

// CHECK-LABEL: func.func @remove_all_successor_operands(
// CHECK-SAME: %[[CONDITION:[^ ,)]+]]: i1, %[[STOP:[^ ,)]+]]: i1, %[[A:[^ ,)]+]]: i32, %[[B:[^ ,)]+]]: i32, %[[C:[^ ,)]+]]: i32, %[[D:[^ ,)]+]]: i32, %[[E:[^ ,)]+]]: i32, %[[F:[^ ,)]+]]: i32) {
// CHECK: "test.segmented_region_branch"(%[[CONDITION]], %[[STOP]]) <{operandSegmentSizes = array<i32: 1, 0, 0, 1>}> ({
// CHECK-NOT: ^bb
// CHECK: "test.segmented_region_branch_terminator"(%[[CONDITION]], %[[STOP]]) <{operandSegmentSizes = array<i32: 1, 0, 0, 1>}>
// CHECK-NOT: ^bb
// CHECK: "test.segmented_region_branch_terminator"(%[[CONDITION]], %[[STOP]]) <{operandSegmentSizes = array<i32: 1, 0, 0, 1>}>
// CHECK: })
func.func @remove_all_successor_operands(%condition: i1, %stop: i1, %a: i32, %b: i32, %c: i32, %d: i32, %e: i32, %f: i32) {
  "test.segmented_region_branch"(%condition, %a, %b, %c, %d, %e, %f, %stop) <{operandSegmentSizes = array<i32: 1, 3, 3, 1>}> ({
  ^bb0(%r0_0: i32, %r0_1: i32, %r0_2: i32):
    "test.segmented_region_branch_terminator"(%condition, %d, %e, %f, %a, %b, %c, %stop) <{operandSegmentSizes = array<i32: 1, 3, 3, 1>}> : (i1, i32, i32, i32, i32, i32, i32, i1) -> ()
  }, {
  ^bb0(%r1_0: i32, %r1_1: i32, %r1_2: i32):
    "test.segmented_region_branch_terminator"(%condition, %d, %e, %f, %a, %b, %c, %stop) <{operandSegmentSizes = array<i32: 1, 3, 3, 1>}> : (i1, i32, i32, i32, i32, i32, i32, i1) -> ()
  }) : (i1, i32, i32, i32, i32, i32, i32, i1) -> ()
  return
}

// -----

// CHECK-LABEL: func.func @empty_segments_and_absent_control(
// CHECK-SAME: %[[CONDITION:[^ ,)]+]]: i1, %[[STOP:[^ ,)]+]]: i1, %[[A:[^ ,)]+]]: i32, %[[B:[^ ,)]+]]: i32, %[[C:[^ ,)]+]]: i32, %[[D:[^ ,)]+]]: i32, %[[E:[^ ,)]+]]: i32, %[[F:[^ ,)]+]]: i32) {
// CHECK: "test.segmented_region_branch"(%[[CONDITION]], %[[E]]) <{operandSegmentSizes = array<i32: 1, 0, 1, 0>}> ({
// CHECK-NOT: ^bb
// CHECK: "test.segmented_region_branch_terminator"(%[[CONDITION]], %[[B]]) <{operandSegmentSizes = array<i32: 1, 0, 1, 0>}>
// CHECK: ^bb0(%[[R1_1:[^ ,)]+]]: i32):
// CHECK: call @use1(%[[R1_1]])
// CHECK: "test.segmented_region_branch_terminator"(%[[CONDITION]], %[[B]]) <{operandSegmentSizes = array<i32: 1, 0, 1, 0>}>
// CHECK: })
func.func @empty_segments_and_absent_control(%condition: i1, %stop: i1, %a: i32, %b: i32, %c: i32, %d: i32, %e: i32, %f: i32) {
  "test.segmented_region_branch"(%condition, %d, %e, %f) <{operandSegmentSizes = array<i32: 1, 0, 3, 0>}> ({
  ^bb0:
    "test.segmented_region_branch_terminator"(%condition, %a, %b, %c) <{operandSegmentSizes = array<i32: 1, 0, 3, 0>}> : (i1, i32, i32, i32) -> ()
  }, {
  ^bb0(%r1_0: i32, %r1_1: i32, %r1_2: i32):
    func.call @use1(%r1_1) : (i32) -> ()
    "test.segmented_region_branch_terminator"(%condition, %a, %b, %c) <{operandSegmentSizes = array<i32: 1, 0, 3, 0>}> : (i1, i32, i32, i32) -> ()
  }) : (i1, i32, i32, i32) -> ()
  return
}
func.func private @use1(i32)

// -----

// CHECK-LABEL: func.func @retain_all_operands(
// CHECK-SAME: %[[CONDITION:[^ ,)]+]]: i1, %[[STOP:[^ ,)]+]]: i1, %[[A:[^ ,)]+]]: i32, %[[B:[^ ,)]+]]: i32, %[[C:[^ ,)]+]]: i32, %[[D:[^ ,)]+]]: i32, %[[E:[^ ,)]+]]: i32, %[[F:[^ ,)]+]]: i32) {
// CHECK: "test.segmented_region_branch"(%[[CONDITION]], %[[A]], %[[B]], %[[C]], %[[D]], %[[E]], %[[F]]) <{operandSegmentSizes = array<i32: 1, 3, 3, 0>}> ({
// CHECK: ^bb0(%[[R0_0:[^ ,)]+]]: i32, %[[R0_1:[^ ,)]+]]: i32, %[[R0_2:[^ ,)]+]]: i32):
// CHECK: call @use3(%[[R0_0]], %[[R0_1]], %[[R0_2]])
// CHECK: "test.segmented_region_branch_terminator"(%[[CONDITION]], %[[D]], %[[E]], %[[F]], %[[A]], %[[B]], %[[C]]) <{operandSegmentSizes = array<i32: 1, 3, 3, 0>}>
// CHECK: ^bb0(%[[R1_0:[^ ,)]+]]: i32, %[[R1_1:[^ ,)]+]]: i32, %[[R1_2:[^ ,)]+]]: i32):
// CHECK: call @use3(%[[R1_0]], %[[R1_1]], %[[R1_2]])
// CHECK: "test.segmented_region_branch_terminator"(%[[CONDITION]], %[[D]], %[[E]], %[[F]], %[[A]], %[[B]], %[[C]]) <{operandSegmentSizes = array<i32: 1, 3, 3, 0>}>
// CHECK: })
func.func @retain_all_operands(%condition: i1, %stop: i1, %a: i32, %b: i32, %c: i32, %d: i32, %e: i32, %f: i32) {
  "test.segmented_region_branch"(%condition, %a, %b, %c, %d, %e, %f) <{operandSegmentSizes = array<i32: 1, 3, 3, 0>}> ({
  ^bb0(%r0_0: i32, %r0_1: i32, %r0_2: i32):
    func.call @use3(%r0_0, %r0_1, %r0_2) : (i32, i32, i32) -> ()
    "test.segmented_region_branch_terminator"(%condition, %d, %e, %f, %a, %b, %c) <{operandSegmentSizes = array<i32: 1, 3, 3, 0>}> : (i1, i32, i32, i32, i32, i32, i32) -> ()
  }, {
  ^bb0(%r1_0: i32, %r1_1: i32, %r1_2: i32):
    func.call @use3(%r1_0, %r1_1, %r1_2) : (i32, i32, i32) -> ()
    "test.segmented_region_branch_terminator"(%condition, %d, %e, %f, %a, %b, %c) <{operandSegmentSizes = array<i32: 1, 3, 3, 0>}> : (i1, i32, i32, i32, i32, i32, i32) -> ()
  }) : (i1, i32, i32, i32, i32, i32, i32) -> ()
  return
}
func.func private @use3(i32, i32, i32)
