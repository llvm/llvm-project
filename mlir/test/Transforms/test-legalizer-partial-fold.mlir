// RUN: mlir-opt -split-input-file -test-legalize-patterns="allow-pattern-rollback=1" -verify-diagnostics %s | FileCheck %s --check-prefixes=CHECK,ROLLBACK
// RUN: mlir-opt -split-input-file -test-legalize-patterns="allow-pattern-rollback=0" -verify-diagnostics %s | FileCheck %s --check-prefixes=CHECK,NO-ROLLBACK

// The replacement of some results of an op cannot be rolled back. So only the
// "no rollback" mode applies a partial fold. The test ops are not in the
// conversion target, so each op stays and gets a remark in both modes.

// CHECK-LABEL: func @partial_fold
// CHECK-SAME: (%[[ARG0:[a-z0-9]+]]: i32)
func.func @partial_fold(%arg0: i32) -> (i32, i32, i32) {
  // ROLLBACK-NEXT: %[[RES:[a-z0-9]+]]:3 = "test.op_partial_fold"(%[[ARG0]])
  // ROLLBACK-NEXT: "test.return"(%[[RES]]#0, %[[RES]]#1, %[[RES]]#2)
  // NO-ROLLBACK-NEXT: %[[C42:[a-z0-9]+]] = "test.constant"() <{value = 42 : i32}> : () -> i32
  // NO-ROLLBACK-NEXT: %[[RES:[a-z0-9]+]]:3 = "test.op_partial_fold"(%[[ARG0]])
  // NO-ROLLBACK-NEXT: "test.return"(%[[C42]], %[[ARG0]], %[[RES]]#2)
  // expected-remark@+1 {{op 'test.op_partial_fold' is not legalizable}}
  %0:3 = "test.op_partial_fold"(%arg0) : (i32) -> (i32, i32, i32)
  "test.return"(%0#0, %0#1, %0#2) : (i32, i32, i32) -> ()
}

// -----

// The "rollback" mode applies only the in-place change of the fold.

// CHECK-LABEL: func @partial_fold_in_place
// CHECK-SAME: (%[[ARG0:[a-z0-9]+]]: i32)
func.func @partial_fold_in_place(%arg0: i32) -> (i32, i32) {
  // CHECK-NEXT: %[[FORWARDED:[a-z0-9_]+]], %[[KEPT:[a-z0-9_]+]] = "test.op_partial_fold_in_place"(%[[ARG0]]) <{folded}>
  // ROLLBACK-NEXT: "test.return"(%[[FORWARDED]], %[[KEPT]])
  // NO-ROLLBACK-NEXT: "test.return"(%[[ARG0]], %[[KEPT]])
  // expected-remark@+1 {{op 'test.op_partial_fold_in_place' is not legalizable}}
  %0:2 = "test.op_partial_fold_in_place"(%arg0) : (i32) -> (i32, i32)
  "test.return"(%0#0, %0#1) : (i32, i32) -> ()
}

// -----

// CHECK-LABEL: func @fold_keep_all
func.func @fold_keep_all() -> (i32, i32) {
  // CHECK-NEXT: %[[RES:[a-z0-9]+]]:2 = "test.op_fold_keep_all"()
  // CHECK-NEXT: "test.return"(%[[RES]]#0, %[[RES]]#1)
  // expected-remark@+1 {{op 'test.op_fold_keep_all' is not legalizable}}
  %0:2 = "test.op_fold_keep_all"() : () -> (i32, i32)
  "test.return"(%0#0, %0#1) : (i32, i32) -> ()
}

// -----

// The replaced results have no uses, so the fold makes no progress.

// CHECK-LABEL: func @partial_fold_dead_replaced_results
// CHECK-SAME: (%[[ARG0:[a-z0-9]+]]: i32)
func.func @partial_fold_dead_replaced_results(%arg0: i32) -> i32 {
  // CHECK-NEXT: %[[RES:[a-z0-9]+]]:3 = "test.op_partial_fold"(%[[ARG0]])
  // CHECK-NEXT: "test.return"(%[[RES]]#2)
  // expected-remark@+1 {{op 'test.op_partial_fold' is not legalizable}}
  %0:3 = "test.op_partial_fold"(%arg0) : (i32) -> (i32, i32, i32)
  "test.return"(%0#2) : (i32) -> ()
}

// -----

// CHECK-LABEL: func @partial_fold_unmaterializable
func.func @partial_fold_unmaterializable() -> (i32, i32, i32) {
  // CHECK-NEXT: %[[RES:[a-z0-9]+]]:3 = "test.op_partial_fold_unmaterializable"()
  // CHECK-NEXT: "test.return"(%[[RES]]#0, %[[RES]]#1, %[[RES]]#2)
  // expected-remark@+1 {{op 'test.op_partial_fold_unmaterializable' is not legalizable}}
  %0:3 = "test.op_partial_fold_unmaterializable"() : () -> (i32, i32, i32)
  "test.return"(%0#0, %0#1, %0#2) : (i32, i32, i32) -> ()
}

// -----

// In a graph region, an operand of the op can be a result of the same op. A
// fold that forwards such an operand keeps the result.

// CHECK-LABEL: func @partial_fold_graph_region
// CHECK-SAME: (%[[ARG0:[a-z0-9]+]]: i32)
func.func @partial_fold_graph_region(%arg0: i32) {
  // CHECK-NEXT: test.graph_region {
  // CHECK-NEXT: %[[RES_0:[a-z0-9]+]]:2 = "test.op_fold_forward_operands"(%[[RES_0]]#0, %[[RES_0]]#1)
  // CHECK-NEXT: %[[RES_1:[a-z0-9]+]]:2 = "test.op_fold_forward_operands"(%[[ARG0]], %[[RES_1]]#1)
  // ROLLBACK-NEXT: "test.valid"(%[[RES_0]]#0, %[[RES_0]]#1, %[[RES_1]]#0, %[[RES_1]]#1)
  // NO-ROLLBACK-NEXT: "test.valid"(%[[RES_0]]#0, %[[RES_0]]#1, %[[ARG0]], %[[RES_1]]#1)
  // expected-remark@+1 {{op 'test.graph_region' is not legalizable}}
  test.graph_region {
    // expected-remark@+1 {{op 'test.op_fold_forward_operands' is not legalizable}}
    %0:2 = "test.op_fold_forward_operands"(%0#0, %0#1) : (i32, i32) -> (i32, i32)
    // expected-remark@+1 {{op 'test.op_fold_forward_operands' is not legalizable}}
    %1:2 = "test.op_fold_forward_operands"(%arg0, %1#1) : (i32, i32) -> (i32, i32)
    "test.valid"(%0#0, %0#1, %1#0, %1#1) : (i32, i32, i32, i32) -> ()
  }
  "test.return"() : () -> ()
}

// -----

// The replacement does not replace a use in the op that defines the
// replacement value. The fold of the first op replaces no use, so it makes no
// progress. The fold of the second op replaces the use in "test.valid".

// CHECK-LABEL: func @partial_fold_graph_region_skipped_use
func.func @partial_fold_graph_region_skipped_use() {
  // CHECK-NEXT: test.graph_region {
  // CHECK-NEXT: %[[RES_0:[a-z0-9]+]]:2 = "test.op_fold_forward_operands"(%[[RES_1:[a-z0-9]+]]#0, %[[RES_0]]#1)
  // CHECK-NEXT: %[[RES_1]]:2 = "test.op_fold_forward_operands"(%[[RES_0]]#0, %[[RES_1]]#1)
  // ROLLBACK-NEXT: "test.valid"(%[[RES_1]]#0, %[[RES_1]]#1, %[[RES_0]]#1)
  // NO-ROLLBACK-NEXT: "test.valid"(%[[RES_0]]#0, %[[RES_1]]#1, %[[RES_0]]#1)
  // expected-remark@+1 {{op 'test.graph_region' is not legalizable}}
  test.graph_region {
    // expected-remark@+1 {{op 'test.op_fold_forward_operands' is not legalizable}}
    %0:2 = "test.op_fold_forward_operands"(%1#0, %0#1) : (i32, i32) -> (i32, i32)
    // expected-remark@+1 {{op 'test.op_fold_forward_operands' is not legalizable}}
    %1:2 = "test.op_fold_forward_operands"(%0#0, %1#1) : (i32, i32) -> (i32, i32)
    "test.valid"(%1#0, %1#1, %0#1) : (i32, i32, i32) -> ()
  }
  "test.return"() : () -> ()
}
