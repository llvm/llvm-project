// RUN: mlir-opt -split-input-file -test-legalize-patterns="allow-pattern-rollback=1" -verify-diagnostics %s | FileCheck %s --check-prefixes=CHECK,ROLLBACK
// RUN: mlir-opt -split-input-file -test-legalize-patterns="allow-pattern-rollback=0" -verify-diagnostics %s | FileCheck %s --check-prefixes=CHECK,NO-ROLLBACK

// Without rollback, the legalizer applies a partial fold. With rollback, it
// applies only the in-place change of a partial fold. The test ops are not in
// the conversion target, so each op stays and gets a remark in both modes.

// CHECK-LABEL: func @partial_fold
// CHECK-SAME: (%[[ARG0:[a-z0-9]+]]: i32)
func.func @partial_fold(%arg0: i32) -> (i32, i32, i32) {
  // NO-ROLLBACK-NEXT: %[[C42:[a-z0-9]+]] = "test.constant"() <{value = 42 : i32}> : () -> i32
  // NO-ROLLBACK-NEXT: %[[RES:[a-z0-9]+]]:3 = "test.fold_dispatch"(%[[ARG0]])
  // NO-ROLLBACK-NEXT: "test.return"(%[[C42]], %[[ARG0]], %[[RES]]#2)
  // ROLLBACK-NEXT: %[[RES:[a-z0-9]+]]:3 = "test.fold_dispatch"(%[[ARG0]])
  // ROLLBACK-NEXT: "test.return"(%[[RES]]#0, %[[RES]]#1, %[[RES]]#2)
  // expected-remark@+1 {{op 'test.fold_dispatch' is not legalizable}}
  %0:3 = "test.fold_dispatch"(%arg0)
      {fold = {replace = [42 : i32, #test.fold_ref<operand 0>,
                          #test.fold_ref<keep>]}}
      : (i32) -> (i32, i32, i32)
  "test.return"(%0#0, %0#1, %0#2) : (i32, i32, i32) -> ()
}

// -----

// CHECK-LABEL: func @partial_fold_in_place
// CHECK-SAME: (%[[ARG0:[a-z0-9]+]]: i32)
func.func @partial_fold_in_place(%arg0: i32) -> (i32, i32) {
  // CHECK-NEXT: %[[RES:[a-z0-9]+]]:2 = "test.fold_dispatch"(%[[ARG0]]) {fold = {replace = [#test.fold_ref<operand 0>, #test.fold_ref<keep>]}}
  // NO-ROLLBACK-NEXT: "test.return"(%[[ARG0]], %[[RES]]#1)
  // ROLLBACK-NEXT: "test.return"(%[[RES]]#0, %[[RES]]#1)
  // expected-remark@+1 {{op 'test.fold_dispatch' is not legalizable}}
  %0:2 = "test.fold_dispatch"(%arg0)
      {fold = {replace = [#test.fold_ref<operand 0>, #test.fold_ref<keep>],
               in_place}}
      : (i32) -> (i32, i32)
  "test.return"(%0#0, %0#1) : (i32, i32) -> ()
}

// -----

// CHECK-LABEL: func @fold_keep_all
func.func @fold_keep_all() -> (i32, i32) {
  // CHECK-NEXT: %[[RES:[a-z0-9]+]]:2 = "test.fold_dispatch"()
  // CHECK-NEXT: "test.return"(%[[RES]]#0, %[[RES]]#1)
  // expected-remark@+1 {{op 'test.fold_dispatch' is not legalizable}}
  %0:2 = "test.fold_dispatch"()
      {fold = {replace = [#test.fold_ref<result 0>, #test.fold_ref<result 1>]}}
      : () -> (i32, i32)
  "test.return"(%0#0, %0#1) : (i32, i32) -> ()
}

// -----

// The replaced results have no uses, so the fold makes no progress.

// CHECK-LABEL: func @partial_fold_dead_replaced_results
// CHECK-SAME: (%[[ARG0:[a-z0-9]+]]: i32)
func.func @partial_fold_dead_replaced_results(%arg0: i32) -> i32 {
  // CHECK-NEXT: %[[RES:[a-z0-9]+]]:3 = "test.fold_dispatch"(%[[ARG0]])
  // CHECK-NEXT: "test.return"(%[[RES]]#2)
  // expected-remark@+1 {{op 'test.fold_dispatch' is not legalizable}}
  %0:3 = "test.fold_dispatch"(%arg0)
      {fold = {replace = [42 : i32, #test.fold_ref<operand 0>,
                          #test.fold_ref<keep>]}}
      : (i32) -> (i32, i32, i32)
  "test.return"(%0#2) : (i32) -> ()
}

// -----

// CHECK-LABEL: func @partial_fold_unmaterializable
func.func @partial_fold_unmaterializable() -> (i32, i32, i32) {
  // CHECK-NEXT: %[[RES:[a-z0-9]+]]:3 = "test.fold_dispatch"()
  // CHECK-NEXT: "test.return"(%[[RES]]#0, %[[RES]]#1, %[[RES]]#2)
  // expected-remark@+1 {{op 'test.fold_dispatch' is not legalizable}}
  %0:3 = "test.fold_dispatch"()
      {fold = {replace = [42 : i32, "unmaterializable", #test.fold_ref<keep>]}}
      : () -> (i32, i32, i32)
  "test.return"(%0#0, %0#1, %0#2) : (i32, i32, i32) -> ()
}

// -----

// In a graph region, an operand of the op can be a result of the same op. A
// fold that forwards such an operand keeps the result.

// CHECK-LABEL: func @partial_fold_graph_region
// CHECK-SAME: (%[[ARG0:[a-z0-9]+]]: i32)
func.func @partial_fold_graph_region(%arg0: i32) {
  // CHECK-NEXT: test.graph_region {
  // CHECK-NEXT: %[[RES_0:[a-z0-9]+]]:2 = "test.fold_dispatch"(%[[RES_0]]#0, %[[RES_0]]#1)
  // CHECK-NEXT: %[[RES_1:[a-z0-9]+]]:2 = "test.fold_dispatch"(%[[ARG0]], %[[RES_1]]#1)
  // NO-ROLLBACK-NEXT: "test.valid"(%[[RES_0]]#0, %[[RES_0]]#1, %[[ARG0]], %[[RES_1]]#1)
  // ROLLBACK-NEXT: "test.valid"(%[[RES_0]]#0, %[[RES_0]]#1, %[[RES_1]]#0, %[[RES_1]]#1)
  // expected-remark@+1 {{op 'test.graph_region' is not legalizable}}
  test.graph_region {
    // expected-remark@+1 {{op 'test.fold_dispatch' is not legalizable}}
    %0:2 = "test.fold_dispatch"(%0#0, %0#1)
        {fold = {replace = [#test.fold_ref<operand 0>,
                            #test.fold_ref<operand 1>]}}
        : (i32, i32) -> (i32, i32)
    // expected-remark@+1 {{op 'test.fold_dispatch' is not legalizable}}
    %1:2 = "test.fold_dispatch"(%arg0, %1#1)
        {fold = {replace = [#test.fold_ref<operand 0>,
                            #test.fold_ref<operand 1>]}}
        : (i32, i32) -> (i32, i32)
    "test.valid"(%0#0, %0#1, %1#0, %1#1) : (i32, i32, i32, i32) -> ()
  }
  "test.return"() : () -> ()
}

// -----

// The replacement does not replace a use in the op that defines the
// replacement value. Without rollback, the fold of the first op replaces no
// use, so it makes no progress, and the fold of the second op replaces the use
// in "test.valid". With rollback, neither partial fold applies.

// CHECK-LABEL: func @partial_fold_graph_region_skipped_use
func.func @partial_fold_graph_region_skipped_use() {
  // CHECK-NEXT: test.graph_region {
  // CHECK-NEXT: %[[RES_0:[a-z0-9]+]]:2 = "test.fold_dispatch"(%[[RES_1:[a-z0-9]+]]#0, %[[RES_0]]#1)
  // CHECK-NEXT: %[[RES_1]]:2 = "test.fold_dispatch"(%[[RES_0]]#0, %[[RES_1]]#1)
  // ROLLBACK-NEXT: "test.valid"(%[[RES_1]]#0, %[[RES_1]]#1, %[[RES_0]]#1)
  // NO-ROLLBACK-NEXT: "test.valid"(%[[RES_0]]#0, %[[RES_1]]#1, %[[RES_0]]#1)
  // expected-remark@+1 {{op 'test.graph_region' is not legalizable}}
  test.graph_region {
    // expected-remark@+1 {{op 'test.fold_dispatch' is not legalizable}}
    %0:2 = "test.fold_dispatch"(%1#0, %0#1)
        {fold = {replace = [#test.fold_ref<operand 0>,
                            #test.fold_ref<operand 1>]}}
        : (i32, i32) -> (i32, i32)
    // expected-remark@+1 {{op 'test.fold_dispatch' is not legalizable}}
    %1:2 = "test.fold_dispatch"(%0#0, %1#1)
        {fold = {replace = [#test.fold_ref<operand 0>,
                            #test.fold_ref<operand 1>]}}
        : (i32, i32) -> (i32, i32)
    "test.valid"(%1#0, %1#1, %0#1) : (i32, i32, i32) -> ()
  }
  "test.return"() : () -> ()
}

// -----

// A constant fails to materialize. Both modes keep the in-place change of the
// fold, which consumes the `in_place` key, and the op stays.

// CHECK-LABEL: func @full_fold_unmaterializable_in_place
func.func @full_fold_unmaterializable_in_place() -> (i32, i32) {
  // CHECK-NEXT: %[[RES:[a-z0-9]+]]:2 = "test.fold_dispatch"() {fold = {replace = [42 : i32, "unmaterializable"]}}
  // CHECK-NEXT: "test.return"(%[[RES]]#0, %[[RES]]#1)
  // expected-remark@+1 {{op 'test.fold_dispatch' is not legalizable}}
  %0:2 = "test.fold_dispatch"()
      {fold = {replace = [42 : i32, "unmaterializable"], in_place}}
      : () -> (i32, i32)
  "test.return"(%0#0, %0#1) : (i32, i32) -> ()
}
