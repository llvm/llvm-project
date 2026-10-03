// RUN: mlir-opt %s -pass-pipeline='builtin.module(func.func(canonicalize))' | FileCheck %s  --check-prefixes=CHECK,RS
// RUN: mlir-opt %s -pass-pipeline='builtin.module(func.func(canonicalize{region-simplify=disabled}))' | FileCheck %s --check-prefixes=CHECK,NO-RS

// CHECK-LABEL: func @remove_op_with_inner_ops_pattern
func.func @remove_op_with_inner_ops_pattern() {
  // CHECK-NEXT: return
  "test.op_with_region_pattern"() ({
    "test.op_with_region_terminator"() : () -> ()
  }) : () -> ()
  return
}

// CHECK-LABEL: func @remove_op_with_inner_ops_fold_no_side_effect
func.func @remove_op_with_inner_ops_fold_no_side_effect() {
  // CHECK-NEXT: return
  "test.op_with_region_fold_no_side_effect"() ({
    "test.op_with_region_terminator"() : () -> ()
  }) : () -> ()
  return
}

// CHECK-LABEL: func @remove_op_with_inner_ops_fold
// CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: i32)
func.func @remove_op_with_inner_ops_fold(%arg0 : i32) -> (i32) {
  // CHECK-NEXT: return %[[ARG_0]]
  %0 = "test.op_with_region_fold"(%arg0) ({
    "test.op_with_region_terminator"() : () -> ()
  }) : (i32) -> (i32)
  return %0 : i32
}

// CHECK-LABEL: func @remove_op_with_variadic_results_and_folder
// CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: i32, %[[ARG_1:[a-z0-9]*]]: i32)
func.func @remove_op_with_variadic_results_and_folder(%arg0 : i32, %arg1 : i32) -> (i32, i32) {
  // CHECK-NEXT: return %[[ARG_0]], %[[ARG_1]]
  %0, %1 = "test.op_with_variadic_results_and_folder"(%arg0, %arg1) : (i32, i32) -> (i32, i32)
  return %0, %1 : i32, i32
}

// Without operands, the fold replaces no result, so it fails.
// CHECK-LABEL: func @keep_op_with_variadic_results_and_folder_no_operands
func.func @keep_op_with_variadic_results_and_folder_no_operands() {
  // CHECK-NEXT: "test.op_with_variadic_results_and_folder"() : () -> ()
  // CHECK-NEXT: return
  "test.op_with_variadic_results_and_folder"() : () -> ()
  return
}

// CHECK-LABEL: func @test_commutative_multi
// CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: i32, %[[ARG_1:[a-z0-9]*]]: i32)
func.func @test_commutative_multi(%arg0: i32, %arg1: i32) -> (i32, i32) {
  // CHECK-DAG: %[[C42:.*]] = arith.constant 42 : i32
  %c42_i32 = arith.constant 42 : i32
  // CHECK-DAG: %[[C43:.*]] = arith.constant 43 : i32
  %c43_i32 = arith.constant 43 : i32
  // CHECK-NEXT: %[[O0:.*]] = "test.op_commutative"(%[[ARG_0]], %[[ARG_1]], %[[C42]], %[[C43]]) : (i32, i32, i32, i32) -> i32
  %y = "test.op_commutative"(%c42_i32, %arg0, %arg1, %c43_i32) : (i32, i32, i32, i32) -> i32

  // CHECK-NEXT: %[[O1:.*]] = "test.op_commutative"(%[[ARG_0]], %[[ARG_1]], %[[C42]], %[[C43]]) : (i32, i32, i32, i32) -> i32
  %z = "test.op_commutative"(%arg0, %c42_i32, %c43_i32, %arg1): (i32, i32, i32, i32) -> i32
  // CHECK-NEXT: return %[[O0]], %[[O1]]
  return %y, %z: i32, i32
}


// CHECK-LABEL: func @test_commutative_multi_cst
func.func @test_commutative_multi_cst(%arg0: i32, %arg1: i32) -> (i32, i32) {
  // CHECK-NEXT: %c42_i32 = arith.constant 42 : i32
  %c42_i32 = arith.constant 42 : i32
  %c42_i32_2 = arith.constant 42 : i32
  // CHECK-NEXT: %[[O0:.*]] = "test.op_commutative"(%arg0, %arg1, %c42_i32, %c42_i32) : (i32, i32, i32, i32) -> i32
  %y = "test.op_commutative"(%c42_i32, %arg0, %arg1, %c42_i32_2) : (i32, i32, i32, i32) -> i32

  %c42_i32_3 = arith.constant 42 : i32

  // CHECK-NEXT: %[[O1:.*]] = "test.op_commutative"(%arg0, %arg1, %c42_i32, %c42_i32) : (i32, i32, i32, i32) -> i32
  %z = "test.op_commutative"(%arg0, %c42_i32_3, %c42_i32_2, %arg1): (i32, i32, i32, i32) -> i32
  // CHECK-NEXT: return %[[O0]], %[[O1]]
  return %y, %z: i32, i32
}

// CHECK-LABEL: test_dialect_canonicalizer
func.func @test_dialect_canonicalizer() -> (i32) {
  %0 = "test.dialect_canonicalizable"() : () -> (i32)
  // CHECK: %[[CST:.*]] = arith.constant 42 : i32
  // CHECK: return %[[CST]]
  return %0 : i32
}

// Check that the option to control region simplification actually works
// CHECK-LABEL: test_region_simplify
func.func @test_region_simplify(%input1 : i32, %cond : i1) -> i32 {
  // RS-NEXT: "test.br"(%arg0)[^bb1] : (i32) -> ()
  // NO-RS-NEXT: "test.br"(%arg0, %arg0)[^bb1] : (i32, i32) -> ()
   "test.br"(%input1, %input1)[^bb1] : (i32, i32) -> ()
^bb1(%used_arg : i32, %unused_arg : i32):
  return %used_arg : i32
}

// When constant materialization fails for one result, the fold must not apply.
// The driver erases only the constant that it materialized for the first
// result, and keeps the op that defines the forwarded operand.
// CHECK-LABEL: func @fold_unmaterializable_existing_op
// CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: i32, %[[ARG_1:[a-z0-9]*]]: i32)
func.func @fold_unmaterializable_existing_op(%arg0 : i32, %arg1 : i32) -> (i32, i32, i32) {
  // CHECK-NEXT: %[[ADD:[a-z0-9]+]] = "test.addi"(%[[ARG_0]], %[[ARG_1]])
  %0 = "test.addi"(%arg0, %arg1) : (i32, i32) -> i32
  // CHECK-NEXT: %[[RES:[a-z0-9]+]]:3 = "test.op_fold_unmaterializable"(%[[ADD]])
  %1:3 = "test.op_fold_unmaterializable"(%0) : (i32) -> (i32, i32, i32)
  // CHECK-NEXT: return %[[RES]]#0, %[[RES]]#1, %[[RES]]#2
  return %1#0, %1#1, %1#2 : i32, i32, i32
}

// Same as above, but the forwarded operand is a block argument.
// CHECK-LABEL: func @fold_unmaterializable_block_arg
// CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: i32)
func.func @fold_unmaterializable_block_arg(%arg0 : i32) -> (i32, i32, i32) {
  // CHECK-NEXT: %[[RES:[a-z0-9]+]]:3 = "test.op_fold_unmaterializable"(%[[ARG_0]])
  %0:3 = "test.op_fold_unmaterializable"(%arg0) : (i32) -> (i32, i32, i32)
  // CHECK-NEXT: return %[[RES]]#0, %[[RES]]#1, %[[RES]]#2
  return %0#0, %0#1, %0#2 : i32, i32, i32
}

// A partial fold replaces the uses of the replaced results and keeps the op.
// CHECK-LABEL: func @partial_fold
// CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: i32)
func.func @partial_fold(%arg0 : i32) -> (i32, i32, i32) {
  // CHECK-NEXT: %[[C42:[a-z0-9_]+]] = "test.constant"() <{value = 42 : i32}> : () -> i32
  // CHECK-NEXT: %[[RES:[a-z0-9]+]]:3 = "test.fold_dispatch"(%[[ARG_0]])
  %0:3 = "test.fold_dispatch"(%arg0)
      {fold = {replace = [42 : i32, #test.fold_ref<operand 0>,
                          #test.fold_ref<keep>]}}
      : (i32) -> (i32, i32, i32)
  // CHECK-NEXT: return %[[C42]], %[[ARG_0]], %[[RES]]#2
  return %0#0, %0#1, %0#2 : i32, i32, i32
}

// A partial fold can also change the op in place.
// CHECK-LABEL: func @partial_fold_in_place
// CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: i32)
func.func @partial_fold_in_place(%arg0 : i32) -> (i32, i32) {
  // CHECK-NEXT: %[[RES:[a-z0-9]+]]:2 = "test.fold_dispatch"(%[[ARG_0]]) {fold = {replace = [#test.fold_ref<operand 0>, #test.fold_ref<keep>]}}
  %0:2 = "test.fold_dispatch"(%arg0)
      {fold = {replace = [#test.fold_ref<operand 0>, #test.fold_ref<keep>],
               in_place}}
      : (i32) -> (i32, i32)
  // CHECK-NEXT: return %[[ARG_0]], %[[RES]]#1
  return %0#0, %0#1 : i32, i32
}

// A fold that keeps every result and does not change the op in place fails.
// CHECK-LABEL: func @fold_keep_all
func.func @fold_keep_all() -> (i32, i32) {
  // CHECK-NEXT: %[[RES:[a-z0-9]+]]:2 = "test.fold_dispatch"()
  %0:2 = "test.fold_dispatch"()
      {fold = {replace = [#test.fold_ref<result 0>, #test.fold_ref<result 1>]}}
      : () -> (i32, i32)
  // CHECK-NEXT: return %[[RES]]#0, %[[RES]]#1
  return %0#0, %0#1 : i32, i32
}

// The driver does not materialize a replaced result without uses.
// CHECK-LABEL: func @partial_fold_dead_replaced_results
// CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: i32)
func.func @partial_fold_dead_replaced_results(%arg0 : i32) -> i32 {
  // CHECK-NEXT: %[[RES:[a-z0-9]+]]:3 = "test.fold_dispatch"(%[[ARG_0]])
  %0:3 = "test.fold_dispatch"(%arg0)
      {fold = {replace = [42 : i32, #test.fold_ref<operand 0>,
                          #test.fold_ref<keep>]}}
      : (i32) -> (i32, i32, i32)
  // CHECK-NEXT: return %[[RES]]#2
  return %0#2 : i32
}

// A fold that replaces every result also materializes the results without
// uses. The unmaterializable result makes the fold fail, so the op stays.
// CHECK-LABEL: func @fold_dead_unmaterializable_result
// CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: i32)
func.func @fold_dead_unmaterializable_result(%arg0 : i32) -> (i32, i32) {
  // CHECK-NEXT: %[[RES:[a-z0-9]+]]:3 = "test.op_fold_unmaterializable"(%[[ARG_0]])
  %0:3 = "test.op_fold_unmaterializable"(%arg0) : (i32) -> (i32, i32, i32)
  // CHECK-NEXT: return %[[RES]]#0, %[[RES]]#1
  return %0#0, %0#1 : i32, i32
}

// When constant materialization fails, the partial fold does not apply and the
// driver inserts no constant.
// CHECK-LABEL: func @partial_fold_unmaterializable
func.func @partial_fold_unmaterializable() -> (i32, i32, i32) {
  // CHECK-NEXT: %[[RES:[a-z0-9]+]]:3 = "test.fold_dispatch"()
  %0:3 = "test.fold_dispatch"()
      {fold = {replace = [42 : i32, "unmaterializable", #test.fold_ref<keep>]}}
      : () -> (i32, i32, i32)
  // CHECK-NEXT: return %[[RES]]#0, %[[RES]]#1, %[[RES]]#2
  return %0#0, %0#1, %0#2 : i32, i32, i32
}

// Same as above, but the unmaterializable result has no uses, so the partial
// fold applies.
// CHECK-LABEL: func @partial_fold_dead_unmaterializable_result
func.func @partial_fold_dead_unmaterializable_result() -> (i32, i32) {
  // CHECK-NEXT: %[[C42:[a-z0-9_]+]] = "test.constant"() <{value = 42 : i32}> : () -> i32
  // CHECK-NEXT: %[[RES:[a-z0-9]+]]:3 = "test.fold_dispatch"()
  %0:3 = "test.fold_dispatch"()
      {fold = {replace = [42 : i32, "unmaterializable", #test.fold_ref<keep>]}}
      : () -> (i32, i32, i32)
  // CHECK-NEXT: return %[[C42]], %[[RES]]#2
  return %0#0, %0#2 : i32, i32
}

// In a graph region, an operand of the op can be a result of the same op. A
// fold that forwards such an operand keeps the result.
// CHECK-LABEL: func @partial_fold_graph_region
// CHECK-SAME: (%[[ARG_0:[a-z0-9]*]]: i32)
func.func @partial_fold_graph_region(%arg0 : i32) {
  // CHECK-NEXT: test.graph_region {
  test.graph_region {
    // CHECK-NEXT: %[[RES_0:[a-z0-9]+]]:2 = "test.fold_dispatch"(%[[RES_0]]#0, %[[RES_0]]#1)
    %0:2 = "test.fold_dispatch"(%0#0, %0#1)
        {fold = {replace = [#test.fold_ref<operand 0>,
                            #test.fold_ref<operand 1>]}}
        : (i32, i32) -> (i32, i32)
    // CHECK-NEXT: %[[RES_1:[a-z0-9]+]]:2 = "test.fold_dispatch"(%[[ARG_0]], %[[RES_1]]#1)
    %1:2 = "test.fold_dispatch"(%arg0, %1#1)
        {fold = {replace = [#test.fold_ref<operand 0>,
                            #test.fold_ref<operand 1>]}}
        : (i32, i32) -> (i32, i32)
    // CHECK-NEXT: "test.valid"(%[[RES_0]]#0, %[[RES_0]]#1, %[[ARG_0]], %[[RES_1]]#1)
    "test.valid"(%0#0, %0#1, %1#0, %1#1) : (i32, i32, i32, i32) -> ()
  }
  return
}

// A fold of an op without results can change the op in place.
// CHECK-LABEL: func @zero_results_fold_in_place
func.func @zero_results_fold_in_place() {
  // CHECK-NEXT: "test.fold_dispatch"() : () -> ()
  "test.fold_dispatch"() {fold = {in_place}} : () -> ()
  // CHECK-NEXT: return
  return
}
