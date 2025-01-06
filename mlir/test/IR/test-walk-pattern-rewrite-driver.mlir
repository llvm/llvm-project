// RUN: mlir-opt %s --test-walk-pattern-rewrite-driver="dump-notifications=true" \
// RUN:   --allow-unregistered-dialect --split-input-file | FileCheck %s

// The following op is updated in-place and will not be added back to the worklist.
// CHECK-LABEL: func.func @inplace_update()
// CHECK: "test.any_attr_of_i32_str"() <{attr = 1 : i32}> : () -> ()
// CHECK: "test.any_attr_of_i32_str"() <{attr = 2 : i32}> : () -> ()
func.func @inplace_update() {
  "test.any_attr_of_i32_str"() {attr = 0 : i32} : () -> ()
  "test.any_attr_of_i32_str"() {attr = 1 : i32} : () -> ()
  return
}

// Check that the driver does not fold visited ops.
// CHECK-LABEL: func.func @add_no_fold()
// CHECK: arith.constant
// CHECK: arith.constant
// CHECK: %[[RES:.+]] = arith.addi
// CHECK: return %[[RES]]
func.func @add_no_fold() -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %res = arith.addi %c0, %c1 : i32
  return %res : i32
}

// Check that the driver handles rewriter.moveBefore.
// CHECK-LABEL: func.func @move_before(
// CHECK: "test.move_before_parent_op"
// CHECK: "test.any_attr_of_i32_str"() <{attr = 1 : i32}> : () -> ()
// CHECK: scf.if
// CHECK: return
func.func @move_before(%cond : i1) {
  scf.if %cond {
    "test.move_before_parent_op"() ({
      "test.any_attr_of_i32_str"() {attr = 0 : i32} : () -> ()
    }) : () -> ()
  }
  return
}

// Check that the driver handles rewriter.moveAfter. In this case, we expect
// the moved op to be visited only once since walk uses `make_early_inc_range`.
// CHECK-LABEL: func.func @move_after(
// CHECK: scf.if
// CHECK: }
// CHECK: "test.move_after_parent_op"
// CHECK: "test.any_attr_of_i32_str"() <{attr = 1 : i32}> : () -> ()
// CHECK: return
func.func @move_after(%cond : i1) {
  scf.if %cond {
    "test.move_after_parent_op"() ({
      "test.any_attr_of_i32_str"() {attr = 0 : i32} : () -> ()
    }) : () -> ()
  }
  return
}

// Check that the driver handles rewriter.moveAfter. In this case, we expect
// the moved op to be visited twice since we advance its position to the next
// node after the parent.
// CHECK-LABEL: func.func @move_forward_and_revisit(
// CHECK: scf.if
// CHECK: }
// CHECK: arith.addi
// CHECK: "test.move_after_parent_op"
// CHECK: "test.any_attr_of_i32_str"() <{attr = 2 : i32}> : () -> ()
// CHECK: arith.addi
// CHECK: return
func.func @move_forward_and_revisit(%cond : i1) {
  scf.if %cond {
    "test.move_after_parent_op"() ({
      "test.any_attr_of_i32_str"() {attr = 0 : i32} : () -> ()
    }) {advance = 1 : i32} : () -> ()
  }
  %a = arith.addi %cond, %cond : i1
  %b = arith.addi %a, %cond : i1
  return
}

// Operation inserted just after the currently visited one won't be visited.
// CHECK-LABEL: func.func @insert_just_after
// CHECK: "test.clone_me"() ({
// CHECK:   "test.any_attr_of_i32_str"() <{attr = 1 : i32}> : () -> ()
// CHECK: }) {was_cloned} : () -> ()
// CHECK: "test.clone_me"() ({
// CHECK:   "test.any_attr_of_i32_str"() <{attr = 1 : i32}> : () -> ()
// CHECK: }) : () -> ()
// CHECK: return
func.func @insert_just_after(%cond : i1) {
  "test.clone_me"() ({
    "test.any_attr_of_i32_str"() {attr = 0 : i32} : () -> ()
  }) : () -> ()
  return
}

// Check that we can replace the current operation with a new one.
// Note that the new op won't be visited.
// CHECK-LABEL: func.func @replace_with_new_op
// CHECK: %[[NEW:.+]] = "test.new_op"
// CHECK: %[[RES:.+]] = arith.addi %[[NEW]], %[[NEW]]
// CHECK: return %[[RES]]
func.func @replace_with_new_op() -> i32 {
  %a = "test.replace_with_new_op"() : () -> (i32)
  %res = arith.addi %a, %a : i32
  return %res : i32
}

// Check that we can erase nested blocks.
// CHECK-LABEL: func.func @erase_nested_block
// CHECK:         %[[RES:.+]] = "test.erase_first_block"
// CHECK-NEXT:    foo.bar
// CHECK:         return %[[RES]]
func.func @erase_nested_block() -> i32 {
  %a = "test.erase_first_block"() ({
    "foo.foo"() : () -> ()
    ^bb1:
    "foo.bar"() : () -> ()
  }): () -> (i32)
  return %a : i32
}
