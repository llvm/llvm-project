// RUN: mlir-opt \
// RUN:     -test-strict-pattern-driver="strictness=AnyOp" \
// RUN:     --split-input-file %s | FileCheck %s --check-prefix=CHECK-AN

// RUN: mlir-opt \
// RUN:     -test-strict-pattern-driver="strictness=ExistingAndNewOps" \
// RUN:     --split-input-file %s | FileCheck %s --check-prefix=CHECK-EN

// RUN: mlir-opt \
// RUN:     -test-strict-pattern-driver="strictness=ExistingOps" \
// RUN:     --split-input-file %s | FileCheck %s --check-prefix=CHECK-EX

// CHECK-EN-LABEL: func @test_erase
//  CHECK-EN-SAME:     pattern_driver_all_erased = true, pattern_driver_changed = true}
//       CHECK-EN:   "test.arg0"
//       CHECK-EN:   "test.arg1"
//   CHECK-EN-NOT:   "test.erase_op"
func.func @test_erase() {
  %0 = "test.arg0"() : () -> (i32)
  %1 = "test.arg1"() : () -> (i32)
  %erase = "test.erase_op"(%0, %1) : (i32, i32) -> (i32)
  return
}

// -----

// CHECK-EN: notifyOperationInserted: test.insert_same_op, was unlinked
// CHECK-EN-LABEL: func @test_insert_same_op
//  CHECK-EN-SAME:     {pattern_driver_all_erased = false, pattern_driver_changed = true}
//       CHECK-EN:   "test.insert_same_op"() {skip = true}
//       CHECK-EN:   "test.insert_same_op"() {skip = true}
func.func @test_insert_same_op() {
  %0 = "test.insert_same_op"() : () -> (i32)
  return
}

// -----

// CHECK-EN: notifyOperationInserted: test.new_op, was unlinked
// CHECK-EN-LABEL: func @test_replace_with_new_op
//  CHECK-EN-SAME:     {pattern_driver_all_erased = true, pattern_driver_changed = true}
//       CHECK-EN:   %[[n:.*]] = "test.new_op"
//       CHECK-EN:   "test.dummy_user"(%[[n]])
//       CHECK-EN:   "test.dummy_user"(%[[n]])
func.func @test_replace_with_new_op() {
  %0 = "test.replace_with_new_op"() : () -> (i32)
  %1 = "test.dummy_user"(%0) : (i32) -> (i32)
  %2 = "test.dummy_user"(%0) : (i32) -> (i32)
  return
}

// -----

// CHECK-EN: notifyOperationInserted: test.erase_op, was unlinked
// CHECK-EN: notifyOperationRemoved: test.replace_with_new_op
// CHECK-EN: notifyOperationRemoved: test.erase_op
// CHECK-EN-LABEL: func @test_replace_with_erase_op
//  CHECK-EN-SAME:     {pattern_driver_all_erased = true, pattern_driver_changed = true}
//   CHECK-EN-NOT:   "test.replace_with_new_op"
//   CHECK-EN-NOT:   "test.erase_op"

// CHECK-EX-LABEL: func @test_replace_with_erase_op
//  CHECK-EX-SAME:     {pattern_driver_all_erased = true, pattern_driver_changed = true}
//   CHECK-EX-NOT:   "test.replace_with_new_op"
//       CHECK-EX:   "test.erase_op"
func.func @test_replace_with_erase_op() {
  "test.replace_with_new_op"() {create_erase_op} : () -> ()
  return
}

// -----

// CHECK-AN-LABEL: func @test_trigger_rewrite_through_block
//       CHECK-AN: "test.change_block_op"()[^[[BB0:.*]], ^[[BB0]]]
//       CHECK-AN: return
//       CHECK-AN: ^[[BB1:[^:]*]]:
//       CHECK-AN: "test.implicit_change_op"()[^[[BB1]]]
func.func @test_trigger_rewrite_through_block() {
  return
^bb1:
  // Uses bb1. ChangeBlockOp replaces that and all other usages of bb1 with bb2.
  "test.change_block_op"() [^bb1, ^bb2] : () -> ()
^bb2:
  return
^bb3:
  // Also uses bb1. ChangeBlockOp replaces that usage with bb2. This triggers
  // this op being put on the worklist, which triggers ImplicitChangeOp, which,
  // in turn, replaces the successor with bb3.
  "test.implicit_change_op"() [^bb1] : () -> ()
}

// -----

// CHECK-AN: notifyOperationRemoved: test.foo_b
// CHECK-AN: notifyOperationRemoved: test.foo_a
// CHECK-AN: notifyOperationRemoved: test.graph_region
// CHECK-AN: notifyOperationRemoved: test.erase_op
// CHECK-AN-LABEL: func @test_remove_graph_region()
//  CHECK-AN-NEXT:   return
func.func @test_remove_graph_region() {
  "test.erase_op"() ({
    test.graph_region {
      %0 = "test.foo_a"(%1) : (i1) -> (i1)
      %1 = "test.foo_b"(%0) : (i1) -> (i1)
    }
  }) : () -> ()
  return
}

// -----

// CHECK-AN: notifyOperationRemoved: cf.br
// CHECK-AN: notifyOperationRemoved: test.bar
// CHECK-AN: notifyOperationRemoved: cf.br
// CHECK-AN: notifyOperationRemoved: test.foo
// CHECK-AN: notifyOperationRemoved: cf.br
// CHECK-AN: notifyOperationRemoved: test.dummy_op
// CHECK-AN: notifyOperationRemoved: test.erase_op
// CHECK-AN-LABEL: func @test_remove_cyclic_blocks()
//  CHECK-AN-NEXT:   return
func.func @test_remove_cyclic_blocks() {
  "test.erase_op"() ({
    %x = "test.dummy_op"() : () -> (i1)
    cf.br ^bb1(%x: i1)
  ^bb1(%arg0: i1):
    "test.foo"(%x) : (i1) -> ()
    cf.br ^bb2(%arg0: i1)
  ^bb2(%arg1: i1):
    "test.bar"(%x) : (i1) -> ()
    cf.br ^bb1(%arg1: i1)
  }) : () -> ()
  return
}

// -----

// CHECK-AN: notifyOperationRemoved: test.dummy_op
// CHECK-AN: notifyOperationRemoved: test.bar
// CHECK-AN: notifyOperationRemoved: test.qux
// CHECK-AN: notifyOperationRemoved: test.qux_unreachable
// CHECK-AN: notifyOperationRemoved: test.nested_dummy
// CHECK-AN: notifyOperationRemoved: cf.br
// CHECK-AN: notifyOperationRemoved: test.foo
// CHECK-AN: notifyOperationRemoved: test.erase_op
// CHECK-AN-LABEL: func @test_remove_dead_blocks()
//  CHECK-AN-NEXT:   return
func.func @test_remove_dead_blocks() {
  "test.erase_op"() ({
    "test.dummy_op"() : () -> (i1)
  // The following blocks are not reachable. Still, ^bb2 should be deleted
  // befire ^bb1.
  ^bb1(%arg0: i1):
    "test.foo"() : () -> ()
    cf.br ^bb2(%arg0: i1)
  ^bb2(%arg1: i1):
    "test.nested_dummy"() ({
      "test.qux"() : () -> ()
    // The following block is unreachable.
    ^bb3:
      "test.qux_unreachable"() : () -> ()
    }) : () -> ()
    "test.bar"() : () -> ()
  }) : () -> ()
  return
}

// -----

// test.nested_* must be deleted before test.foo.
// test.bar must be deleted before test.foo.

// CHECK-AN: notifyOperationRemoved: cf.br
// CHECK-AN: notifyOperationRemoved: test.bar
// CHECK-AN: notifyOperationRemoved: cf.br
// CHECK-AN: notifyOperationRemoved: test.nested_b
// CHECK-AN: notifyOperationRemoved: test.nested_a
// CHECK-AN: notifyOperationRemoved: test.nested_d
// CHECK-AN: notifyOperationRemoved: cf.br
// CHECK-AN: notifyOperationRemoved: test.nested_e
// CHECK-AN: notifyOperationRemoved: cf.br
// CHECK-AN: notifyOperationRemoved: test.nested_c
// CHECK-AN: notifyOperationRemoved: test.foo
// CHECK-AN: notifyOperationRemoved: cf.br
// CHECK-AN: notifyOperationRemoved: test.dummy_op
// CHECK-AN: notifyOperationRemoved: test.erase_op
// CHECK-AN-LABEL: func @test_remove_nested_ops()
//  CHECK-AN-NEXT:   return
func.func @test_remove_nested_ops() {
  "test.erase_op"() ({
    %x = "test.dummy_op"() : () -> (i1)
    cf.br ^bb1(%x: i1)
  ^bb1(%arg0: i1):
    "test.foo"() ({
      "test.nested_a"() : () -> ()
      "test.nested_b"() : () -> ()
    ^dead1:
      "test.nested_c"() : () -> ()
      cf.br ^dead3
    ^dead2:
      "test.nested_d"() : () -> ()
    ^dead3:
      "test.nested_e"() : () -> ()
      cf.br ^dead2
    }) : () -> ()
    cf.br ^bb2(%arg0: i1)
  ^bb2(%arg1: i1):
    "test.bar"(%x) : (i1) -> ()
    cf.br ^bb1(%arg1: i1)
  }) : () -> ()
  return
}

// -----

// CHECK-AN: notifyOperationRemoved: test.qux
// CHECK-AN: notifyOperationRemoved: cf.br
// CHECK-AN: notifyOperationRemoved: test.foo
// CHECK-AN: notifyOperationRemoved: cf.br
// CHECK-AN: notifyOperationRemoved: test.bar
// CHECK-AN: notifyOperationRemoved: cf.cond_br
// CHECK-AN-LABEL: func @test_remove_diamond(
//  CHECK-AN-NEXT:   return
func.func @test_remove_diamond(%c: i1) {
  "test.erase_op"() ({
    cf.cond_br %c, ^bb1, ^bb2
  ^bb1:
    "test.foo"() : () -> ()
    cf.br ^bb3
  ^bb2:
    "test.bar"() : () -> ()
    cf.br ^bb3
  ^bb3:
    "test.qux"() : () -> ()
  }) : () -> ()
  return
}

// -----

// CHECK-AN: notifyOperationInserted: test.move_before_parent_op, previous = test.dummy_terminator
// CHECK-AN-LABEL: func @test_move_op_before(
//       CHECK-AN:   test.move_before_parent_op
//       CHECK-AN:   test.op_with_region
//       CHECK-AN:     test.dummy_terminator
func.func @test_move_op_before() {
  "test.op_with_region"() ({
    "test.move_before_parent_op"() : () -> ()
    "test.dummy_terminator"() : () ->()
  }) : () -> ()
  return
}

// -----

// CHECK-AN: notifyOperationInserted: test.op_1, previous = test.op_2
// CHECK-AN: notifyOperationInserted: test.op_2, previous = test.op_3
// CHECK-AN: notifyOperationInserted: test.op_3, was last in block
// CHECK-AN-LABEL: func @test_inline_block_before(
//       CHECK-AN:   test.op_1
//       CHECK-AN:   test.op_2
//       CHECK-AN:   test.op_3
//       CHECK-AN:   test.inline_blocks_into_parent
//       CHECK-AN:   return
func.func @test_inline_block_before() {
  "test.inline_blocks_into_parent"() ({
    "test.op_1"() : () -> ()
    "test.op_2"() : () -> ()
    "test.op_3"() : () -> ()
  }) : () -> ()
  return
}

// -----

// CHECK-AN: notifyBlockInserted into test.op_with_region: was unlinked
// CHECK-AN: notifyOperationInserted: test.op_3, was last in block
// CHECK-AN: notifyOperationInserted: test.op_2, was last in block
// CHECK-AN: notifyOperationInserted: test.split_block_here, was last in block
// CHECK-AN: notifyOperationInserted: test.new_op, was unlinked
// CHECK-AN: notifyOperationRemoved: test.split_block_here
// CHECK-AN-LABEL: func @test_split_block(
//       CHECK-AN:   "test.op_with_region"() ({
//       CHECK-AN:     test.op_1
//       CHECK-AN:   ^{{.*}}:
//       CHECK-AN:     test.new_op
//       CHECK-AN:     test.op_2
//       CHECK-AN:     test.op_3
//       CHECK-AN:   }) : () -> ()
func.func @test_split_block() {
  "test.op_with_region"() ({
    "test.op_1"() : () -> ()
    "test.split_block_here"() : () -> ()
    "test.op_2"() : () -> ()
    "test.op_3"() : () -> ()
  }) : () -> ()
  return
}

// -----

// CHECK-AN: notifyOperationInserted: test.clone_me, was unlinked
// CHECK-AN: notifyBlockInserted into test.clone_me: was unlinked
// CHECK-AN: notifyBlockInserted into test.clone_me: was unlinked
// CHECK-AN: notifyOperationInserted: test.foo, was unlinked
// CHECK-AN: notifyOperationInserted: test.bar, was unlinked
// CHECK-AN-LABEL: func @clone_op(
// CHECK-AN:         "test.clone_me"() ({
// CHECK-AN:           "test.foo"() : () -> ()
// CHECK-AN:         ^bb1:  // no predecessors
// CHECK-AN:           "test.bar"() : () -> ()
// CHECK-AN:         }) {was_cloned} : () -> ()
// CHECK-AN:         "test.clone_me"() ({
// CHECK-AN:           "test.foo"() : () -> ()
// CHECK-AN:         ^bb1:  // no predecessors
// CHECK-AN:           "test.bar"() : () -> ()
// CHECK-AN:         }) : () -> ()
func.func @clone_op() {
  "test.clone_me"() ({
  ^bb0:
    "test.foo"() : () -> ()
  ^bb1:
    "test.bar"() : () -> ()
  }) : () -> ()
  return
}


// -----

// CHECK-AN: notifyBlockInserted into func.func: was unlinked
// CHECK-AN: notifyOperationInserted: test.op_1, was unlinked
// CHECK-AN: notifyBlockInserted into func.func: was unlinked
// CHECK-AN: notifyOperationInserted: test.op_2, was unlinked
// CHECK-AN: notifyBlockInserted into test.op_2: was unlinked
// CHECK-AN: notifyOperationInserted: test.op_3, was unlinked
// CHECK-AN: notifyOperationInserted: test.op_4, was unlinked
// CHECK-AN-LABEL: func @test_clone_region_before(
// CHECK-AN:         "test.op_1"() : () -> ()
// CHECK-AN:       ^{{.*}}:
// CHECK-AN:         "test.op_2"() ({
// CHECK-AN:           "test.op_3"() : () -> ()
// CHECK-AN:         }) : () -> ()
// CHECK-AN:         "test.op_4"() : () -> ()
// CHECK-AN:       ^{{.*}}:
// CHECK-AN:         "test.clone_region_before"() ({
func.func @test_clone_region_before() {
  "test.clone_region_before"() ({
    "test.op_1"() : () -> ()
  ^bb0:
    "test.op_2"() ({
      "test.op_3"() : () -> ()
    }) : () -> ()
    "test.op_4"() : () -> ()
  }) : () -> ()
  return
}
