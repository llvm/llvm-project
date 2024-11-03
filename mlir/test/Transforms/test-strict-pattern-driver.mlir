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

// CHECK-EN-LABEL: func @test_insert_same_op
//  CHECK-EN-SAME:     {pattern_driver_all_erased = false, pattern_driver_changed = true}
//       CHECK-EN:   "test.insert_same_op"() {skip = true}
//       CHECK-EN:   "test.insert_same_op"() {skip = true}
func.func @test_insert_same_op() {
  %0 = "test.insert_same_op"() : () -> (i32)
  return
}

// -----

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
