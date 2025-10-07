// RUN: mlir-opt -allow-unregistered-dialect -split-input-file -test-legalize-patterns -verify-diagnostics -profile-actions-to=- %s | FileCheck %s

// expected-remark@+1 {{applyPartialConversion failed}}
module {
func.func @fail_to_convert_illegal_op_in_region() {
  // expected-error@+1 {{failed to legalize operation 'test.region_builder'}}
  "test.region_builder"() : () -> ()
  return
}
}

// -----

// Check that the entry block arguments of a region are untouched in the case
// of failure.

// expected-remark@+1 {{applyPartialConversion failed}}
module {
func.func @fail_to_convert_region() {
  // CHECK: "test.region"
  // CHECK-NEXT: ^bb{{.*}}(%{{.*}}: i64):
  "test.region"() ({
    ^bb1(%i0: i64):
      // expected-error@+1 {{failed to legalize operation 'test.region_builder'}}
      "test.region_builder"() : () -> ()
      "test.valid"() : () -> ()
  }) : () -> ()
  return
}
}

// -----

// CHECK-LABEL: @create_illegal_block
func.func @create_illegal_block() {
  // Check that we can undo block creation, i.e. that the block was removed.
  // CHECK: test.create_illegal_block
  // CHECK-NOT: ^{{.*}}(%{{.*}}: i32, %{{.*}}: i32):
  // expected-remark@+1 {{op 'test.create_illegal_block' is not legalizable}}
  "test.create_illegal_block"() : () -> ()

  // expected-remark@+1 {{op 'func.return' is not legalizable}}
  return
}

// -----

// CHECK-LABEL: @undo_block_arg_replace
// expected-remark@+1{{applyPartialConversion failed}}
module {
func.func @undo_block_arg_replace() {
  "test.legal_op"() ({
  ^bb0(%arg0: i32, %arg1: i16):
    // CHECK: ^bb0(%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i16):
    // CHECK-NEXT: "test.value_replace"(%[[ARG0]], %[[ARG1]]) {trigger_rollback}
    // CHECK-NEXT: "test.return"(%[[ARG0]]) : (i32)

    // expected-error@+1{{failed to legalize operation 'test.value_replace' that was explicitly marked illegal}}
    "test.value_replace"(%arg0, %arg1) {trigger_rollback} : (i32, i16) -> ()
    "test.return"(%arg0) : (i32) -> ()
  }) : () -> ()
  return
}
}

// -----

// The op in this function is rewritten to itself (and thus remains illegal) by
// a pattern that removes its second block after adding an operation into it.
// Check that we can undo block removal successfully.
// CHECK-LABEL: @undo_block_erase
func.func @undo_block_erase() {
  // CHECK: test.undo_block_erase
  "test.undo_block_erase"() ({
    // expected-remark@-1 {{not legalizable}}
    // CHECK: "unregistered.return"()[^[[BB:.*]]]
    "unregistered.return"()[^bb1] : () -> ()
    // expected-remark@-1 {{not legalizable}}
  // CHECK: ^[[BB]]
  ^bb1:
    // CHECK: unregistered.return
    "unregistered.return"() : () -> ()
    // expected-remark@-1 {{not legalizable}}
  }) : () -> ()
}

// -----

// The op in this function is attempted to be rewritten to another illegal op
// with an attached region containing an invalid terminator. The terminator is
// created before the parent op. The deletion should not crash when deleting
// created ops in the inverse order, i.e. deleting the parent op and then the
// child op.
// CHECK-LABEL: @undo_child_created_before_parent
func.func @undo_child_created_before_parent() {
  // expected-remark@+1 {{is not legalizable}}
  "test.illegal_op_with_region_anchor"() : () -> ()
  // expected-remark@+1 {{op 'func.return' is not legalizable}}
  return
}

// -----

// expected-remark@+1 {{applyPartialConversion failed}}
builtin.module {
func.func @create_unregistered_op_in_pattern() -> i32 {
  // expected-error@+1 {{failed to legalize operation 'test.illegal_op_g'}}
  %0 = "test.illegal_op_g"() : () -> (i32)
  "test.return"(%0) : (i32) -> ()
}
}

// -----

// CHECK-LABEL: func @test_move_op_before_rollback()
func.func @test_move_op_before_rollback() {
  // CHECK: "test.one_region_op"()
  // CHECK: "test.hoist_me"()
  "test.one_region_op"() ({
    // expected-remark @below{{'test.hoist_me' is not legalizable}}
    %0 = "test.hoist_me"() : () -> (i32)
    "test.valid"(%0) : (i32) -> ()
  }) : () -> ()
  "test.return"() : () -> ()
}

// -----

// CHECK-LABEL: func @test_properties_rollback()
func.func @test_properties_rollback() {
  // CHECK: test.with_properties a = 32,
  // expected-remark @below{{op 'test.with_properties' is not legalizable}}
  test.with_properties
      a = 32, b = "foo", c = "bar", flag = true, array = [1, 2, 3, 4], array32 = [5, 6]
      {modify_inplace}
  "test.return"() : () -> ()
}

// -----

// expected-remark@+1 {{applyPartialConversion failed}}
builtin.module {
// Test that region cloning can be properly undone.
func.func @test_undo_region_clone() {
  "test.region"() ({
    ^bb1(%i0: i64):
      "test.invalid"(%i0) : (i64) -> ()
  }) {legalizer.should_clone} : () -> ()

  // expected-error@+1 {{failed to legalize operation 'test.illegal_op_f'}}
  %ignored = "test.illegal_op_f"() : () -> (i32)
  "test.return"() : () -> ()
}
}

// -----

// expected-remark@+1 {{applyPartialConversion failed}}
builtin.module {
func.func @create_unregistered_op_in_pattern() -> i32 {
  // expected-error@+1 {{failed to legalize operation 'test.illegal_op_g'}}
  %0 = "test.illegal_op_g"() : () -> (i32)
  "test.return"(%0) : (i32) -> ()
}
}
