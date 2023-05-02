// RUN: mlir-opt -allow-unregistered-dialect -split-input-file -test-legalize-patterns -verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: verifyDirectPattern
func.func @verifyDirectPattern() -> i32 {
  // CHECK-NEXT:  "test.legal_op_a"() <{status = "Success"}
  %result = "test.illegal_op_a"() : () -> (i32)
  // expected-remark@+1 {{op 'func.return' is not legalizable}}
  return %result : i32
}

// CHECK-LABEL: verifyLargerBenefit
func.func @verifyLargerBenefit() -> i32 {
  // CHECK-NEXT:  "test.legal_op_a"() <{status = "Success"}
  %result = "test.illegal_op_c"() : () -> (i32)
  // expected-remark@+1 {{op 'func.return' is not legalizable}}
  return %result : i32
}

// CHECK-LABEL: func private @remap_input_1_to_0()
func.func private @remap_input_1_to_0(i16)

// CHECK-LABEL: func @remap_input_1_to_1(%arg0: f64)
func.func @remap_input_1_to_1(%arg0: i64) {
  // CHECK-NEXT: "test.valid"{{.*}} : (f64)
  "test.invalid"(%arg0) : (i64) -> ()
}

// CHECK-LABEL: func @remap_call_1_to_1(%arg0: f64)
func.func @remap_call_1_to_1(%arg0: i64) {
  // CHECK-NEXT: call @remap_input_1_to_1(%arg0) : (f64) -> ()
  call @remap_input_1_to_1(%arg0) : (i64) -> ()
  // expected-remark@+1 {{op 'func.return' is not legalizable}}
  return
}

// CHECK-LABEL: func @remap_input_1_to_N({{.*}}f16, {{.*}}f16)
func.func @remap_input_1_to_N(%arg0: f32) -> f32 {
  // CHECK-NEXT: "test.return"{{.*}} : (f16, f16) -> ()
  "test.return"(%arg0) : (f32) -> ()
}

// CHECK-LABEL: func @remap_input_1_to_N_remaining_use(%arg0: f16, %arg1: f16)
func.func @remap_input_1_to_N_remaining_use(%arg0: f32) {
  // CHECK-NEXT: [[CAST:%.*]] = "test.cast"(%arg0, %arg1) : (f16, f16) -> f32
  // CHECK-NEXT: "work"([[CAST]]) : (f32) -> ()
  // expected-remark@+1 {{op 'work' is not legalizable}}
  "work"(%arg0) : (f32) -> ()
}

// CHECK-LABEL: func @remap_materialize_1_to_1(%{{.*}}: i43)
func.func @remap_materialize_1_to_1(%arg0: i42) {
  // CHECK: %[[V:.*]] = "test.cast"(%arg0) : (i43) -> i42
  // CHECK: "test.return"(%[[V]])
  "test.return"(%arg0) : (i42) -> ()
}

// CHECK-LABEL: func @remap_input_to_self
func.func @remap_input_to_self(%arg0: index) {
  // CHECK-NOT: test.cast
  // CHECK: "work"
  // expected-remark@+1 {{op 'work' is not legalizable}}
  "work"(%arg0) : (index) -> ()
}

// CHECK-LABEL: func @remap_multi(%arg0: f64, %arg1: f64) -> (f64, f64)
func.func @remap_multi(%arg0: i64, %unused: i16, %arg1: i64) -> (i64, i64) {
 // CHECK-NEXT: "test.valid"{{.*}} : (f64, f64)
 "test.invalid"(%arg0, %arg1) : (i64, i64) -> ()
}

// CHECK-LABEL: func @no_remap_nested
func.func @no_remap_nested() {
  // CHECK-NEXT: "foo.region"
  // expected-remark@+1 {{op 'foo.region' is not legalizable}}
  "foo.region"() ({
    // CHECK-NEXT: ^bb0(%{{.*}}: i64, %{{.*}}: i16, %{{.*}}: i64):
    ^bb0(%i0: i64, %unused: i16, %i1: i64):
      // CHECK-NEXT: "test.valid"{{.*}} : (i64, i64)
      "test.invalid"(%i0, %i1) : (i64, i64) -> ()
  }) : () -> ()
  // expected-remark@+1 {{op 'func.return' is not legalizable}}
  return
}

// CHECK-LABEL: func @remap_moved_region_args
func.func @remap_moved_region_args() {
  // CHECK-NEXT: return
  // CHECK-NEXT: ^bb1(%{{.*}}: f64, %{{.*}}: f64, %{{.*}}: f16, %{{.*}}: f16):
  // CHECK-NEXT: "test.cast"{{.*}} : (f16, f16) -> f32
  // CHECK-NEXT: "test.valid"{{.*}} : (f64, f64, f32)
  "test.region"() ({
    ^bb1(%i0: i64, %unused: i16, %i1: i64, %2: f32):
      "test.invalid"(%i0, %i1, %2) : (i64, i64, f32) -> ()
  }) : () -> ()
  // expected-remark@+1 {{op 'func.return' is not legalizable}}
  return
}

// CHECK-LABEL: func @remap_cloned_region_args
func.func @remap_cloned_region_args() {
  // CHECK-NEXT: return
  // CHECK-NEXT: ^bb1(%{{.*}}: f64, %{{.*}}: f64, %{{.*}}: f16, %{{.*}}: f16):
  // CHECK-NEXT: "test.cast"{{.*}} : (f16, f16) -> f32
  // CHECK-NEXT: "test.valid"{{.*}} : (f64, f64, f32)
  "test.region"() ({
    ^bb1(%i0: i64, %unused: i16, %i1: i64, %2: f32):
      "test.invalid"(%i0, %i1, %2) : (i64, i64, f32) -> ()
  }) {legalizer.should_clone} : () -> ()
  // expected-remark@+1 {{op 'func.return' is not legalizable}}
  return
}

// CHECK-LABEL: func @remap_drop_region
func.func @remap_drop_region() {
  // CHECK-NEXT: return
  // CHECK-NEXT: }
  "test.drop_region_op"() ({
    ^bb1(%i0: i64, %unused: i16, %i1: i64, %2: f32):
      "test.invalid"(%i0, %i1, %2) : (i64, i64, f32) -> ()
  }) : () -> ()
  // expected-remark@+1 {{op 'func.return' is not legalizable}}
  return
}

// CHECK-LABEL: func @dropped_input_in_use
func.func @dropped_input_in_use(%arg: i16, %arg2: i64) {
  // CHECK-NEXT: "test.cast"{{.*}} : () -> i16
  // CHECK-NEXT: "work"{{.*}} : (i16)
  // expected-remark@+1 {{op 'work' is not legalizable}}
  "work"(%arg) : (i16) -> ()
}

// CHECK-LABEL: func @up_to_date_replacement
func.func @up_to_date_replacement(%arg: i8) -> i8 {
  // CHECK-NEXT: return
  %repl_1 = "test.rewrite"(%arg) : (i8) -> i8
  %repl_2 = "test.rewrite"(%repl_1) : (i8) -> i8
  // expected-remark@+1 {{op 'func.return' is not legalizable}}
  return %repl_2 : i8
}

// CHECK-LABEL: func @remove_foldable_op
// CHECK-SAME:                          (%[[ARG_0:[a-z0-9]*]]: i32)
func.func @remove_foldable_op(%arg0 : i32) -> (i32) {
  // CHECK-NEXT: return %[[ARG_0]]
  %0 = "test.op_with_region_fold"(%arg0) ({
    "foo.op_with_region_terminator"() : () -> ()
  }) : (i32) -> (i32)
  // expected-remark@+1 {{op 'func.return' is not legalizable}}
  return %0 : i32
}

// CHECK-LABEL: @create_block
func.func @create_block() {
  // Check that we created a block with arguments.
  // CHECK-NOT: test.create_block
  // CHECK: ^{{.*}}(%{{.*}}: i32, %{{.*}}: i32):
  "test.create_block"() : () -> ()

  // expected-remark@+1 {{op 'func.return' is not legalizable}}
  return
}

// CHECK-LABEL: @bounded_recursion
func.func @bounded_recursion() {
  // CHECK: test.recursive_rewrite 0
  test.recursive_rewrite 3
  // expected-remark@+1 {{op 'func.return' is not legalizable}}
  return
}

// -----

// expected-remark@+1 {{applyPartialConversion failed}}
builtin.module {

  func.func @fail_to_convert_illegal_op() -> i32 {
    // expected-error@+1 {{failed to legalize operation 'test.illegal_op_f'}}
    %result = "test.illegal_op_f"() : () -> (i32)
    return %result : i32
  }

}

// -----

// expected-remark@+1 {{applyPartialConversion failed}}
builtin.module {

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
builtin.module {

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
func.func @undo_block_arg_replace() {
  // expected-remark@+1 {{op 'test.undo_block_arg_replace' is not legalizable}}
  "test.undo_block_arg_replace"() ({
  ^bb0(%arg0: i32):
    // CHECK: ^bb0(%[[ARG:.*]]: i32):
    // CHECK-NEXT: "test.return"(%[[ARG]]) : (i32)

    "test.return"(%arg0) : (i32) -> ()
  }) : () -> ()
  // expected-remark@+1 {{op 'func.return' is not legalizable}}
  return
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

// Check that a conversion pattern on `test.blackhole` can mark the producer
// for deletion.
// CHECK-LABEL: @blackhole
func.func @blackhole() {
  %input = "test.blackhole_producer"() : () -> (i32)
  "test.blackhole"(%input) : (i32) -> ()
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

// The "passthrough_fold" folder will naively return its operand, but we don't
// want to fold here because of the type mismatch.
func.func @typemismatch(%arg: f32) -> i32 {
  // expected-remark@+1 {{op 'test.passthrough_fold' is not legalizable}}
  %0 = "test.passthrough_fold"(%arg) : (f32) -> (i32)
  "test.return"(%0) : (i32) -> ()
}

// -----

// expected-remark @below {{applyPartialConversion failed}}
module {
  func.func private @callee(%0 : f32) -> f32

  func.func @caller( %arg: f32) {
    // expected-error @below {{failed to legalize}}
    %1 = func.call @callee(%arg) : (f32) -> f32
    return
  }
}
