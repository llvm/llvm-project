// RUN: mlir-opt -allow-unregistered-dialect -split-input-file -test-legalize-patterns="allow-pattern-rollback=1" -verify-diagnostics %s | FileCheck %s
// RUN: mlir-opt -allow-unregistered-dialect -split-input-file -test-legalize-patterns="allow-pattern-rollback=1" -verify-diagnostics -profile-actions-to=- %s | FileCheck %s --check-prefix=CHECK-PROFILER
// RUN: mlir-opt -allow-unregistered-dialect -split-input-file -test-legalize-patterns="allow-pattern-rollback=0" -verify-diagnostics %s | FileCheck %s

// CHECK-PROFILER: "name": "pass-execution", "cat": "PERF", "ph": "B"
// CHECK-PROFILER: "name": "apply-conversion", "cat": "PERF", "ph": "B"
// CHECK-PROFILER: "name": "apply-pattern", "cat": "PERF", "ph": "B"
// CHECK-PROFILER: "name": "apply-pattern", "cat": "PERF", "ph": "E"
// CHECK-PROFILER: "name": "apply-conversion", "cat": "PERF", "ph": "E"
// CHECK-PROFILER: "name": "pass-execution", "cat": "PERF", "ph": "E"

// Note: Listener notifications appear after the pattern application because
// the conversion driver sends all notifications at the end of the conversion
// in bulk.
//      CHECK: notifyOperationInserted: test.legal_op_a, was unlinked
// CHECK-NEXT: notifyOperationReplaced: test.illegal_op_a
// CHECK-NEXT: notifyOperationModified: func.return
// CHECK-NEXT: notifyOperationErased: test.illegal_op_a
// CHECK-LABEL: verifyDirectPattern
func.func @verifyDirectPattern() -> i32 {
  // CHECK-NEXT:  "test.legal_op_a"() <{status = "Success"}
  %result = "test.illegal_op_a"() : () -> (i32)
  // expected-remark@+1 {{op 'func.return' is not legalizable}}
  return %result : i32
}

// -----

//      CHECK: notifyOperationInserted: test.illegal_op_e, was unlinked
// CHECK-NEXT: notifyOperationReplaced: test.illegal_op_c
// CHECK-NEXT: notifyOperationModified: func.return
// CHECK-NEXT: notifyOperationErased: test.illegal_op_c
// CHECK-NEXT: notifyOperationInserted: test.legal_op_a, was unlinked
// CHECK-NEXT: notifyOperationReplaced: test.illegal_op_e
// Note: func.return is modified a second time when running in no-rollback
//       mode.
//      CHECK: notifyOperationErased: test.illegal_op_e

// CHECK-LABEL: verifyLargerBenefit
func.func @verifyLargerBenefit() -> i32 {
  // CHECK-NEXT:  "test.legal_op_a"() <{status = "Success"}
  %result = "test.illegal_op_c"() : () -> (i32)
  // expected-remark@+1 {{op 'func.return' is not legalizable}}
  return %result : i32
}

// -----

// CHECK: notifyOperationModified: func.func
// Note: No block insertion because this function is external and no block
// signature conversion is performed.

// CHECK-LABEL: func private @remap_input_1_to_0()
func.func private @remap_input_1_to_0(i16)

// -----

// CHECK-LABEL: func @remap_input_1_to_1(%arg0: f64)
func.func @remap_input_1_to_1(%arg0: i64) {
  // CHECK-NEXT: "test.valid"{{.*}} : (f64)
  "test.invalid"(%arg0) : (i64) -> ()
}

// CHECK: func @remap_call_1_to_1(%arg0: f64)
func.func @remap_call_1_to_1(%arg0: i64) {
  // CHECK-NEXT: call @remap_input_1_to_1(%arg0) : (f64) -> ()
  call @remap_input_1_to_1(%arg0) : (i64) -> ()
  // expected-remark@+1 {{op 'func.return' is not legalizable}}
  return
}

// -----

// Block signature conversion: new block is inserted.
// CHECK:      notifyBlockInserted into func.func: was unlinked

// Contents of the old block are moved to the new block.
// CHECK-NEXT: notifyOperationInserted: test.return

// The old block is erased.
// CHECK-NEXT: notifyBlockErased

// The function op gets a new type attribute.
// CHECK-NEXT: notifyOperationModified: func.func

// "test.return" is replaced.
// CHECK-NEXT: notifyOperationInserted: test.return, was unlinked
// CHECK-NEXT: notifyOperationReplaced: test.return
// CHECK-NEXT: notifyOperationErased: test.return

// CHECK-LABEL: func @remap_input_1_to_N({{.*}}f16, {{.*}}f16)
func.func @remap_input_1_to_N(%arg0: f32) -> f32 {
  // CHECK-NEXT: "test.return"{{.*}} : (f16, f16) -> ()
  "test.return"(%arg0) : (f32) -> ()
}

// -----

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

// -----

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

// -----

// CHECK-LABEL: func @no_remap_nested
func.func @no_remap_nested() {
  // CHECK-NEXT: "foo.region"
  // expected-remark@+1 {{op 'foo.region' is not legalizable}}
  "foo.region"() ({
    // CHECK-NEXT: ^bb0(%{{.*}}: f64, %{{.*}}: i16, %{{.*}}: f64):
    ^bb0(%i0: f64, %unused: i16, %i1: f64):
      // CHECK-NEXT: "test.valid"{{.*}} : (f64, f64)
      "test.invalid"(%i0, %i1) : (f64, f64) -> ()
  }) : () -> ()
  // expected-remark@+1 {{op 'func.return' is not legalizable}}
  return
}

// -----

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

// -----

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

// -----

// CHECK-LABEL: func @dropped_input_in_use
func.func @dropped_input_in_use(%arg: i16, %arg2: i64) {
  // CHECK-NEXT: "test.cast"{{.*}} : () -> i16
  // CHECK-NEXT: "work"{{.*}} : (i16)
  // expected-remark@+1 {{op 'work' is not legalizable}}
  "work"(%arg) : (i16) -> ()
}

// -----

// CHECK-LABEL: func @up_to_date_replacement
func.func @up_to_date_replacement(%arg: i8) -> i8 {
  // CHECK-NEXT: return
  %repl_1 = "test.rewrite"(%arg) : (i8) -> i8
  %repl_2 = "test.rewrite"(%repl_1) : (i8) -> i8
  // expected-remark@+1 {{op 'func.return' is not legalizable}}
  return %repl_2 : i8
}

// -----

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

// -----

// CHECK-LABEL: @create_block
func.func @create_block() {
  // Check that we created a block with arguments.
  // CHECK-NOT: test.create_block
  // CHECK: ^{{.*}}(%{{.*}}: i32, %{{.*}}: i32):
  "test.create_block"() : () -> ()

  // expected-remark@+1 {{op 'func.return' is not legalizable}}
  return
}

// -----

//      CHECK: notifyOperationModified: test.recursive_rewrite
// CHECK-NEXT: notifyOperationModified: test.recursive_rewrite
// CHECK-NEXT: notifyOperationModified: test.recursive_rewrite

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

// CHECK-LABEL: @replace_block_arg_1_to_n
func.func @replace_block_arg_1_to_n() {
  // CHECK: "test.block_arg_replace"
  "test.block_arg_replace"() ({
  ^bb0(%arg0: i32, %arg1: i16):
    // CHECK: ^bb0(%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i16):
    // CHECK: %[[cast:.*]] = "test.cast"(%[[ARG1]], %[[ARG1]]) : (i16, i16) -> i32
    // CHECK-NEXT: "test.return"(%[[cast]]) : (i32)
    "test.return"(%arg0) : (i32) -> ()
  }) : () -> ()
  "test.return"() : () -> ()
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

module {
// CHECK-LABEL: func.func private @callee() -> (f16, f16)
func.func private @callee() -> (f32, i24)

// CHECK: func.func @caller()
func.func @caller() {
  // f32 is converted to (f16, f16).
  // i24 is converted to ().
  // CHECK: %[[call:.*]]:2 = call @callee() : () -> (f16, f16)
  %0:2 = func.call @callee() : () -> (f32, i24)

  // CHECK-DAG: %[[cast1:.*]] = "test.cast"() : () -> i24
  // CHECK-DAG: %[[cast0:.*]] = "test.cast"(%[[call]]#0, %[[call]]#1) : (f16, f16) -> f32
  // CHECK: "test.some_user"(%[[cast0]], %[[cast1]]) : (f32, i24) -> ()
  // expected-remark @below{{'test.some_user' is not legalizable}}
  "test.some_user"(%0#0, %0#1) : (f32, i24) -> ()
  "test.return"() : () -> ()
}
}

// -----

//      CHECK: func.func @use_of_replaced_bbarg(
// CHECK-SAME:     %[[arg0:.*]]: f64)
//      CHECK:   "test.valid"(%[[arg0]])
func.func @use_of_replaced_bbarg(%arg0: i64) {
  %0 = "test.op_with_region_fold"(%arg0) ({
    "foo.op_with_region_terminator"() : () -> ()
  }) : (i64) -> (i64)
  "test.invalid"(%0) : (i64) -> ()
}

// -----

// CHECK-LABEL: @fold_legalization
func.func @fold_legalization() -> i32 {
  // CHECK: op_in_place_self_fold
  // CHECK-SAME: folded
  %1 = "test.op_in_place_self_fold"() : () -> (i32)
  "test.return"(%1) : (i32) -> ()
}

// -----

// CHECK-LABEL: func @convert_detached_signature()
//       CHECK:   "test.legal_op"() ({
//       CHECK:   ^bb0(%arg0: f64):
//       CHECK:     "test.return"() : () -> ()
//       CHECK:   }) : () -> ()
func.func @convert_detached_signature() {
  "test.detached_signature_conversion"() ({
  ^bb0(%arg0: i64):
    "test.return"() : () -> ()
  }) : () -> ()
  "test.return"() : () -> ()
}

// -----

// CHECK: notifyOperationReplaced: test.erase_op
// CHECK: notifyOperationErased: test.dummy_op_lvl_2
// CHECK: notifyBlockErased
// CHECK: notifyOperationErased: test.dummy_op_lvl_1
// CHECK: notifyBlockErased
// CHECK: notifyOperationErased: test.erase_op
// CHECK: notifyOperationInserted: test.valid, was unlinked
// CHECK: notifyOperationReplaced: test.drop_operands_and_replace_with_valid
// CHECK: notifyOperationErased: test.drop_operands_and_replace_with_valid

// CHECK-LABEL: func @circular_mapping()
//  CHECK-NEXT:   "test.valid"() : () -> ()
func.func @circular_mapping() {
  // Regression test that used to crash due to circular
  // unrealized_conversion_cast ops. 
  %0 = "test.erase_op"() ({
    "test.dummy_op_lvl_1"() ({
      "test.dummy_op_lvl_2"() : () -> ()
    }) : () -> ()
  }): () -> (i64)
  "test.drop_operands_and_replace_with_valid"(%0) : (i64) -> ()
}

// -----

// CHECK-LABEL: func @test_duplicate_block_arg()
//       CHECK:   test.convert_block_args  is_legal duplicate {
//       CHECK:   ^{{.*}}(%[[arg0:.*]]: i64, %[[arg1:.*]]: i64):
//       CHECK:     "test.valid"(%[[arg0]], %[[arg1]])
//       CHECK:   }
func.func @test_duplicate_block_arg() {
  test.convert_block_args duplicate {
  ^bb0(%arg0: i64):
    "test.repetitive_1_to_n_consumer"(%arg0) : (i64) -> ()
  } : () -> ()
  "test.return"() : () -> ()
}

// -----

// CHECK-LABEL: func @test_remap_block_arg()
//       CHECK:      %[[repl:.*]] = "test.legal_op"() : () -> i32
//       CHECK:      test.convert_block_args %[[repl]]  is_legal replace_with_operand {
//       CHECK-NEXT:   "test.valid"(%[[repl]], %[[repl]])
//       CHECK:      }
func.func @test_remap_block_arg() {
  %0 = "test.legal_op"() : () -> (i32)
  test.convert_block_args %0 replace_with_operand {
  ^bb0(%arg0: i32):
    "test.repetitive_1_to_n_consumer"(%arg0) : (i32) -> ()
  } : (i32) -> ()
  "test.return"() : () -> ()
}

// -----

// CHECK: notifyOperationInserted: test.step_1
// CHECK: notifyOperationReplaced: test.multiple_1_to_n_replacement
// CHECK: notifyOperationErased: test.multiple_1_to_n_replacement
// CHECK: notifyOperationInserted: test.legal_op
// CHECK: notifyOperationReplaced: test.step_1
// CHECK: notifyOperationErased: test.step_1

// CHECK-LABEL: func @test_multiple_1_to_n_replacement()
//       CHECK:   %[[legal_op:.*]]:4 = "test.legal_op"() : () -> (f16, f16, f16, f16)
// Note: There is a bug in the rollback-based conversion driver: it emits a
// "test.cast" : (f16, f16, f16, f16) -> f16, when it should be emitting
// three consecutive casts of (f16, f16) -> f16.
//       CHECK:   "test.valid"(%{{.*}}) : (f16) -> ()
func.func @test_multiple_1_to_n_replacement() {
  %0 = "test.multiple_1_to_n_replacement"() : () -> (f16)
  "test.invalid"(%0) : (f16) -> ()
}

// -----

// CHECK-LABEL: func @test_lookup_without_converter
//       CHECK:   %[[producer:.*]] = "test.valid_producer"() : () -> i16
//       CHECK:   %[[cast:.*]] = "test.cast"(%[[producer]]) : (i16) -> f64
//       CHECK:   "test.valid_consumer"(%[[cast]]) : (f64) -> ()
//       CHECK:   "test.valid_consumer"(%[[producer]]) : (i16) -> ()
func.func @test_lookup_without_converter() {
  %0 = "test.replace_with_valid_producer"() {type = i16} : () -> (i64)
  "test.replace_with_valid_consumer"(%0) {with_converter} : (i64) -> ()
  // Make sure that the second "replace_with_valid_consumer" lowering does not
  // lookup the materialization that was created for the above op.
  "test.replace_with_valid_consumer"(%0) : (i64) -> ()
  // expected-remark@+1 {{op 'func.return' is not legalizable}}
  return
}

// -----
// expected-remark@-1 {{applyPartialConversion failed}}

func.func @test_skip_1to1_pattern(%arg0: f32) {
  // expected-error@+1 {{failed to legalize operation 'test.type_consumer'}}
  "test.type_consumer"(%arg0) : (f32) -> ()
  return
}

// -----

// Demonstrate that the pattern generally works, but only for 1:1 type
// conversions.

// CHECK-LABEL: @test_working_1to1_pattern(
func.func @test_working_1to1_pattern(%arg0: f16) {
  // CHECK-NEXT: "test.return"() : () -> ()
  "test.type_consumer"(%arg0) : (f16) -> ()
  "test.return"() : () -> ()
}
