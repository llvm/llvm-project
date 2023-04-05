// RUN: mlir-opt %s --test-transform-dialect-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics

transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  // expected-remark @below {{applying transformation}}
  transform.test_transform_op
}

// -----

transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  %0 = transform.test_produce_self_handle_or_forward_operand { foo = "bar" }
  // expected-remark @below {{succeeded}}
  transform.test_consume_operand_of_op_kind_or_fail %0, "transform.test_produce_self_handle_or_forward_operand"
}

// -----

transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  %0 = transform.test_produce_self_handle_or_forward_operand { foo = "bar" }
  // expected-error @below {{expected the operand to be associated a payload op of kind transform.sequence got transform.test_produce_self_handle_or_forward_operand}}
  transform.test_consume_operand_of_op_kind_or_fail %0, "transform.sequence"
}

// -----

// It is okay to have multiple handles to the same payload op as long
// as only one of them is consumed. The expensive checks mode is necessary
// to detect double-consumption.
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  %0 = transform.test_produce_self_handle_or_forward_operand { foo = "bar" }
  %1 = transform.test_copy_payload %0
  // expected-remark @below {{succeeded}}
  transform.test_consume_operand_of_op_kind_or_fail %0, "transform.test_produce_self_handle_or_forward_operand"
}

// -----

transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  sequence %arg0 : !pdl.operation failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    // expected-remark @below {{applying transformation "a"}}
    test_transform_op "a"
    // expected-remark @below {{applying transformation "b"}}
    test_transform_op "b"
    // expected-remark @below {{applying transformation "c"}}
    test_transform_op "c"
  }
  // expected-remark @below {{applying transformation "d"}}
  test_transform_op "d"
  // expected-remark @below {{applying transformation "e"}}
  test_transform_op "e"
}

// -----

transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  %0 = test_produce_self_handle_or_forward_operand
  sequence %0 : !pdl.operation failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    // expected-remark @below {{succeeded}}
    test_consume_operand_of_op_kind_or_fail %arg1, "transform.test_produce_self_handle_or_forward_operand"
  }
}

// -----

transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  %0 = sequence %arg0 : !pdl.operation -> !pdl.operation failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %1 = test_produce_self_handle_or_forward_operand
    yield %1 : !pdl.operation
  }
  // expected-remark @below {{succeeded}}
  test_consume_operand_of_op_kind_or_fail %0, "transform.test_produce_self_handle_or_forward_operand"
}

// -----

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  sequence %arg0 : !pdl.operation failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = pdl_match @some in %arg1 : (!pdl.operation) -> !pdl.operation
    test_print_remark_at_operand %0, "matched" : !pdl.operation
  }

  pdl.pattern @some : benefit(1) {
    %0 = pdl.operation "test.some_op"
    pdl.rewrite %0 with "transform.dialect"
  }

  pdl.pattern @other : benefit(1) {
    %0 = pdl.operation "test.other_op"
    pdl.rewrite %0 with "transform.dialect"
  }
}

// expected-remark @below {{matched}}
"test.some_op"() : () -> ()
"test.other_op"() : () -> ()
// expected-remark @below {{matched}}
"test.some_op"() : () -> ()

// -----

// expected-remark @below {{parent function}}
func.func @foo() {
  %0 = arith.constant 0 : i32
  return
}

// expected-remark @below {{parent function}}
func.func @bar() {
  %0 = arith.constant 0 : i32
  %1 = arith.constant 1 : i32
  return
}

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  pdl.pattern @const : benefit(1) {
    %r = pdl.types
    %0 = pdl.operation "arith.constant" -> (%r : !pdl.range<type>)
    pdl.rewrite %0 with "transform.dialect"
  }

  transform.sequence %arg0 : !pdl.operation failures(propagate) {
  ^bb1(%arg1: !pdl.operation):
    %f = pdl_match @const in %arg1 : (!pdl.operation) -> !pdl.operation
    // CHECK: %{{.+}} = get_closest_isolated_parent %{{.+}}
    %m = get_closest_isolated_parent %f : (!pdl.operation) -> !pdl.operation
    test_print_remark_at_operand %m, "parent function" : !pdl.operation
  }
}

// -----

func.func @foo() {
  %0 = arith.constant 0 : i32
  return
}

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  pdl.pattern @match_func : benefit(1) {
    %0 = pdl.operands
    %1 = pdl.types
    %2 = pdl.operation "func.func"(%0 : !pdl.range<value>) -> (%1 : !pdl.range<type>)
    pdl.rewrite %2 with "transform.dialect"
  }

  transform.sequence %arg0 : !pdl.operation failures(propagate) {
  ^bb1(%arg1: !pdl.operation):
    // This is necessary to run the transformation on something other than the
    // top-level module, "alternatives" cannot be run on that.
    %0 = pdl_match @match_func in %arg1 : (!pdl.operation) -> !pdl.operation
    transform.alternatives %0 : !pdl.operation {
    ^bb2(%arg2: !pdl.operation):
      %1 = transform.test_produce_self_handle_or_forward_operand
      // This operation fails, which triggers the next alternative without
      // reporting the error.
      transform.test_consume_operand_of_op_kind_or_fail %1, "transform.sequence"
    }, {
    ^bb2(%arg2: !pdl.operation):
      %1 = transform.test_produce_self_handle_or_forward_operand
      // expected-remark @below {{succeeded}}
      transform.test_consume_operand_of_op_kind_or_fail %1, "transform.test_produce_self_handle_or_forward_operand"
    }
  }
}

// -----

func.func private @bar()

func.func @foo() {
  call @bar() : () -> ()
  return
}

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  pdl.pattern @match_call : benefit(1) {
    %0 = pdl.operands
    %1 = pdl.types
    %2 = pdl.operation "func.call"(%0 : !pdl.range<value>) -> (%1 : !pdl.range<type>)
    pdl.rewrite %2 with "transform.dialect"
  }

  transform.sequence %arg0 : !pdl.operation failures(propagate) {
  ^bb1(%arg1: !pdl.operation):
    %0 = pdl_match @match_call in %arg1 : (!pdl.operation) -> !pdl.operation
    %1 = get_closest_isolated_parent %0 : (!pdl.operation) -> !pdl.operation
    // expected-error @below {{all alternatives failed}}
    transform.alternatives %1 : !pdl.operation {
    ^bb2(%arg2: !pdl.operation):
      %2 = transform.pdl_match @match_call in %arg2 : (!pdl.operation) -> !pdl.operation
      // expected-remark @below {{applying}}
      transform.test_emit_remark_and_erase_operand %2, "applying" {fail_after_erase}
    }
  }
}

// -----

func.func private @bar()

func.func @foo() {
  // expected-remark @below {{still here}}
  call @bar() : () -> ()
  return
}

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  pdl.pattern @match_call : benefit(1) {
    %0 = pdl.operands
    %1 = pdl.types
    %2 = pdl.operation "func.call"(%0 : !pdl.range<value>) -> (%1 : !pdl.range<type>)
    pdl.rewrite %2 with "transform.dialect"
  }

  transform.sequence %arg0 : !pdl.operation failures(propagate) {
  ^bb1(%arg1: !pdl.operation):
    %0 = pdl_match @match_call in %arg1 : (!pdl.operation) -> !pdl.operation
    %1 = get_closest_isolated_parent %0 : (!pdl.operation) -> !pdl.operation
    transform.alternatives %1 : !pdl.operation {
    ^bb2(%arg2: !pdl.operation):
      %2 = transform.pdl_match @match_call in %arg2 : (!pdl.operation) -> !pdl.operation
      // expected-remark @below {{applying}}
      transform.test_emit_remark_and_erase_operand %2, "applying" {fail_after_erase}
    }, {
    ^bb2(%arg2: !pdl.operation):
      %2 = transform.pdl_match @match_call in %arg2 : (!pdl.operation) -> !pdl.operation
      transform.test_print_remark_at_operand %2, "still here" : !pdl.operation
      // This alternative succeeds.
    }, {
    ^bb2(%arg2: !pdl.operation):
      // This alternative is never run, so we must not have a remark here.
      %2 = transform.pdl_match @match_call in %arg2 : (!pdl.operation) -> !pdl.operation
      transform.test_emit_remark_and_erase_operand %2, "should not happen" {fail_after_erase}
    }
  }
}

// -----

func.func private @bar()

// CHECK-LABEL: @erase_call
func.func @erase_call() {
  // CHECK-NOT: call @bar
  call @bar() : () -> ()
  return
}

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  pdl.pattern @match_call : benefit(1) {
    %0 = pdl.operands
    %1 = pdl.types
    %2 = pdl.operation "func.call"(%0 : !pdl.range<value>) -> (%1 : !pdl.range<type>)
    pdl.rewrite %2 with "transform.dialect"
  }

  transform.sequence %arg0 : !pdl.operation failures(propagate) {
  ^bb1(%arg1: !pdl.operation):
    %0 = pdl_match @match_call in %arg1 : (!pdl.operation) -> !pdl.operation
    %1 = get_closest_isolated_parent %0 : (!pdl.operation) -> !pdl.operation
    transform.alternatives %1 : !pdl.operation {
    ^bb2(%arg2: !pdl.operation):
      %2 = transform.pdl_match @match_call in %arg2 : (!pdl.operation) -> !pdl.operation
      // expected-remark @below {{applying}}
      transform.test_emit_remark_and_erase_operand %2, "applying" {fail_after_erase}
    }, {
    ^bb2(%arg2: !pdl.operation):
      %2 = transform.pdl_match @match_call in %arg2 : (!pdl.operation) -> !pdl.operation
      // expected-remark @below {{applying second time}}
      transform.test_emit_remark_and_erase_operand %2, "applying second time"
    }
  }
}

// -----

func.func private @bar()

func.func @foo() {
  call @bar() : () -> ()
  return
}

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  pdl.pattern @match_call : benefit(1) {
    %0 = pdl.operands
    %1 = pdl.types
    %2 = pdl.operation "func.call"(%0 : !pdl.range<value>) -> (%1 : !pdl.range<type>)
    pdl.rewrite %2 with "transform.dialect"
  }

  transform.sequence %arg0 : !pdl.operation failures(propagate) {
  ^bb1(%arg1: !pdl.operation):
    %0 = pdl_match @match_call in %arg1 : (!pdl.operation) -> !pdl.operation
    %1 = get_closest_isolated_parent %0 : (!pdl.operation) -> !pdl.operation
    %2 = transform.alternatives %1 : !pdl.operation -> !pdl.operation {
    ^bb2(%arg2: !pdl.operation):
      %3 = transform.pdl_match @match_call in %arg2 : (!pdl.operation) -> !pdl.operation
      // expected-remark @below {{applying}}
      transform.test_emit_remark_and_erase_operand %3, "applying" {fail_after_erase}
      %4 = transform.test_produce_self_handle_or_forward_operand %3
      transform.yield %4 : !pdl.operation
    }, {
    ^bb2(%arg2: !pdl.operation):
      %4 = transform.test_produce_self_handle_or_forward_operand
      transform.yield %4 : !pdl.operation
    }
    // The first alternative failed, so the returned value is taken from the
    // second alternative, associated test_produce_self_handle_or_forward_operand rather
    // than pdl_match.
    // expected-remark @below {{succeeded}}
    transform.test_consume_operand_of_op_kind_or_fail %2, "transform.test_produce_self_handle_or_forward_operand"
  }
}

// -----

// expected-note @below {{scope}}
module {
  func.func @foo() {
    %0 = arith.constant 0 : i32
    return
  }

  func.func @bar() {
    %0 = arith.constant 0 : i32
    %1 = arith.constant 1 : i32
    return
  }

  transform.sequence failures(propagate) {
  ^bb1(%arg1: !pdl.operation):
    // expected-error @below {{scope must not contain the transforms being applied}}
    transform.alternatives %arg1 : !pdl.operation {
    ^bb2(%arg2: !pdl.operation):
      %0 = transform.test_produce_self_handle_or_forward_operand
      transform.test_consume_operand_of_op_kind_or_fail %0, "transform.sequence"
    }, {
    ^bb2(%arg2: !pdl.operation):
      %0 = transform.test_produce_self_handle_or_forward_operand
      transform.test_consume_operand_of_op_kind_or_fail %0, "transform.test_produce_self_handle_or_forward_operand"
    }
  }
}

// -----

func.func @foo(%arg0: index, %arg1: index, %arg2: index) {
  // expected-note @below {{scope}}
  scf.for %i = %arg0 to %arg1 step %arg2 {
    %0 = arith.constant 0 : i32
  }
  return
}

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  pdl.pattern @match_const : benefit(1) {
    %0 = pdl.operands
    %1 = pdl.types
    %2 = pdl.operation "arith.constant"(%0 : !pdl.range<value>) -> (%1 : !pdl.range<type>)
    pdl.rewrite %2 with "transform.dialect"
  }


  sequence %arg0 : !pdl.operation failures(propagate) {
  ^bb1(%arg1: !pdl.operation):
    %0 = transform.pdl_match @match_const in %arg1 : (!pdl.operation) -> !pdl.operation
    %1 = transform.loop.get_parent_for %0 : (!pdl.operation) -> !pdl.operation
    // expected-error @below {{only isolated-from-above ops can be alternative scopes}}
    alternatives %1 : !pdl.operation {
    ^bb2(%arg2: !pdl.operation):
    }
  }
}
// -----

func.func @foo() {
  // expected-note @below {{when applied to this op}}
  "op" () : () -> ()
  return
}

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  pdl.pattern @some : benefit(1) {
    %0 = pdl.operands
    %1 = pdl.types
    %2 = pdl.operation "op"(%0 : !pdl.range<value>) -> (%1 : !pdl.range<type>)
    pdl.rewrite %2 with "transform.dialect"
  }

  transform.sequence %arg0 : !pdl.operation failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = pdl_match @some in %arg1 : (!pdl.operation) -> !pdl.operation
    // expected-error @below {{application of transform.test_wrong_number_of_results expected to produce 3 results (actually produced 1).}}
    // expected-note @below {{if you need variadic results, consider a generic `apply` instead of the specialized `applyToOne`.}}
    transform.test_wrong_number_of_results %0
  }
}

// -----

func.func @foo() {
  "op" () : () -> ()
  // expected-note @below {{when applied to this op}}
  "op" () : () -> ()
  return
}

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  pdl.pattern @some : benefit(1) {
    %0 = pdl.operands
    %1 = pdl.types
    %2 = pdl.operation "op"(%0 : !pdl.range<value>) -> (%1 : !pdl.range<type>)
    pdl.rewrite %2 with "transform.dialect"
  }

  transform.sequence %arg0 : !pdl.operation failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = pdl_match @some in %arg1 : (!pdl.operation) -> !pdl.operation
    // expected-error @below {{application of transform.test_wrong_number_of_multi_results expected to produce 1 results (actually produced 0)}}
    // expected-note @below {{if you need variadic results, consider a generic `apply` instead of the specialized `applyToOne`.}}
    transform.test_wrong_number_of_multi_results %0
  }
}

// -----

func.func @foo() {
  "op" () : () -> ()
  "op" () : () -> ()
  "op" () : () -> ()
  return
}

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  pdl.pattern @some : benefit(1) {
    %0 = pdl.operands
    %1 = pdl.types
    %2 = pdl.operation "op"(%0 : !pdl.range<value>) -> (%1 : !pdl.range<type>)
    pdl.rewrite %2 with "transform.dialect"
  }

  transform.sequence %arg0 : !pdl.operation failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = pdl_match @some in %arg1 : (!pdl.operation) -> !pdl.operation
    // Transform matches 3 ops and produces 2 results.
    %1:2 = transform.test_correct_number_of_multi_results %0
  }
}

// -----

func.func @foo() {
  "wrong_op_name" () : () -> ()
  return
}

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  pdl.pattern @some : benefit(1) {
    %0 = pdl.operands
    %1 = pdl.types
    %2 = pdl.operation "op"(%0 : !pdl.range<value>) -> (%1 : !pdl.range<type>)
    pdl.rewrite %2 with "transform.dialect"
  }

  transform.sequence %arg0 : !pdl.operation failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = pdl_match @some in %arg1 : (!pdl.operation) -> !pdl.operation
    // Transform fails to match any but still produces 2 results.
    %1:2 = transform.test_correct_number_of_multi_results %0
  }
}

// -----

// This should not fail.

func.func @foo() {
  "op" () : () -> ()
  return
}

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  pdl.pattern @some : benefit(1) {
    %0 = pdl.operands
    %1 = pdl.types
    %2 = pdl.operation "op"(%0 : !pdl.range<value>) -> (%1 : !pdl.range<type>)
    pdl.rewrite %2 with "transform.dialect"
  }

  transform.sequence %arg0 : !pdl.operation failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = pdl_match @some in %arg1 : (!pdl.operation) -> !pdl.operation
    transform.test_mixed_null_and_non_null_results %0
  }
}

// -----

// Expecting to match all operations by merging the handles that matched addi
// and subi separately.
func.func @foo(%arg0: index) {
  // expected-remark @below {{matched}}
  %0 = arith.addi %arg0, %arg0 : index
  // expected-remark @below {{matched}}
  %1 = arith.subi %arg0, %arg0 : index
  // expected-remark @below {{matched}}
  %2 = arith.addi %0, %1 : index
  return
}

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  pdl.pattern @addi : benefit(1) {
    %0 = pdl.operands
    %1 = pdl.types
    %2 = pdl.operation "arith.addi"(%0 : !pdl.range<value>) -> (%1 : !pdl.range<type>)
    pdl.rewrite %2 with "transform.dialect"
  }
  pdl.pattern @subi : benefit(1) {
    %0 = pdl.operands
    %1 = pdl.types
    %2 = pdl.operation "arith.subi"(%0 : !pdl.range<value>) -> (%1 : !pdl.range<type>)
    pdl.rewrite %2 with "transform.dialect"
  }

  transform.sequence %arg0 : !pdl.operation failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = pdl_match @addi in %arg1 : (!pdl.operation) -> !pdl.operation
    %1 = pdl_match @subi in %arg1 : (!pdl.operation) -> !pdl.operation
    %2 = merge_handles %0, %1 : !pdl.operation
    test_print_remark_at_operand %2, "matched" : !pdl.operation
  }
}

// -----

func.func @foo() {
  "op" () { target_me } : () -> ()
  // expected-note @below {{when applied to this op}}
  "op" () : () -> ()
  return
}

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  pdl.pattern @some : benefit(1) {
    %0 = pdl.operands
    %1 = pdl.types
    %2 = pdl.operation "op"(%0 : !pdl.range<value>) -> (%1 : !pdl.range<type>)
    pdl.rewrite %2 with "transform.dialect"
  }

  transform.sequence %arg0 : !pdl.operation failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = pdl_match @some in %arg1 : (!pdl.operation) -> !pdl.operation
    // expected-error @below {{failed to apply}}
    transform.test_mixed_sucess_and_silenceable %0
  }
}

// -----

func.func @foo() {
  "op" () : () -> ()
  return
}

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  pdl.pattern @some : benefit(1) {
    %0 = pdl.operands
    %1 = pdl.types
    %2 = pdl.operation "op"(%0 : !pdl.range<value>) -> (%1 : !pdl.range<type>)
    pdl.rewrite %2 with "transform.dialect"
  }

  transform.sequence %arg0 : !pdl.operation failures(suppress) {
  ^bb0(%arg1: !pdl.operation):
    %0 = pdl_match @some in %arg1 : (!pdl.operation) -> !pdl.operation
    // Not expecting error here because we are suppressing it.
    // expected-remark @below {{foo}}
    test_emit_remark_and_erase_operand %0, "foo" {fail_after_erase}
  }
}

// -----

func.func @foo() {
  "op" () : () -> ()
  return
}

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  pdl.pattern @some : benefit(1) {
    %0 = pdl.operands
    %1 = pdl.types
    %2 = pdl.operation "op"(%0 : !pdl.range<value>) -> (%1 : !pdl.range<type>)
    pdl.rewrite %2 with "transform.dialect"
  }

  transform.sequence %arg0 : !pdl.operation failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = pdl_match @some in %arg1 : (!pdl.operation) -> !pdl.operation
    // expected-error @below {{silenceable error}}
    // expected-remark @below {{foo}}
    test_emit_remark_and_erase_operand %0, "foo" {fail_after_erase}
  }
}


// -----

module {
  func.func private @foo()
  func.func private @bar()

  transform.with_pdl_patterns {
  ^bb0(%arg0: !pdl.operation):
    pdl.pattern @func : benefit(1) {
      %0 = pdl.operands
      %1 = pdl.types
      %2 = pdl.operation "func.func"(%0 : !pdl.range<value>) -> (%1 : !pdl.range<type>)
      pdl.rewrite %2 with "transform.dialect"
    }

    transform.sequence %arg0 : !pdl.operation failures(propagate) {
    ^bb0(%arg1: !pdl.operation):
      %0 = pdl_match @func in %arg1 : (!pdl.operation) -> !pdl.operation
      %1 = replicate num(%0) %arg1 : !pdl.operation, !pdl.operation
      // expected-remark @below {{2}}
      test_print_number_of_associated_payload_ir_ops %1
      %2 = replicate num(%0) %1 : !pdl.operation, !pdl.operation
      // expected-remark @below {{4}}
      test_print_number_of_associated_payload_ir_ops %2
    }
  }
}

// -----

func.func @bar() {
  // expected-remark @below {{transform applied}}
  %0 = arith.constant 0 : i32
  // expected-remark @below {{transform applied}}
  %1 = arith.constant 1 : i32
  return
}

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  pdl.pattern @const : benefit(1) {
    %r = pdl.types
    %0 = pdl.operation "arith.constant" -> (%r : !pdl.range<type>)
    pdl.rewrite %0 with "transform.dialect"
  }

  transform.sequence %arg0 : !pdl.operation failures(propagate) {
  ^bb1(%arg1: !pdl.operation):
    %f = pdl_match @const in %arg1 : (!pdl.operation) -> !pdl.operation
    transform.foreach %f : !pdl.operation {
    ^bb2(%arg2: !pdl.operation):
      // expected-remark @below {{1}}
      transform.test_print_number_of_associated_payload_ir_ops %arg2
      transform.test_print_remark_at_operand %arg2, "transform applied" : !pdl.operation
    }
  }
}

// -----

func.func @bar() {
  scf.execute_region {
    // expected-remark @below {{transform applied}}
    %0 = arith.constant 0 : i32
    scf.yield
  }

  scf.execute_region {
    // expected-remark @below {{transform applied}}
    %1 = arith.constant 1 : i32
    // expected-remark @below {{transform applied}}
    %2 = arith.constant 2 : i32
    scf.yield
  }

  return
}

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  pdl.pattern @const : benefit(1) {
    %r = pdl.types
    %0 = pdl.operation "arith.constant" -> (%r : !pdl.range<type>)
    pdl.rewrite %0 with "transform.dialect"
  }

  pdl.pattern @execute_region : benefit(1) {
    %r = pdl.types
    %0 = pdl.operation "scf.execute_region" -> (%r : !pdl.range<type>)
    pdl.rewrite %0 with "transform.dialect"
  }

  transform.sequence %arg0 : !pdl.operation failures(propagate) {
  ^bb1(%arg1: !pdl.operation):
    %f = pdl_match @execute_region in %arg1 : (!pdl.operation) -> !pdl.operation
    %results = transform.foreach %f : !pdl.operation -> !pdl.operation {
    ^bb2(%arg2: !pdl.operation):
      %g = transform.pdl_match @const in %arg2 : (!pdl.operation) -> !pdl.operation
      transform.yield %g : !pdl.operation
    }

    // expected-remark @below {{3}}
    transform.test_print_number_of_associated_payload_ir_ops %results
    transform.test_print_remark_at_operand %results, "transform applied" : !pdl.operation
  }
}

// -----

func.func @get_parent_for_op_no_loop(%arg0: index, %arg1: index) {
  // expected-remark @below {{found muli}}
  %0 = arith.muli %arg0, %arg1 : index
  arith.addi %0, %arg1 : index
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %addi = transform.structured.match ops{["arith.addi"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %muli = get_producer_of_operand %addi[0] : (!pdl.operation) -> !pdl.operation
  transform.test_print_remark_at_operand %muli, "found muli" : !pdl.operation
}

// -----

func.func @get_parent_for_op_no_loop(%arg0: index, %arg1: index) {
  // expected-note @below {{target op}}
  %0 = arith.muli %arg0, %arg1 : index
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %muli = transform.structured.match ops{["arith.muli"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  // expected-error @below {{could not find a producer for operand number: 0 of}}
  %bbarg = get_producer_of_operand %muli[0] : (!pdl.operation) -> !pdl.operation

}

// -----

func.func @get_consumer(%arg0: index, %arg1: index) {
  %0 = arith.muli %arg0, %arg1 : index
  // expected-remark @below {{found addi}}
  arith.addi %0, %arg1 : index
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %muli = transform.structured.match ops{["arith.muli"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  %addi = get_consumers_of_result %muli[0] : (!pdl.operation) -> !pdl.operation
  transform.test_print_remark_at_operand %addi, "found addi" : !pdl.operation
}

// -----

func.func @get_consumer_fail_1(%arg0: index, %arg1: index) {
  %0 = arith.muli %arg0, %arg1 : index
  %1 = arith.muli %arg0, %arg1 : index
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %muli = transform.structured.match ops{["arith.muli"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  // expected-error @below {{handle must be mapped to exactly one payload op}}
  %bbarg = get_consumers_of_result %muli[0] : (!pdl.operation) -> !pdl.operation

}

// -----

func.func @get_consumer_fail_2(%arg0: index, %arg1: index) {
  %0 = arith.muli %arg0, %arg1 : index
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %muli = transform.structured.match ops{["arith.muli"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  // expected-error @below {{result number overflow}}
  %bbarg = get_consumers_of_result %muli[1] : (!pdl.operation) -> !pdl.operation

}

// -----

func.func @split_handles(%a: index, %b: index, %c: index) {
  %0 = arith.muli %a, %b : index
  %1 = arith.muli %a, %c : index
  return
}

transform.sequence failures(propagate) {
^bb1(%fun: !pdl.operation):
  %muli = transform.structured.match ops{["arith.muli"]} in %fun : (!pdl.operation) -> !pdl.operation
  %h:2 = split_handles %muli in [2] : (!pdl.operation) -> (!pdl.operation, !pdl.operation)
  // expected-remark @below {{1}}
  transform.test_print_number_of_associated_payload_ir_ops %h#0
  %muli_2 = transform.structured.match ops{["arith.muli"]} in %fun : (!pdl.operation) -> !pdl.operation
  // expected-error @below {{expected to contain 3 operation handles but it only contains 2 handles}}
  %h_2:3 = split_handles %muli_2 in [3] : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation)
}

// -----

func.func @split_handles(%a: index, %b: index, %c: index) {
  %0 = arith.muli %a, %b : index
  %1 = arith.muli %a, %c : index
  return
}

transform.sequence failures(suppress) {
^bb1(%fun: !pdl.operation):
  %muli = transform.structured.match ops{["arith.muli"]} in %fun : (!pdl.operation) -> !pdl.operation
  %h:2 = split_handles %muli in [2] : (!pdl.operation) -> (!pdl.operation, !pdl.operation)
  // expected-remark @below {{1}}
  transform.test_print_number_of_associated_payload_ir_ops %h#0
  %muli_2 = transform.structured.match ops{["arith.muli"]} in %fun : (!pdl.operation) -> !pdl.operation
  // Silenceable failure and all handles are now empty.
  %h_2:3 = split_handles %muli_2 in [3] : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation)
  // expected-remark @below {{0}}
  transform.test_print_number_of_associated_payload_ir_ops %h_2#0
}

// -----

"test.some_op"() : () -> ()
"other_dialect.other_op"() : () -> ()

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  pdl.pattern @some : benefit(1) {
    %0 = pdl.operation "test.some_op"
    pdl.rewrite %0 with "transform.dialect"
  }

  sequence %arg0 : !pdl.operation failures(propagate) {
  ^bb1(%arg1: !pdl.operation):
    %0 = pdl_match @some in %arg1 : (!pdl.operation) -> !pdl.operation
    %2 = transform.cast %0 : !pdl.operation to !transform.test_dialect_op
    transform.cast %2 : !transform.test_dialect_op to !pdl.operation
  }
}

// -----

"test.some_op"() : () -> ()
"other_dialect.other_op"() : () -> ()

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  pdl.pattern @other : benefit(1) {
    %0 = pdl.operation "other_dialect.other_op"
    pdl.rewrite %0 with "transform.dialect"
  }

  sequence %arg0 : !pdl.operation failures(propagate) {
  ^bb1(%arg1: !pdl.operation):
    %0 = pdl_match @other in %arg1 : (!pdl.operation) -> !pdl.operation
    // expected-error @below {{expected the payload operation to belong to the 'test' dialect}}
    %2 = transform.cast %0 : !pdl.operation to !transform.test_dialect_op
    transform.cast %2 : !transform.test_dialect_op to !pdl.operation
  }
}

// -----

"test.some_op"() : () -> ()
"other_dialect.other_op"() : () -> ()

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  pdl.pattern @some : benefit(1) {
    %0 = pdl.operation "test.some_op"
    pdl.rewrite %0 with "transform.dialect"
  }

  sequence %arg0 : !pdl.operation failures(propagate) {
  ^bb1(%arg1: !pdl.operation):
    %0 = pdl_match @some in %arg1 : (!pdl.operation) -> !pdl.operation
    %2 = transform.cast %0 : !pdl.operation to !transform.op<"test.some_op">
    transform.cast %2 : !transform.op<"test.some_op"> to !pdl.operation
  }
}

// -----

"test.some_op"() : () -> ()
// expected-note @below {{payload operation}}
"other_dialect.other_op"() : () -> ()

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  pdl.pattern @other : benefit(1) {
    %0 = pdl.operation "other_dialect.other_op"
    pdl.rewrite %0 with "transform.dialect"
  }

  sequence %arg0 : !pdl.operation failures(propagate) {
  ^bb1(%arg1: !pdl.operation):
    %0 = pdl_match @other in %arg1 : (!pdl.operation) -> !pdl.operation
    // expected-error @below {{incompatible payload operation name}}
    %2 = transform.cast %0 : !pdl.operation to !transform.op<"test.some_op">
    transform.cast %2 : !transform.op<"test.some_op"> to !pdl.operation
  }
}

// -----

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  transform.sequence %arg0 : !pdl.operation failures(propagate) {
  ^bb0(%arg1: !pdl.operation):
    %0 = pdl_match @some in %arg1 : (!pdl.operation) -> !pdl.operation
    // here, the handles nested under are {%arg0, %arg1, %0}
    // expected-remark @below {{3 handles nested under}}
    transform.test_report_number_of_tracked_handles_nested_under %arg1
    // expected-remark @below {{erased}}
    transform.test_emit_remark_and_erase_operand %0, "erased"
    // here, the handles nested under are only {%arg0, %arg1}
    // expected-remark @below {{2 handles nested under}}
    transform.test_report_number_of_tracked_handles_nested_under %arg1
  }

  pdl.pattern @some : benefit(1) {
    %0 = pdl.operation "test.some_op"
    pdl.rewrite %0 with "transform.dialect"
  }
}

"test.some_op"() : () -> ()

// -----

func.func @split_handles(%a: index, %b: index, %c: index) {
  %0 = arith.muli %a, %b : index
  %1 = arith.muli %a, %c : index
  return
}

transform.sequence -> !pdl.operation failures(propagate) {
^bb1(%fun: !pdl.operation):
  %muli = transform.structured.match ops{["arith.muli"]} in %fun : (!pdl.operation) -> !pdl.operation
  // expected-error @below {{expected to contain 3 operation handles but it only contains 2 handles}}
  %h_2:3 = split_handles %muli in [3] : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation)
  /// Test that yield does not crash in the presence of silenceable error in
  /// propagate mode.
  yield %fun : !pdl.operation
}

// -----

transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  %0 = transform.test_produce_integer_param_with_type i32 : !transform.test_dialect_param
  // expected-remark @below {{0 : i32}}
  transform.test_print_param %0 : !transform.test_dialect_param
}

// -----

transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  // expected-error @below {{expected the type of the parameter attribute ('i32') to match the parameter type ('i64')}}
  transform.test_produce_integer_param_with_type i32 : !transform.param<i64>
}

// -----


transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  %0 = transform.test_add_to_param 40
  %1 = transform.test_add_to_param %0, 2
  // expected-remark @below {{42 : i32}}
  transform.test_print_param %1 : !transform.test_dialect_param
}

// -----

transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!pdl.operation) -> !pdl.operation
  %1 = transform.test_produce_param_with_number_of_test_ops %0 : !pdl.operation
  // expected-remark @below {{1 : i32, 3 : i32}}
  transform.test_print_param %1 : !transform.test_dialect_param
  %2 = transform.test_add_to_param %1, 100
  // expected-remark @below {{101 : i32, 103 : i32}}
  transform.test_print_param %2 : !transform.test_dialect_param
}

func.func private @one_test_op(%arg0: i32) {
  "test.op_a"(%arg0) { attr = 0 : i32} : (i32) -> i32
  return
}

func.func private @three_test_ops(%arg0: i32) {
  "test.op_a"(%arg0) { attr = 0 : i32} : (i32) -> i32
  "test.op_a"(%arg0) { attr = 0 : i32} : (i32) -> i32
  "test.op_a"(%arg0) { attr = 0 : i32} : (i32) -> i32
  return
}

// -----

// expected-note @below {{when applied to this op}}
module {
  transform.sequence failures(propagate) {
  ^bb0(%arg0: !transform.any_op):
    // expected-error @below {{expected to produce an Operation * for result #0}}
    transform.test_produce_transform_param_or_forward_operand %arg0
      { first_result_is_param }
      : (!transform.any_op) -> (!transform.any_op, !transform.param<i64>)
  }
}

// -----

// Should not fail.

module {
  transform.sequence failures(propagate) {
  ^bb0(%arg0: !transform.any_op):
    transform.test_produce_transform_param_or_forward_operand %arg0
      { first_result_is_null }
      : (!transform.any_op) -> (!transform.any_op, !transform.param<i64>)
  }
}

// -----

// expected-note @below {{when applied to this op}}
module {
  transform.sequence failures(propagate) {
  ^bb0(%arg0: !transform.any_op):
    // expected-error @below {{expected to produce an Attribute for result #1}}
    transform.test_produce_transform_param_or_forward_operand %arg0
      { second_result_is_handle }
      : (!transform.any_op) -> (!transform.any_op, !transform.param<i64>)
  }
}

// -----

// expected-note @below {{when applied to this op}}
module {
  transform.sequence failures(propagate) {
  ^bb0(%arg0: !transform.any_op):
    // expected-error @below {{expected to produce a Value for result #0}}
    transform.test_produce_transform_param_or_forward_operand %arg0
      { second_result_is_handle }
      : (!transform.any_op) -> (!transform.any_value, !transform.param<i64>)
  }
}

// -----

transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  // expected-error @below {{attempting to assign a null payload op to this transform value}}
  %0 = transform.test_produce_null_payload : !transform.any_op
}

// -----

transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  // expected-error @below {{attempting to assign a null parameter to this transform value}}
  %0 = transform.test_produce_null_param : !transform.param<i64>
}

// -----

transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  // expected-error @below {{attempting to assign a null payload value to this transform handle}}
  %0 = transform.test_produce_null_value : !transform.any_value
}

// -----

// expected-error @below {{could not find a nested top-level transform op}}
// expected-note @below {{use the 'transform-file-name' option to provide transform as external file}}
module {
}

// -----

// expected-note @below {{previous top-level transform op}}
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
}

// expected-error @below {{more than one top-level transform op}}
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
}

// -----

transform.sequence failures(propagate) {
// expected-remark @below {{value handle}}
// expected-note @below {{value handle points to a block argument #0 in block #0 in region #0}}
^bb1(%arg0: !transform.any_op):
  %0 = test_produce_value_handle_to_self_operand %arg0 : (!transform.any_op) -> !transform.any_value
  test_print_remark_at_operand_value %0, "value handle" : !transform.any_value
}

// -----

// expected-remark @below {{result handle}}
// expected-note @below {{value handle points to an op result #1}}
%0:2 = "test.get_two_results"() : () -> (i32, i32)
// expected-remark @below {{result handle}}
// expected-note @below {{value handle points to an op result #1}}
%1:3 = "test.get_three_results"() : () -> (i32, i32, f32)

transform.sequence failures(propagate) {
^bb1(%arg0: !transform.any_op):
  %2 = transform.structured.match ops{["test.get_two_results", "test.get_three_results"]} in %arg0 : (!transform.any_op) -> !transform.any_op
  %3 = test_produce_value_handle_to_result %2, 1 : (!transform.any_op) -> !transform.any_value
  test_print_remark_at_operand_value %3, "result handle" : !transform.any_value
}

// -----

"test.op_with_regions"() ({
^bb0:
  "test.regon_terminator"() : () -> ()
}, {
^bb1:
  "test.regon_terminator"() : () -> ()
// expected-remark @below {{block argument handle}}
// expected-note @below {{value handle points to a block argument #2 in block #1 in region #1}}
^bb2(%arg0: i32, %arg1: f64, %arg3: index):
  "test.match_anchor"() : () -> ()
  "test.regon_terminator"() : () -> ()
}) : () -> ()

transform.sequence failures(propagate) {
^bb1(%arg0: !transform.any_op):
  %2 = transform.structured.match ops{["test.match_anchor"]} in %arg0 : (!transform.any_op) -> !transform.any_op
  %3 = test_produce_value_handle_to_argument_of_parent_block %2, 2 : (!transform.any_op) -> !transform.any_value
  test_print_remark_at_operand_value %3, "block argument handle" : !transform.any_value
}

// -----

transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  // expected-note @below {{value defined here with type '!transform.test_dialect_param'}}
  %0 = test_produce_param_with_number_of_test_ops %arg0 : !transform.any_op
  // expected-error @below {{unexpectedly consumed a value that is not a handle as operand #0}}
  test_consume_operand %0 : !transform.test_dialect_param
}

// -----

func.func @get_result_of_op(%arg0: index, %arg1: index) -> index {
  // expected-remark @below {{addi result}}
  // expected-note @below {{value handle points to an op result #0}}
  %r = arith.addi %arg0, %arg1 : index
  return %r : index
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %addi = transform.structured.match ops{["arith.addi"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  %result = transform.get_result %addi[0] : (!transform.any_op) -> !transform.any_value
  transform.test_print_remark_at_operand_value %result, "addi result" : !transform.any_value
}

// -----

func.func @get_out_of_bounds_result_of_op(%arg0: index, %arg1: index) -> index {
  // expected-note @below {{target op}}
  %r = arith.addi %arg0, %arg1 : index
  return %r : index
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %addi = transform.structured.match ops{["arith.addi"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  // expected-error @below {{targeted op does not have enough results}}
  %result = transform.get_result %addi[1] : (!transform.any_op) -> !transform.any_value
  transform.test_print_remark_at_operand_value %result, "addi result" : !transform.any_value
}

// -----

func.func @get_result_of_op(%arg0: index, %arg1: index) -> index {
  // expected-remark @below {{matched}}
  %r = arith.addi %arg0, %arg1 : index
  return %r : index
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %addi = transform.structured.match ops{["arith.addi"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  %result = transform.get_result %addi[0] : (!transform.any_op) -> !transform.any_value
  %op = transform.get_defining_op %result : (!transform.any_value) -> !transform.any_op
  transform.test_print_remark_at_operand %op, "matched" : !transform.any_op
}

// -----

// expected-note @below {{target value}}
func.func @get_result_of_op_bbarg(%arg0: index, %arg1: index) -> index {
  %r = arith.addi %arg0, %arg1 : index
  return %r : index
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %addi = transform.structured.match ops{["arith.addi"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  %bbarg = test_produce_value_handle_to_argument_of_parent_block %addi, 0 : (!transform.any_op) -> !transform.any_value
  // expected-error @below {{cannot get defining op of block argument}}
  %op = transform.get_defining_op %bbarg : (!transform.any_value) -> !transform.any_op
  transform.test_print_remark_at_operand %op, "matched" : !transform.any_op
}

// -----

module @named_inclusion attributes { transform.with_named_sequence } {

  transform.named_sequence @foo(%arg0: !transform.any_op {transform.readonly}) -> () {
    // expected-remark @below {{applying transformation "a"}}
    transform.test_transform_op "a"
    transform.yield
  }

  transform.sequence failures(propagate) {
  ^bb0(%arg0: !transform.any_op):
    include @foo failures(propagate) (%arg0) : (!transform.any_op) -> ()
  }
}

// -----

module @named_inclusion_in_named attributes { transform.with_named_sequence } {

  transform.named_sequence @foo(%arg0: !transform.any_op {transform.readonly}) -> () {
    // expected-remark @below {{applying transformation "a"}}
    transform.test_transform_op "a"
    transform.yield
  }

  transform.named_sequence @bar(%arg0: !transform.any_op {transform.readonly}) -> () {
    // expected-remark @below {{applying transformation "b"}}
    transform.test_transform_op "b"
    transform.include @foo failures(propagate) (%arg0) : (!transform.any_op) -> ()
    transform.yield
  }

  transform.sequence failures(propagate) {
  ^bb0(%arg0: !transform.any_op):
    transform.include @bar failures(suppress) (%arg0) : (!transform.any_op) -> ()
  }
}

// -----

// expected-remark @below {{operation}}
module @named_operands attributes { transform.with_named_sequence } {

  transform.named_sequence @foo(%arg0: !transform.any_op {transform.readonly},
                                %arg1: !transform.any_value {transform.readonly}) -> () {
    transform.test_print_remark_at_operand %arg0, "operation" : !transform.any_op
    transform.test_print_remark_at_operand_value %arg1, "value" : !transform.any_value
    transform.yield
  }

  transform.sequence failures(propagate) {
  // expected-remark @below {{value}}
  // expected-note @below {{value handle points to a block argument #0 in block #0 in region #0}}
  ^bb0(%arg0: !transform.any_op):
    %0 = transform.test_produce_value_handle_to_self_operand %arg0 : (!transform.any_op) -> !transform.any_value
    include @foo failures(propagate) (%arg0, %0) : (!transform.any_op, !transform.any_value) -> ()
  }
}

// -----

// expected-remark @below {{operation}}
module @named_return attributes { transform.with_named_sequence } {

  // expected-remark @below {{value}}
  // expected-note @below {{value handle points to a block argument #0 in block #0 in region #0}}
  transform.named_sequence @foo(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_value) {
    %0 = transform.test_produce_value_handle_to_self_operand %arg0 : (!transform.any_op) -> !transform.any_value
    transform.yield %arg0, %0 : !transform.any_op, !transform.any_value
  }

  transform.sequence failures(propagate) {
  ^bb0(%arg0: !transform.any_op):
    %0:2 = include @foo failures(propagate) (%arg0) : (!transform.any_op) -> (!transform.any_op, !transform.any_value)
    transform.test_print_remark_at_operand %0#0, "operation" : !transform.any_op
    transform.test_print_remark_at_operand_value %0#1, "value" : !transform.any_value
  }
}
