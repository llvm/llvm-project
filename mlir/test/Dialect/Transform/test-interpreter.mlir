// RUN: mlir-opt %s --test-transform-dialect-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics

// expected-remark @below {{applying transformation}}
transform.test_transform_op

// -----

%0 = transform.test_produce_param_or_forward_operand 42 { foo = "bar" }
// expected-remark @below {{succeeded}}
transform.test_consume_operand_if_matches_param_or_fail %0[42]

// -----

%0 = transform.test_produce_param_or_forward_operand 42 { foo = "bar" }
// expected-error @below {{expected the operand to be associated with 21 got 42}}
transform.test_consume_operand_if_matches_param_or_fail %0[21]

// -----

// expected-error @below {{operation tracked by two handles}}
%0 = transform.test_produce_param_or_forward_operand 42
// expected-note @below {{handle}}
%1 = transform.test_produce_param_or_forward_operand from %0
// expected-note @below {{handle}}
%2 = transform.test_produce_param_or_forward_operand from %0
transform.test_consume_operand_if_matches_param_or_fail %1[42]
transform.test_consume_operand_if_matches_param_or_fail %2[42]

// -----

transform.sequence {
^bb0(%arg0: !pdl.operation):
  sequence %arg0 {
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

transform.sequence {
^bb0(%arg0: !pdl.operation):
  %0 = test_produce_param_or_forward_operand 42
  sequence %0 {
  ^bb0(%arg1: !pdl.operation):
    // expected-remark @below {{succeeded}}
    test_consume_operand_if_matches_param_or_fail %arg1[42]
  }
}

// -----

transform.sequence {
^bb0(%arg0: !pdl.operation):
  %0 = sequence %arg0 {
  ^bb0(%arg1: !pdl.operation):
    %1 = test_produce_param_or_forward_operand 42
    yield %1 : !pdl.operation
  } : !pdl.operation
  // expected-remark @below {{succeeded}}
  test_consume_operand_if_matches_param_or_fail %0[42]
}

// -----

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  sequence %arg0 {
  ^bb0(%arg1: !pdl.operation):
    %0 = pdl_match @some in %arg1
    test_print_remark_at_operand %0, "matched"
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

  transform.sequence %arg0 {
  ^bb1(%arg1: !pdl.operation):
    %f = pdl_match @const in %arg1
    // CHECK: %{{.+}} = get_closest_isolated_parent %{{.+}}
    %m = get_closest_isolated_parent %f
    test_print_remark_at_operand %m, "parent function"
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

  transform.sequence %arg0 {
  ^bb1(%arg1: !pdl.operation):
    // This is necessary to run the transformation on something other than the
    // top-level module, "alternatives" cannot be run on that.
    %0 = pdl_match @match_func in %arg1
    transform.alternatives %0 {
    ^bb2(%arg2: !pdl.operation):
      %1 = transform.test_produce_param_or_forward_operand 42
      // This operation fails, which triggers the next alternative without
      // reporting the error.
      transform.test_consume_operand_if_matches_param_or_fail %1[43]
    }, {
    ^bb2(%arg2: !pdl.operation):
      %1 = transform.test_produce_param_or_forward_operand 42
      // expected-remark @below {{succeeded}}
      transform.test_consume_operand_if_matches_param_or_fail %1[42]
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

  transform.sequence %arg0 {
  ^bb1(%arg1: !pdl.operation):
    %0 = pdl_match @match_call in %arg1
    %1 = get_closest_isolated_parent %0
    // expected-error @below {{all alternatives failed}}
    transform.alternatives %1 {
    ^bb2(%arg2: !pdl.operation):
      %2 = transform.pdl_match @match_call in %arg2
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

  transform.sequence %arg0 {
  ^bb1(%arg1: !pdl.operation):
    %0 = pdl_match @match_call in %arg1
    %1 = get_closest_isolated_parent %0
    transform.alternatives %1 {
    ^bb2(%arg2: !pdl.operation):
      %2 = transform.pdl_match @match_call in %arg2
      // expected-remark @below {{applying}}
      transform.test_emit_remark_and_erase_operand %2, "applying" {fail_after_erase}
    }, {
    ^bb2(%arg2: !pdl.operation):
      %2 = transform.pdl_match @match_call in %arg2
      transform.test_print_remark_at_operand %2, "still here"
      // This alternative succeeds.
    }, {
    ^bb2(%arg2: !pdl.operation):
      // This alternative is never run, so we must not have a remark here.
      %2 = transform.pdl_match @match_call in %arg2
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

  transform.sequence %arg0 {
  ^bb1(%arg1: !pdl.operation):
    %0 = pdl_match @match_call in %arg1
    %1 = get_closest_isolated_parent %0
    transform.alternatives %1 {
    ^bb2(%arg2: !pdl.operation):
      %2 = transform.pdl_match @match_call in %arg2
      // expected-remark @below {{applying}}
      transform.test_emit_remark_and_erase_operand %2, "applying" {fail_after_erase}
    }, {
    ^bb2(%arg2: !pdl.operation):
      %2 = transform.pdl_match @match_call in %arg2
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

  transform.sequence %arg0 {
  ^bb1(%arg1: !pdl.operation):
    %0 = pdl_match @match_call in %arg1
    %1 = get_closest_isolated_parent %0
    %2 = transform.alternatives %1 -> !pdl.operation {
    ^bb2(%arg2: !pdl.operation):
      %3 = transform.pdl_match @match_call in %arg2
      // expected-remark @below {{applying}}
      transform.test_emit_remark_and_erase_operand %3, "applying" {fail_after_erase}
      %4 = transform.test_produce_param_or_forward_operand 43
      transform.yield %4 : !pdl.operation
    }, {
    ^bb2(%arg2: !pdl.operation):
      %4 = transform.test_produce_param_or_forward_operand 42
      transform.yield %4 : !pdl.operation
    }
    // The first alternative failed, so the returned value is taken from the
    // second alternative.
    // expected-remark @below {{succeeded}}
    transform.test_consume_operand_if_matches_param_or_fail %2[42]
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

  transform.sequence {
  ^bb1(%arg1: !pdl.operation):
    // expected-error @below {{scope must not contain the transforms being applied}}
    transform.alternatives %arg1 {
    ^bb2(%arg2: !pdl.operation):
      %0 = transform.test_produce_param_or_forward_operand 42
      transform.test_consume_operand_if_matches_param_or_fail %0[43]
    }, {
    ^bb2(%arg2: !pdl.operation):
      %0 = transform.test_produce_param_or_forward_operand 42
      transform.test_consume_operand_if_matches_param_or_fail %0[42]
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


  sequence %arg0 {
  ^bb1(%arg1: !pdl.operation):
    %0 = transform.pdl_match @match_const in %arg1
    %1 = transform.loop.get_parent_for %0
    // expected-error @below {{only isolated-from-above ops can be alternative scopes}}
    alternatives %1 {
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

  transform.sequence %arg0 {
  ^bb0(%arg1: !pdl.operation):
    %0 = pdl_match @some in %arg1
    // expected-error @below {{applications of transform.test_wrong_number_of_results expected to produce 3 results (actually produced 1).}}
    // expected-note @below {{If you need variadic results, consider a generic `apply` instead of the specialized `applyToOne`.}}
    // expected-note @below {{Producing 3 null results is allowed if the use case warrants it.}}
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

  transform.sequence %arg0 {
  ^bb0(%arg1: !pdl.operation):
    %0 = pdl_match @some in %arg1
    // expected-error @below {{applications of transform.test_wrong_number_of_multi_results expected to produce 1 results (actually produced 0)}}
    // expected-note @below {{If you need variadic results, consider a generic `apply` instead of the specialized `applyToOne`.}}
    // expected-note @below {{Producing 1 null results is allowed if the use case warrants it.}}
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

  transform.sequence %arg0 {
  ^bb0(%arg1: !pdl.operation):
    %0 = pdl_match @some in %arg1
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

  transform.sequence %arg0 {
  ^bb0(%arg1: !pdl.operation):
    %0 = pdl_match @some in %arg1
    // Transform fails to match any but still produces 2 results.
    %1:2 = transform.test_correct_number_of_multi_results %0
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

  transform.sequence %arg0 {
  ^bb0(%arg1: !pdl.operation):
    %0 = pdl_match @some in %arg1
    // expected-error @below {{unexpected application of transform.test_mixed_null_and_non_null_results produces both null and non null results.}}
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

  transform.sequence %arg0 {
  ^bb0(%arg1: !pdl.operation):
    %0 = pdl_match @addi in %arg1
    %1 = pdl_match @subi in %arg1
    %2 = merge_handles %0, %1
    test_print_remark_at_operand %2, "matched"
  }
}

// -----

func.func @foo() {
  "op" () { target_me } : () -> ()
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

  transform.sequence %arg0 {
  ^bb0(%arg1: !pdl.operation):
    %0 = pdl_match @some in %arg1
    transform.test_mixed_sucess_and_silenceable %0
  }
}
