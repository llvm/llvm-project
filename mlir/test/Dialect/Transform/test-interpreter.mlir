// RUN: mlir-opt %s --transform-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics | FileCheck %s

// UNSUPPORTED: target=aarch64-pc-windows-msvc

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    // expected-remark @below {{applying transformation}}
    transform.test_transform_op
    transform.yield
  }
}

// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.test_produce_self_handle_or_forward_operand { foo = "bar" } : () -> !transform.any_op
    // expected-remark @below {{succeeded}}
    transform.test_consume_operand_of_op_kind_or_fail %0, "transform.test_produce_self_handle_or_forward_operand" : !transform.any_op
    transform.yield
  }
}

// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.test_produce_self_handle_or_forward_operand { foo = "bar" } : () -> !transform.any_op
    // expected-error @below {{expected the operand to be associated a payload op of kind transform.sequence got transform.test_produce_self_handle_or_forward_operand}}
    transform.test_consume_operand_of_op_kind_or_fail %0, "transform.sequence" : !transform.any_op
    transform.yield
  }
}

// -----

// It is okay to have multiple handles to the same payload op as long
// as only one of them is consumed. The expensive checks mode is necessary
// to detect double-consumption.
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.test_produce_self_handle_or_forward_operand { foo = "bar" } : () -> !transform.any_op
    %1 = transform.test_copy_payload %0 : (!transform.any_op) -> !transform.any_op
    // expected-remark @below {{succeeded}}
    transform.test_consume_operand_of_op_kind_or_fail %0, "transform.test_produce_self_handle_or_forward_operand" : !transform.any_op
    transform.yield
  }
}

// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    transform.sequence %arg0 : !transform.any_op failures(propagate) {
    ^bb0(%arg1: !transform.any_op):
      // expected-remark @below {{applying transformation "a"}}
      test_transform_op "a"
      // expected-remark @below {{applying transformation "b"}}
      test_transform_op "b"
      // expected-remark @below {{applying transformation "c"}}
      test_transform_op "c"
    }
    // expected-remark @below {{applying transformation "d"}}
    transform.test_transform_op "d"
    // expected-remark @below {{applying transformation "e"}}
    transform.test_transform_op "e"
    transform.yield
  }
}

// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.test_produce_self_handle_or_forward_operand : () -> !transform.any_op
    transform.sequence %0 : !transform.any_op failures(propagate) {
    ^bb0(%arg1: !transform.any_op):
      // expected-remark @below {{succeeded}}
      test_consume_operand_of_op_kind_or_fail %arg1, "transform.test_produce_self_handle_or_forward_operand" : !transform.any_op
    }
    transform.yield
  }
}

// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.sequence %arg0 : !transform.any_op -> !transform.any_op failures(propagate) {
    ^bb0(%arg1: !transform.any_op):
      %1 = test_produce_self_handle_or_forward_operand : () -> !transform.any_op
      yield %1 : !transform.any_op
    }
    // expected-remark @below {{succeeded}}
    transform.test_consume_operand_of_op_kind_or_fail %0, "transform.test_produce_self_handle_or_forward_operand" : !transform.any_op
    transform.yield
  }
}

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

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.with_pdl_patterns %root : !transform.any_op {
    ^bb0(%arg0: !transform.any_op):
      pdl.pattern @const : benefit(1) {
        %r = pdl.types
        %0 = pdl.operation "arith.constant" -> (%r : !pdl.range<type>)
        pdl.rewrite %0 with "transform.dialect"
      }

      transform.sequence %arg0 : !transform.any_op failures(propagate) {
      ^bb1(%arg1: !transform.any_op):
        %f = pdl_match @const in %arg1 : (!transform.any_op) -> !transform.any_op
        %m = get_parent_op %f {isolated_from_above} : (!transform.any_op) -> !transform.any_op
        transform.debug.emit_remark_at %m, "parent function" : !transform.any_op
      }
    }
    transform.yield
  }
}

// -----

func.func @test_get_nth_parent() {
  "test.foo"() ({
    // expected-remark @below{{2nd parent}}
    "test.foo"() ({
      "test.qux"() ({
        // expected-remark @below{{1st parent}}
        "test.foo"() ({
          "test.bar"() : () -> ()
        }) : () -> ()
      }) : () -> ()
    }) : () -> ()
  }) : () -> ()
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %f = transform.structured.match ops{["test.bar"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %parent = transform.get_parent_op %f {nth_parent = 1, op_name = "test.foo"} : (!transform.any_op) -> !transform.any_op
    transform.debug.emit_remark_at %parent, "1st parent" : !transform.any_op
    %parent2 = transform.get_parent_op %f {nth_parent = 2, op_name = "test.foo"} : (!transform.any_op) -> !transform.any_op
    transform.debug.emit_remark_at %parent2, "2nd parent" : !transform.any_op
    transform.yield
  }
}

// -----

func.func @foo() {
  %0 = arith.constant 0 : i32
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.with_pdl_patterns %root : !transform.any_op {
    ^bb0(%arg0: !transform.any_op):
      pdl.pattern @match_func : benefit(1) {
        %0 = pdl.operands
        %1 = pdl.types
        %2 = pdl.operation "func.func"(%0 : !pdl.range<value>) -> (%1 : !pdl.range<type>)
        pdl.rewrite %2 with "transform.dialect"
      }

      transform.sequence %arg0 : !transform.any_op failures(propagate) {
      ^bb1(%arg1: !transform.any_op):
        // This is necessary to run the transformation on something other than the
        // top-level module, "alternatives" cannot be run on that.
        %0 = pdl_match @match_func in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.alternatives %0 : !transform.any_op {
        ^bb2(%arg2: !transform.any_op):
          %1 = transform.test_produce_self_handle_or_forward_operand : () -> !transform.any_op
          // This operation fails, which triggers the next alternative without
          // reporting the error.
          transform.test_consume_operand_of_op_kind_or_fail %1, "transform.sequence" : !transform.any_op
        }, {
        ^bb2(%arg2: !transform.any_op):
          %1 = transform.test_produce_self_handle_or_forward_operand : () -> !transform.any_op
          // expected-remark @below {{succeeded}}
          transform.test_consume_operand_of_op_kind_or_fail %1, "transform.test_produce_self_handle_or_forward_operand" : !transform.any_op
        }
      }
    }
    transform.yield
  }
}

// -----

func.func private @bar()

func.func @foo() {
  call @bar() : () -> ()
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.with_pdl_patterns %root : !transform.any_op {
    ^bb0(%arg0: !transform.any_op):
      pdl.pattern @match_call : benefit(1) {
        %0 = pdl.operands
        %1 = pdl.types
        %2 = pdl.operation "func.call"(%0 : !pdl.range<value>) -> (%1 : !pdl.range<type>)
        pdl.rewrite %2 with "transform.dialect"
      }

      transform.sequence %arg0 : !transform.any_op failures(propagate) {
      ^bb1(%arg1: !transform.any_op):
        %0 = pdl_match @match_call in %arg1 : (!transform.any_op) -> !transform.any_op
        %1 = get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
        // expected-error @below {{all alternatives failed}}
        transform.alternatives %1 : !transform.any_op {
        ^bb2(%arg2: !transform.any_op):
          %2 = transform.pdl_match @match_call in %arg2 : (!transform.any_op) -> !transform.any_op
          // expected-remark @below {{applying}}
          transform.test_emit_remark_and_erase_operand %2, "applying" {fail_after_erase} : !transform.any_op
        }
      }
    }
    transform.yield
  }
}

// -----

func.func private @bar()

func.func @foo() {
  // expected-remark @below {{still here}}
  call @bar() : () -> ()
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.with_pdl_patterns %root : !transform.any_op {
    ^bb0(%arg0: !transform.any_op):
      pdl.pattern @match_call : benefit(1) {
        %0 = pdl.operands
        %1 = pdl.types
        %2 = pdl.operation "func.call"(%0 : !pdl.range<value>) -> (%1 : !pdl.range<type>)
        pdl.rewrite %2 with "transform.dialect"
      }

      transform.sequence %arg0 : !transform.any_op failures(propagate) {
      ^bb1(%arg1: !transform.any_op):
        %0 = pdl_match @match_call in %arg1 : (!transform.any_op) -> !transform.any_op
        %1 = get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
        transform.alternatives %1 : !transform.any_op {
        ^bb2(%arg2: !transform.any_op):
          %2 = transform.pdl_match @match_call in %arg2 : (!transform.any_op) -> !transform.any_op
          // expected-remark @below {{applying}}
          transform.test_emit_remark_and_erase_operand %2, "applying" {fail_after_erase} : !transform.any_op
        }, {
        ^bb2(%arg2: !transform.any_op):
          %2 = transform.pdl_match @match_call in %arg2 : (!transform.any_op) -> !transform.any_op
          transform.debug.emit_remark_at %2, "still here" : !transform.any_op
          // This alternative succeeds.
        }, {
        ^bb2(%arg2: !transform.any_op):
          // This alternative is never run, so we must not have a remark here.
          %2 = transform.pdl_match @match_call in %arg2 : (!transform.any_op) -> !transform.any_op
          transform.test_emit_remark_and_erase_operand %2, "should not happen" {fail_after_erase} : !transform.any_op
        }
      }
    }
    transform.yield
  }
}

// -----

func.func private @bar()

func.func @erase_call() {
  call @bar() : () -> ()
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.with_pdl_patterns %root : !transform.any_op {
    ^bb0(%arg0: !transform.any_op):
      pdl.pattern @match_call : benefit(1) {
        %0 = pdl.operands
        %1 = pdl.types
        %2 = pdl.operation "func.call"(%0 : !pdl.range<value>) -> (%1 : !pdl.range<type>)
        pdl.rewrite %2 with "transform.dialect"
      }

      transform.sequence %arg0 : !transform.any_op failures(propagate) {
      ^bb1(%arg1: !transform.any_op):
        %0 = pdl_match @match_call in %arg1 : (!transform.any_op) -> !transform.any_op
        %1 = get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
        transform.alternatives %1 : !transform.any_op {
        ^bb2(%arg2: !transform.any_op):
          %2 = transform.pdl_match @match_call in %arg2 : (!transform.any_op) -> !transform.any_op
          // expected-remark @below {{applying}}
          transform.test_emit_remark_and_erase_operand %2, "applying" {fail_after_erase} : !transform.any_op
        }, {
        ^bb2(%arg2: !transform.any_op):
          %2 = transform.pdl_match @match_call in %arg2 : (!transform.any_op) -> !transform.any_op
          // expected-remark @below {{applying second time}}
          transform.test_emit_remark_and_erase_operand %2, "applying second time" : !transform.any_op
        }
      }
    }
    transform.yield
  }
}

// -----

func.func private @bar()

func.func @foo() {
  call @bar() : () -> ()
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.with_pdl_patterns %root : !transform.any_op {
    ^bb0(%arg0: !transform.any_op):
      pdl.pattern @match_call : benefit(1) {
        %0 = pdl.operands
        %1 = pdl.types
        %2 = pdl.operation "func.call"(%0 : !pdl.range<value>) -> (%1 : !pdl.range<type>)
        pdl.rewrite %2 with "transform.dialect"
      }

      transform.sequence %arg0 : !transform.any_op failures(propagate) {
      ^bb1(%arg1: !transform.any_op):
        %0 = pdl_match @match_call in %arg1 : (!transform.any_op) -> !transform.any_op
        %1 = get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
        %2 = transform.alternatives %1 : !transform.any_op -> !transform.any_op {
        ^bb2(%arg2: !transform.any_op):
          %3 = transform.pdl_match @match_call in %arg2 : (!transform.any_op) -> !transform.any_op
          // expected-remark @below {{applying}}
          transform.test_emit_remark_and_erase_operand %3, "applying" {fail_after_erase} : !transform.any_op
          %4 = transform.test_produce_self_handle_or_forward_operand %3 : (!transform.any_op) -> !transform.any_op
          transform.yield %4 : !transform.any_op
        }, {
        ^bb2(%arg2: !transform.any_op):
          %4 = transform.test_produce_self_handle_or_forward_operand : () -> !transform.any_op
          transform.yield %4 : !transform.any_op
        }
        // The first alternative failed, so the returned value is taken from the
        // second alternative, associated test_produce_self_handle_or_forward_operand rather
        // than pdl_match.
        // expected-remark @below {{succeeded}}
        transform.test_consume_operand_of_op_kind_or_fail %2, "transform.test_produce_self_handle_or_forward_operand" : !transform.any_op
      }
    }
    transform.yield
  }
}

// -----

// expected-note @below {{scope}}
module attributes {transform.with_named_sequence} {
  func.func @foo() {
    %0 = arith.constant 0 : i32
    return
  }

  func.func @bar() {
    %0 = arith.constant 0 : i32
    %1 = arith.constant 1 : i32
    return
  }

  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    // expected-error @below {{scope must not contain the transforms being applied}}
    transform.alternatives %arg1 : !transform.any_op {
    ^bb2(%arg2: !transform.any_op):
      %0 = transform.test_produce_self_handle_or_forward_operand : () -> !transform.any_op
      transform.test_consume_operand_of_op_kind_or_fail %0, "transform.sequence" : !transform.any_op
    }, {
    ^bb2(%arg2: !transform.any_op):
      %0 = transform.test_produce_self_handle_or_forward_operand : () -> !transform.any_op
      transform.test_consume_operand_of_op_kind_or_fail %0, "transform.test_produce_self_handle_or_forward_operand" : !transform.any_op
    }
    transform.yield
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

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.with_pdl_patterns %root : !transform.any_op {
    ^bb0(%arg0: !transform.any_op):
      pdl.pattern @match_const : benefit(1) {
        %0 = pdl.operands
        %1 = pdl.types
        %2 = pdl.operation "arith.constant"(%0 : !pdl.range<value>) -> (%1 : !pdl.range<type>)
        pdl.rewrite %2 with "transform.dialect"
      }


      sequence %arg0 : !transform.any_op failures(propagate) {
      ^bb1(%arg1: !transform.any_op):
        %0 = transform.pdl_match @match_const in %arg1 : (!transform.any_op) -> !transform.any_op
        %1 = transform.get_parent_op %0 {op_name = "scf.for"} : (!transform.any_op) -> !transform.any_op
        // expected-error @below {{only isolated-from-above ops can be alternative scopes}}
        alternatives %1 : !transform.any_op {
        ^bb2(%arg2: !transform.any_op):
        }
      }
    }
    transform.yield
  }
}
// -----

func.func @foo() {
  // expected-note @below {{when applied to this op}}
  "op" () : () -> ()
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.with_pdl_patterns %root : !transform.any_op {
    ^bb0(%arg0: !transform.any_op):
      pdl.pattern @some : benefit(1) {
        %0 = pdl.operands
        %1 = pdl.types
        %2 = pdl.operation "op"(%0 : !pdl.range<value>) -> (%1 : !pdl.range<type>)
        pdl.rewrite %2 with "transform.dialect"
      }

      transform.sequence %arg0 : !transform.any_op failures(propagate) {
      ^bb0(%arg1: !transform.any_op):
        %0 = pdl_match @some in %arg1 : (!transform.any_op) -> !transform.any_op
        // expected-error @below {{application of transform.test_wrong_number_of_results expected to produce 3 results (actually produced 1).}}
        // expected-note @below {{if you need variadic results, consider a generic `apply` instead of the specialized `applyToOne`.}}
        transform.test_wrong_number_of_results %0 : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
      }
    }
    transform.yield
  }
}

// -----

func.func @foo() {
  "op" () : () -> ()
  // expected-note @below {{when applied to this op}}
  "op" () : () -> ()
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.with_pdl_patterns %root : !transform.any_op {
    ^bb0(%arg0: !transform.any_op):
      pdl.pattern @some : benefit(1) {
        %0 = pdl.operands
        %1 = pdl.types
        %2 = pdl.operation "op"(%0 : !pdl.range<value>) -> (%1 : !pdl.range<type>)
        pdl.rewrite %2 with "transform.dialect"
      }

      transform.sequence %arg0 : !transform.any_op failures(propagate) {
      ^bb0(%arg1: !transform.any_op):
        %0 = pdl_match @some in %arg1 : (!transform.any_op) -> !transform.any_op
        // expected-error @below {{application of transform.test_wrong_number_of_multi_results expected to produce 1 results (actually produced 0)}}
        // expected-note @below {{if you need variadic results, consider a generic `apply` instead of the specialized `applyToOne`.}}
        transform.test_wrong_number_of_multi_results %0 : (!transform.any_op) -> (!transform.any_op)
      }
    }
    transform.yield
  }
}

// -----

func.func @foo() {
  "op" () : () -> ()
  "op" () : () -> ()
  "op" () : () -> ()
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.with_pdl_patterns %root : !transform.any_op {
    ^bb0(%arg0: !transform.any_op):
      pdl.pattern @some : benefit(1) {
        %0 = pdl.operands
        %1 = pdl.types
        %2 = pdl.operation "op"(%0 : !pdl.range<value>) -> (%1 : !pdl.range<type>)
        pdl.rewrite %2 with "transform.dialect"
      }

      transform.sequence %arg0 : !transform.any_op failures(propagate) {
      ^bb0(%arg1: !transform.any_op):
        %0 = pdl_match @some in %arg1 : (!transform.any_op) -> !transform.any_op
        // Transform matches 3 ops and produces 2 results.
        %1:2 = transform.test_correct_number_of_multi_results %0 : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
      }
    }
    transform.yield
  }
}

// -----

func.func @foo() {
  "wrong_op_name" () : () -> ()
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.with_pdl_patterns %root : !transform.any_op {
    ^bb0(%arg0: !transform.any_op):
      pdl.pattern @some : benefit(1) {
        %0 = pdl.operands
        %1 = pdl.types
        %2 = pdl.operation "op"(%0 : !pdl.range<value>) -> (%1 : !pdl.range<type>)
        pdl.rewrite %2 with "transform.dialect"
      }

      transform.sequence %arg0 : !transform.any_op failures(propagate) {
      ^bb0(%arg1: !transform.any_op):
        %0 = pdl_match @some in %arg1 : (!transform.any_op) -> !transform.any_op
        // Transform fails to match any but still produces 2 results.
        %1:2 = transform.test_correct_number_of_multi_results %0 : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
      }
    }
    transform.yield
  }
}

// -----

// This should not fail.

func.func @foo() {
  "op" () : () -> ()
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.with_pdl_patterns %root : !transform.any_op {
    ^bb0(%arg0: !transform.any_op):
      pdl.pattern @some : benefit(1) {
        %0 = pdl.operands
        %1 = pdl.types
        %2 = pdl.operation "op"(%0 : !pdl.range<value>) -> (%1 : !pdl.range<type>)
        pdl.rewrite %2 with "transform.dialect"
      }

      transform.sequence %arg0 : !transform.any_op failures(propagate) {
      ^bb0(%arg1: !transform.any_op):
        %0 = pdl_match @some in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.test_mixed_null_and_non_null_results %0 : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
      }
    }
    transform.yield
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

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.with_pdl_patterns %root : !transform.any_op {
    ^bb0(%arg0: !transform.any_op):
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

      transform.sequence %arg0 : !transform.any_op failures(propagate) {
      ^bb0(%arg1: !transform.any_op):
        %0 = pdl_match @addi in %arg1 : (!transform.any_op) -> !transform.any_op
        %1 = pdl_match @subi in %arg1 : (!transform.any_op) -> !transform.any_op
        %2 = merge_handles %0, %1 : !transform.any_op
        transform.debug.emit_remark_at %2, "matched" : !transform.any_op
      }
    }
    transform.yield
  }
}

// -----

func.func @foo(%arg0: index) {
  %0 = arith.addi %arg0, %arg0 : index
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.with_pdl_patterns %root : !transform.any_op {
    ^bb0(%arg0: !transform.any_op):
      pdl.pattern @addi : benefit(1) {
        %0 = pdl.operands
        %1 = pdl.types
        %2 = pdl.operation "arith.addi"(%0 : !pdl.range<value>) -> (%1 : !pdl.range<type>)
        pdl.rewrite %2 with "transform.dialect"
      }

      transform.sequence %arg0 : !transform.any_op failures(propagate) {
      ^bb0(%arg1: !transform.any_op):
        %0 = pdl_match @addi in %arg1 : (!transform.any_op) -> !transform.any_op
        %1 = pdl_match @addi in %arg1 : (!transform.any_op) -> !transform.any_op
        %2 = merge_handles deduplicate %0, %1 : !transform.any_op
        %3 = num_associations %2 : (!transform.any_op) -> !transform.param<i64>
        // expected-remark @below {{1}}
        transform.debug.emit_param_as_remark  %3 : !transform.param<i64>
      }
    }
    transform.yield
  }
}

// -----

func.func @foo() {
  "op" () { target_me } : () -> ()
  // expected-note @below {{when applied to this op}}
  "op" () : () -> ()
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.with_pdl_patterns %root : !transform.any_op {
    ^bb0(%arg0: !transform.any_op):
      pdl.pattern @some : benefit(1) {
        %0 = pdl.operands
        %1 = pdl.types
        %2 = pdl.operation "op"(%0 : !pdl.range<value>) -> (%1 : !pdl.range<type>)
        pdl.rewrite %2 with "transform.dialect"
      }

      transform.sequence %arg0 : !transform.any_op failures(propagate) {
      ^bb0(%arg1: !transform.any_op):
        %0 = pdl_match @some in %arg1 : (!transform.any_op) -> !transform.any_op
        // expected-error @below {{failed to apply}}
        transform.test_mixed_success_and_silenceable %0 : !transform.any_op
      }
    }
    transform.yield
  }
}

// -----

func.func @foo() {
  "op" () : () -> ()
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.with_pdl_patterns %root : !transform.any_op {
    ^bb0(%arg0: !transform.any_op):
      pdl.pattern @some : benefit(1) {
        %0 = pdl.operands
        %1 = pdl.types
        %2 = pdl.operation "op"(%0 : !pdl.range<value>) -> (%1 : !pdl.range<type>)
        pdl.rewrite %2 with "transform.dialect"
      }

      transform.sequence %arg0 : !transform.any_op failures(suppress) {
      ^bb0(%arg1: !transform.any_op):
        %0 = pdl_match @some in %arg1 : (!transform.any_op) -> !transform.any_op
        // Not expecting error here because we are suppressing it.
        // expected-remark @below {{foo}}
        test_emit_remark_and_erase_operand %0, "foo" {fail_after_erase} : !transform.any_op
      }
    }
    transform.yield
  }
}

// -----

func.func @foo() {
  "op" () : () -> ()
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.with_pdl_patterns %root : !transform.any_op {
    ^bb0(%arg0: !transform.any_op):
      pdl.pattern @some : benefit(1) {
        %0 = pdl.operands
        %1 = pdl.types
        %2 = pdl.operation "op"(%0 : !pdl.range<value>) -> (%1 : !pdl.range<type>)
        pdl.rewrite %2 with "transform.dialect"
      }

      transform.sequence %arg0 : !transform.any_op failures(propagate) {
      ^bb0(%arg1: !transform.any_op):
        %0 = pdl_match @some in %arg1 : (!transform.any_op) -> !transform.any_op
        // expected-error @below {{silenceable error}}
        // expected-remark @below {{foo}}
        test_emit_remark_and_erase_operand %0, "foo" {fail_after_erase} : !transform.any_op
      }
    }
    transform.yield
  }
}


// -----

module attributes {transform.with_named_sequence} {
  func.func private @foo()
  func.func private @bar()

  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.with_pdl_patterns %root : !transform.any_op {

    ^bb0(%arg0: !transform.any_op):
      pdl.pattern @func : benefit(1) {
        %0 = pdl.operands
        %1 = pdl.types
        %2 = pdl.operation "func.func"(%0 : !pdl.range<value>) -> (%1 : !pdl.range<type>)
        pdl.rewrite %2 with "transform.dialect"
      }

      transform.sequence %arg0 : !transform.any_op failures(propagate) {
      ^bb0(%arg1: !transform.any_op):
        %0 = pdl_match @func in %arg1 : (!transform.any_op) -> !transform.any_op
        %1 = replicate num(%0) %arg1 : !transform.any_op, !transform.any_op
        %p = num_associations %1 : (!transform.any_op) -> !transform.param<i64>
        // expected-remark @below {{2}}
        transform.debug.emit_param_as_remark  %p : !transform.param<i64>
        %2 = replicate num(%0) %1 : !transform.any_op, !transform.any_op
        %p2 = num_associations %2 : (!transform.any_op) -> !transform.param<i64>
        // expected-remark @below {{4}}
        transform.debug.emit_param_as_remark  %p2 : !transform.param<i64>
      }
    }
    transform.yield
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

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.with_pdl_patterns %root : !transform.any_op {
    ^bb0(%arg0: !transform.any_op):
      pdl.pattern @const : benefit(1) {
        %r = pdl.types
        %0 = pdl.operation "arith.constant" -> (%r : !pdl.range<type>)
        pdl.rewrite %0 with "transform.dialect"
      }

      transform.sequence %arg0 : !transform.any_op failures(propagate) {
      ^bb1(%arg1: !transform.any_op):
        %f = pdl_match @const in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.foreach %f : !transform.any_op {
        ^bb2(%arg2: !transform.any_op):
          %p = transform.num_associations %arg2 : (!transform.any_op) -> !transform.param<i64>
          // expected-remark @below {{1}}
          transform.debug.emit_param_as_remark  %p : !transform.param<i64>
          transform.debug.emit_remark_at %arg2, "transform applied" : !transform.any_op
        }
      }
    }
    transform.yield
  }
}

// -----

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %results, %types = transform.foreach %0 : !transform.any_op -> !transform.any_value, !transform.any_param {
    ^bb0(%op0 : !transform.any_op):
      %result = transform.get_result %op0[0] : (!transform.any_op) -> !transform.any_value
      %type = transform.get_type elemental %result  : (!transform.any_value) -> !transform.any_param
      transform.yield %result, %type : !transform.any_value, !transform.any_param
    }
    transform.debug.emit_remark_at %results, "result selected" : !transform.any_value
    transform.debug.emit_param_as_remark %types, "elemental types" at %0 : !transform.any_param, !transform.any_op

    transform.yield
  }
}

func.func @payload(%lhs: tensor<10x20xf16>,
                   %rhs: tensor<20x15xf32>) -> (tensor<10x15xf64>, tensor<10x15xf32>) {
  %cst64 = arith.constant 0.0 : f64
  %empty64 = tensor.empty() : tensor<10x15xf64>
  %fill64 = linalg.fill ins(%cst64 : f64) outs(%empty64 : tensor<10x15xf64>) -> tensor<10x15xf64>
  // expected-remark @below {{result selected}}
  // expected-note @below {{value handle points to an op result #0}}
  // expected-remark @below {{elemental types f64, f32}}
  %result64 = linalg.matmul ins(%lhs, %rhs: tensor<10x20xf16>, tensor<20x15xf32>)
                         outs(%fill64: tensor<10x15xf64>) -> tensor<10x15xf64>

  %cst32 = arith.constant 0.0 : f32
  %empty32 = tensor.empty() : tensor<10x15xf32>
  %fill32 = linalg.fill ins(%cst32 : f32) outs(%empty32 : tensor<10x15xf32>) -> tensor<10x15xf32>
  // expected-remark @below {{result selected}}
  // expected-note @below {{value handle points to an op result #0}}
  // expected-remark @below {{elemental types f64, f32}}
  %result32 = linalg.matmul ins(%lhs, %rhs: tensor<10x20xf16>, tensor<20x15xf32>)
                           outs(%fill32: tensor<10x15xf32>) -> tensor<10x15xf32>

  return %result64, %result32 : tensor<10x15xf64>, tensor<10x15xf32>

}

// -----

func.func @two_const_ops() {
  %0 = arith.constant 0 : index
  %1 = arith.constant 1 : index
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %two_ops = transform.structured.match ops{["arith.constant"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %one_param = transform.param.constant 1 : i32 -> !transform.test_dialect_param
    // expected-error @below {{prior targets' payload size (2) differs from payload size (1) of target}}
    transform.foreach %two_ops, %one_param : !transform.any_op, !transform.test_dialect_param {
    ^bb2(%op: !transform.any_op, %param: !transform.test_dialect_param):
    }
    transform.yield
  }
}

// -----

func.func @one_const_op() {
  %0 = arith.constant 0 : index
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %one_op = transform.structured.match ops{["arith.constant"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %one_val = transform.test_produce_value_handle_to_self_operand %one_op : (!transform.any_op) -> !transform.any_value
    %param_one = transform.param.constant 1 : i32 -> !transform.test_dialect_param
    %param_two = transform.param.constant 2 : i32 -> !transform.test_dialect_param
    %two_params = transform.merge_handles %param_one, %param_two : !transform.test_dialect_param

    // expected-error @below {{prior targets' payload size (1) differs from payload size (2) of target}}
    transform.foreach %one_val, %one_op, %two_params : !transform.any_value, !transform.any_op, !transform.test_dialect_param {
    ^bb2(%val: !transform.any_value, %op: !transform.any_op, %param: !transform.test_dialect_param):
    }
    transform.yield
  }
}

// -----

// CHECK-LABEL: func @consume_in_foreach()
//  CHECK-NEXT:   return
func.func @consume_in_foreach() {
  %0 = arith.constant 0 : index
  %1 = arith.constant 1 : index
  %2 = arith.constant 2 : index
  %3 = arith.constant 3 : index
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %f = transform.structured.match ops{["arith.constant"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.foreach %f : !transform.any_op {
    ^bb2(%arg2: !transform.any_op):
      // expected-remark @below {{erasing}}
      transform.test_emit_remark_and_erase_operand %arg2, "erasing" : !transform.any_op
    }
    transform.yield
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

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.with_pdl_patterns %root : !transform.any_op {
    ^bb0(%arg0: !transform.any_op):
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

      transform.sequence %arg0 : !transform.any_op failures(propagate) {
      ^bb1(%arg1: !transform.any_op):
        %f = pdl_match @execute_region in %arg1 : (!transform.any_op) -> !transform.any_op
        %results = transform.foreach %f : !transform.any_op -> !transform.any_op {
        ^bb2(%arg2: !transform.any_op):
          %g = transform.pdl_match @const in %arg2 : (!transform.any_op) -> !transform.any_op
          transform.yield %g : !transform.any_op
        }

        %p = transform.num_associations %results : (!transform.any_op) -> !transform.param<i64>
        // expected-remark @below {{3}}
        transform.debug.emit_param_as_remark  %p : !transform.param<i64>
        transform.debug.emit_remark_at %results, "transform applied" : !transform.any_op
      }
    }
    transform.yield
  }
}

// -----

func.func @get_parent_for_op_no_loop(%arg0: index, %arg1: index) {
  // expected-remark @below {{found muli}}
  %0 = arith.muli %arg0, %arg1 : index
  arith.addi %0, %arg1 : index
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %addi = transform.structured.match ops{["arith.addi"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %muli = transform.get_producer_of_operand %addi[0] : (!transform.any_op) -> !transform.any_op
    transform.debug.emit_remark_at %muli, "found muli" : !transform.any_op
    transform.yield
  }
}

// -----

func.func @get_parent_for_op_no_loop(%arg0: index, %arg1: index) {
  // expected-note @below {{target op}}
  %0 = arith.muli %arg0, %arg1 : index
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %muli = transform.structured.match ops{["arith.muli"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{could not find a producer for operand number: 0 of}}
    %bbarg = transform.get_producer_of_operand %muli[0] : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

func.func @get_consumer(%arg0: index, %arg1: index) {
  %0 = arith.muli %arg0, %arg1 : index
  // expected-remark @below {{found addi}}
  arith.addi %0, %arg1 : index
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %muli = transform.structured.match ops{["arith.muli"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %addi = transform.get_consumers_of_result %muli[0] : (!transform.any_op) -> !transform.any_op
    transform.debug.emit_remark_at %addi, "found addi" : !transform.any_op
    transform.yield
  }
}

// -----

func.func @get_consumer_fail_1(%arg0: index, %arg1: index) {
  %0 = arith.muli %arg0, %arg1 : index
  %1 = arith.muli %arg0, %arg1 : index
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %muli = transform.structured.match ops{["arith.muli"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{handle must be mapped to exactly one payload op}}
    %bbarg = transform.get_consumers_of_result %muli[0] : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

func.func @get_consumer_fail_2(%arg0: index, %arg1: index) {
  %0 = arith.muli %arg0, %arg1 : index
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %muli = transform.structured.match ops{["arith.muli"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{result number overflow}}
    %bbarg = transform.get_consumers_of_result %muli[1] : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

func.func @split_handle(%a: index, %b: index, %c: index) {
  %0 = arith.muli %a, %b : index
  %1 = arith.muli %a, %c : index
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%fun: !transform.any_op) {
    %muli = transform.structured.match ops{["arith.muli"]} in %fun : (!transform.any_op) -> !transform.any_op
    %h:2 = transform.split_handle %muli : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %p = transform.num_associations %h#0 : (!transform.any_op) -> !transform.param<i64>
    // expected-remark @below {{1}}
    transform.debug.emit_param_as_remark  %p : !transform.param<i64>
    %muli_2 = transform.structured.match ops{["arith.muli"]} in %fun : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{expected to contain 3 payloads but it contains 2 payloads}}
    %h_2:3 = transform.split_handle %muli_2 : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

func.func @split_handle(%a: index, %b: index, %c: index) {
  %0 = arith.muli %a, %b : index
  %1 = arith.muli %a, %c : index
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.sequence %root : !transform.any_op failures(suppress) {
    ^bb1(%fun: !transform.any_op):
      %muli = transform.structured.match ops{["arith.muli"]} in %fun : (!transform.any_op) -> !transform.any_op
      %h:2 = split_handle %muli : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
      %p = transform.num_associations %h#0 : (!transform.any_op) -> !transform.param<i64>
      // expected-remark @below {{1}}
      transform.debug.emit_param_as_remark  %p : !transform.param<i64>
      %muli_2 = transform.structured.match ops{["arith.muli"]} in %fun : (!transform.any_op) -> !transform.any_op
      // Silenceable failure and all handles are now empty.
      %h_2:3 = split_handle %muli_2 : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
      %p2 = transform.num_associations %h_2#0 : (!transform.any_op) -> !transform.param<i64>
      // expected-remark @below {{0}}
      transform.debug.emit_param_as_remark  %p2 : !transform.param<i64>
    }
    transform.yield
  }
}

// -----

func.func @split_handle(%a: index, %b: index, %c: index) {
  %0 = arith.muli %a, %b : index
  %1 = arith.muli %a, %c : index
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%fun: !transform.any_op) {
    %muli_2 = transform.structured.match ops{["arith.muli"]} in %fun : (!transform.any_op) -> !transform.any_op
    // No error, last result handle is empty.
    %h:3 = transform.split_handle %muli_2 {fail_on_payload_too_small = false} : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %p = transform.num_associations %h#0 : (!transform.any_op) -> !transform.param<i64>
    // expected-remark @below {{1}}
    transform.debug.emit_param_as_remark  %p : !transform.param<i64>
    %p2 = transform.num_associations %h#1 : (!transform.any_op) -> !transform.param<i64>
    // expected-remark @below {{1}}
    transform.debug.emit_param_as_remark  %p2 : !transform.param<i64>
    %p3 = transform.num_associations %h#2 : (!transform.any_op) -> !transform.param<i64>
    // expected-remark @below {{0}}
    transform.debug.emit_param_as_remark  %p3 : !transform.param<i64>
    transform.yield
  }
}

// -----

func.func @split_handle(%a: index, %b: index, %c: index) {
  %0 = arith.muli %a, %b : index
  %1 = arith.muli %a, %c : index
  %2 = arith.muli %a, %c : index
  %3 = arith.muli %a, %c : index
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%fun: !transform.any_op) {
    %muli_2 = transform.structured.match ops{["arith.muli"]} in %fun : (!transform.any_op) -> !transform.any_op
    %h:2 = transform.split_handle %muli_2 {overflow_result = 0} : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %p = transform.num_associations %h#0 : (!transform.any_op) -> !transform.param<i64>
    // expected-remark @below {{3}}
    transform.debug.emit_param_as_remark  %p : !transform.param<i64>
    %p2 = transform.num_associations %h#1 : (!transform.any_op) -> !transform.param<i64>
    // expected-remark @below {{1}}
    transform.debug.emit_param_as_remark  %p2 : !transform.param<i64>
    transform.yield
  }
}

// -----

func.func private @opaque() -> (i32, i32)

func.func @split_handle() {
  func.call @opaque() : () -> (i32, i32)
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%fun: !transform.any_op) {
    %op = transform.structured.match ops{["func.call"]} in %fun : (!transform.any_op) -> !transform.any_op
    %val = transform.get_result %op[all] : (!transform.any_op) -> !transform.any_value
    %p = transform.num_associations %val : (!transform.any_value) -> !transform.any_param
    // expected-remark @below {{total 2}}
    transform.debug.emit_param_as_remark %p, "total" : !transform.any_param
    %h:2 = transform.split_handle %val : (!transform.any_value) -> (!transform.any_value, !transform.any_value)
    %p1 = transform.num_associations %h#0 : (!transform.any_value) -> !transform.any_param
    %p2 = transform.num_associations %h#1 : (!transform.any_value) -> !transform.any_param
    // expected-remark @below {{first 1}}
    transform.debug.emit_param_as_remark %p1, "first" : !transform.any_param
    // expected-remark @below {{second 1}}
    transform.debug.emit_param_as_remark %p1, "second" : !transform.any_param
    transform.yield
  }
}

// -----

func.func private @opaque() -> (i32, i32)

func.func @split_handle() {
  func.call @opaque() : () -> (i32, i32)
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%fun: !transform.any_op) {
    %op = transform.structured.match ops{["func.call"]} in %fun : (!transform.any_op) -> !transform.any_op
    %val = transform.get_result %op[all] : (!transform.any_op) -> !transform.any_value
    %type = transform.get_type %val : (!transform.any_value) -> !transform.any_param
    %p = transform.num_associations %type : (!transform.any_param) -> !transform.any_param
    // expected-remark @below {{total 2}}
    transform.debug.emit_param_as_remark %p, "total" : !transform.any_param
    %h:2 = transform.split_handle %type : (!transform.any_param) -> (!transform.any_param, !transform.any_param)
    %p1 = transform.num_associations %h#0 : (!transform.any_param) -> !transform.any_param
    %p2 = transform.num_associations %h#1 : (!transform.any_param) -> !transform.any_param
    // expected-remark @below {{first 1}}
    transform.debug.emit_param_as_remark %p1, "first" : !transform.any_param
    // expected-remark @below {{second 1}}
    transform.debug.emit_param_as_remark %p1, "second" : !transform.any_param
    transform.yield
  }
}

// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%fun: !transform.any_op) {
    // expected-error @below {{op expects result types to implement the same transform interface as the operand type}}
    transform.split_handle %fun : (!transform.any_op) -> (!transform.any_op, !transform.any_value)
    transform.yield
  }
}

// -----

"test.some_op"() : () -> ()
"other_dialect.other_op"() : () -> ()

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.with_pdl_patterns %root : !transform.any_op {
    ^bb0(%arg0: !transform.any_op):
      pdl.pattern @some : benefit(1) {
        %0 = pdl.operation "test.some_op"
        pdl.rewrite %0 with "transform.dialect"
      }

      sequence %arg0 : !transform.any_op failures(propagate) {
      ^bb1(%arg1: !transform.any_op):
        %0 = pdl_match @some in %arg1 : (!transform.any_op) -> !transform.any_op
        %2 = transform.cast %0 : !transform.any_op to !transform.test_dialect_op
        transform.cast %2 : !transform.test_dialect_op to !transform.any_op
      }
    }
    transform.yield
  }
}

// -----

"test.some_op"() : () -> ()
"other_dialect.other_op"() : () -> ()

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.with_pdl_patterns %root : !transform.any_op {
    ^bb0(%arg0: !transform.any_op):
      pdl.pattern @other : benefit(1) {
        %0 = pdl.operation "other_dialect.other_op"
        pdl.rewrite %0 with "transform.dialect"
      }

      sequence %arg0 : !transform.any_op failures(propagate) {
      ^bb1(%arg1: !transform.any_op):
        %0 = pdl_match @other in %arg1 : (!transform.any_op) -> !transform.any_op
        // expected-error @below {{expected the payload operation to belong to the 'test' dialect}}
        %2 = transform.cast %0 : !transform.any_op to !transform.test_dialect_op
        transform.cast %2 : !transform.test_dialect_op to !transform.any_op
      }
    }
    transform.yield
  }
}

// -----

"test.some_op"() : () -> ()
"other_dialect.other_op"() : () -> ()

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.with_pdl_patterns %root : !transform.any_op {
    ^bb0(%arg0: !transform.any_op):
      pdl.pattern @some : benefit(1) {
        %0 = pdl.operation "test.some_op"
        pdl.rewrite %0 with "transform.dialect"
      }

      sequence %arg0 : !transform.any_op failures(propagate) {
      ^bb1(%arg1: !transform.any_op):
        %0 = pdl_match @some in %arg1 : (!transform.any_op) -> !transform.any_op
        %2 = transform.cast %0 : !transform.any_op to !transform.op<"test.some_op">
        transform.cast %2 : !transform.op<"test.some_op"> to !transform.any_op
      }
    }
    transform.yield
  }
}

// -----

"test.some_op"() : () -> ()
// expected-note @below {{payload operation}}
"other_dialect.other_op"() : () -> ()

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.with_pdl_patterns %root : !transform.any_op {
    ^bb0(%arg0: !transform.any_op):
      pdl.pattern @other : benefit(1) {
        %0 = pdl.operation "other_dialect.other_op"
        pdl.rewrite %0 with "transform.dialect"
      }

      sequence %arg0 : !transform.any_op failures(propagate) {
      ^bb1(%arg1: !transform.any_op):
        %0 = pdl_match @other in %arg1 : (!transform.any_op) -> !transform.any_op
        // expected-error @below {{incompatible payload operation name}}
        %2 = transform.cast %0 : !transform.any_op to !transform.op<"test.some_op">
        transform.cast %2 : !transform.op<"test.some_op"> to !transform.any_op
      }
    }
    transform.yield
  }
}

// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.with_pdl_patterns %root : !transform.any_op {
    ^bb0(%arg0: !transform.any_op):
      transform.sequence %arg0 : !transform.any_op failures(propagate) {
      ^bb0(%arg1: !transform.any_op):
        %0 = pdl_match @some in %arg1 : (!transform.any_op) -> !transform.any_op
        // here, the handles nested under are {%root, %arg0, %arg1, %0}
        // expected-remark @below {{4 handles nested under}}
        transform.test_report_number_of_tracked_handles_nested_under %arg1 : !transform.any_op
        // expected-remark @below {{erased}}
        transform.test_emit_remark_and_erase_operand %0, "erased" : !transform.any_op
        // here, the handles nested under are only {%root, %arg0, %arg1}
        // expected-remark @below {{3 handles nested under}}
        transform.test_report_number_of_tracked_handles_nested_under %arg1 : !transform.any_op
      }

      pdl.pattern @some : benefit(1) {
        %0 = pdl.operation "test.some_op"
        pdl.rewrite %0 with "transform.dialect"
      }
    }
    transform.yield
  }
}

"test.some_op"() : () -> ()

// -----

func.func @split_handle(%a: index, %b: index, %c: index) {
  %0 = arith.muli %a, %b : index
  %1 = arith.muli %a, %c : index
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.sequence %root : !transform.any_op -> !transform.any_op failures(propagate) {
    ^bb1(%fun: !transform.any_op):
      %muli = transform.structured.match ops{["arith.muli"]} in %fun : (!transform.any_op) -> !transform.any_op
      // expected-error @below {{expected to contain 3 payloads but it contains 2 payloads}}
      %h_2:3 = split_handle %muli : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
      /// Test that yield does not crash in the presence of silenceable error in
      /// propagate mode.
      yield %fun : !transform.any_op
    }
    transform.yield
  }
}

// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.sequence %root : !transform.any_op -> !transform.any_op failures(suppress) {
    ^bb0(%arg0: !transform.any_op):
      %muli = transform.structured.match ops{["arith.muli"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      // Edge case propagating empty handles in splitting.
      %0:3 = split_handle %muli : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
      // Test does not crash when accessing the empty handle.
      yield %0#0 : !transform.any_op
    }
    transform.yield
  }
}

// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.test_produce_param (0 : i32) : !transform.test_dialect_param
    // expected-remark @below {{0 : i32}}
    transform.debug.emit_param_as_remark  %0 : !transform.test_dialect_param
    transform.yield
  }
}

// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    // expected-error @below {{expected the type of the parameter attribute ('i32') to match the parameter type ('i64')}}
    transform.test_produce_param (0 : i32) : !transform.param<i64>
    transform.yield
  }
}

// -----


module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.test_add_to_param 40
    %1 = transform.test_add_to_param %0, 2
    // expected-remark @below {{42 : i32}}
    transform.debug.emit_param_as_remark  %1 : !transform.test_dialect_param
    transform.yield
  }
}

// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.test_produce_param_with_number_of_test_ops %0 : !transform.any_op
    // expected-remark @below {{1 : i32, 3 : i32}}
    transform.debug.emit_param_as_remark  %1 : !transform.test_dialect_param
    %2 = transform.test_add_to_param %1, 100
    // expected-remark @below {{101 : i32, 103 : i32}}
    transform.debug.emit_param_as_remark  %2 : !transform.test_dialect_param
    transform.yield
  }
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
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    // expected-error @below {{expected to produce an Operation * for result #0}}
    transform.test_produce_transform_param_or_forward_operand %arg0
      { first_result_is_param }
      : (!transform.any_op) -> (!transform.any_op, !transform.param<i64>)
    transform.yield
  }
}

// -----

// Should not fail.

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    transform.test_produce_transform_param_or_forward_operand %arg0
      { first_result_is_null }
      : (!transform.any_op) -> (!transform.any_op, !transform.param<i64>)
    transform.yield
  }
}

// -----

// expected-note @below {{when applied to this op}}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    // expected-error @below {{expected to produce an Attribute for result #1}}
    transform.test_produce_transform_param_or_forward_operand %arg0
      { second_result_is_handle }
      : (!transform.any_op) -> (!transform.any_op, !transform.param<i64>)
    transform.yield
  }
}

// -----

// expected-note @below {{when applied to this op}}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    // expected-error @below {{expected to produce a Value for result #0}}
    transform.test_produce_transform_param_or_forward_operand %arg0
      { second_result_is_handle }
      : (!transform.any_op) -> (!transform.any_value, !transform.param<i64>)
    transform.yield
  }
}

// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    // expected-error @below {{attempting to assign a null payload op to this transform value}}
    %0 = transform.test_produce_null_payload : !transform.any_op
    transform.yield
  }
}

// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    // expected-error @below {{attempting to assign a null parameter to this transform value}}
    %0 = transform.test_produce_null_param : !transform.param<i64>
    transform.yield
  }
}

// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    // expected-error @below {{attempting to assign a null payload value to this transform handle}}
    %0 = transform.test_produce_null_value : !transform.any_value
    transform.yield
  }
}

// -----

// expected-error @below {{could not find a nested named sequence with name: __transform_main}}
module {
}

// -----

module attributes {transform.with_named_sequence} {
  // expected-remark @below {{value handle}}
  // expected-note @below {{value handle points to a block argument #0 in block #0 in region #0}}
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.test_produce_value_handle_to_self_operand %arg0 : (!transform.any_op) -> !transform.any_value
    transform.debug.emit_remark_at %0, "value handle" : !transform.any_value
    transform.yield
  }
}

// -----

// expected-remark @below {{result handle}}
// expected-note @below {{value handle points to an op result #1}}
%0:2 = "test.get_two_results"() : () -> (i32, i32)
// expected-remark @below {{result handle}}
// expected-note @below {{value handle points to an op result #1}}
%1:3 = "test.get_three_results"() : () -> (i32, i32, f32)

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %2 = transform.structured.match ops{["test.get_two_results", "test.get_three_results"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %3 = transform.test_produce_value_handle_to_result %2, 1 : (!transform.any_op) -> !transform.any_value
    transform.debug.emit_remark_at %3, "result handle" : !transform.any_value
    transform.yield
  }
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

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %2 = transform.structured.match ops{["test.match_anchor"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %3 = transform.test_produce_value_handle_to_argument_of_parent_block %2, 2 : (!transform.any_op) -> !transform.any_value
    transform.debug.emit_remark_at %3, "block argument handle" : !transform.any_value
    transform.yield
  }
}

// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    // expected-note @below {{value defined here with type '!transform.test_dialect_param'}}
    %0 = transform.test_produce_param_with_number_of_test_ops %arg0 : !transform.any_op
    // expected-error @below {{unexpectedly consumed a value that is not a handle as operand #0}}
    transform.test_consume_operand %0 : !transform.test_dialect_param
    transform.yield
  }
}

// -----

// expected-remark @below {{addi operand}}
// expected-note @below {{value handle points to a block argument #0}}
func.func @get_operand_of_op(%arg0: index, %arg1: index) -> index {
  %r = arith.addi %arg0, %arg1 : index
  return %r : index
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %addi = transform.structured.match ops{["arith.addi"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %operand = transform.get_operand %addi[0] : (!transform.any_op) -> !transform.any_value
    transform.debug.emit_remark_at %operand, "addi operand" : !transform.any_value
    transform.yield
  }
}

// -----

func.func @get_out_of_bounds_operand_of_op(%arg0: index, %arg1: index) -> index {
  // expected-note @below {{while considering positions of this payload operation}}
  %r = arith.addi %arg0, %arg1 : index
  return %r : index
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %addi = transform.structured.match ops{["arith.addi"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{position overflow 2 (updated from 2) for maximum 2}}
    %operand = transform.get_operand %addi[2] : (!transform.any_op) -> !transform.any_value
    transform.debug.emit_remark_at %operand, "addi operand" : !transform.any_value
    transform.yield
  }
}

// -----

// expected-remark @below {{addi operand}}
// expected-note @below {{value handle points to a block argument #1}}
func.func @get_inverted_operand_of_op(%arg0: index, %arg1: index) -> index {
  %r = arith.addi %arg0, %arg1 : index
  return %r : index
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %addi = transform.structured.match ops{["arith.addi"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %operand = transform.get_operand %addi[except(0)] : (!transform.any_op) -> !transform.any_value
    transform.debug.emit_remark_at %operand, "addi operand" : !transform.any_value
    transform.yield
  }
}

// -----

func.func @get_multiple_operands_of_op(%arg0: index, %arg1: index) -> index {
  %r = arith.addi %arg0, %arg1 : index
  return %r : index
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %addui = transform.structured.match ops{["arith.addi"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %operands = transform.get_operand %addui[all] : (!transform.any_op) -> !transform.any_value
    %p = transform.num_associations %operands : (!transform.any_value) -> !transform.param<i64>
    // expected-remark @below {{2}}
    transform.debug.emit_param_as_remark %p : !transform.param<i64>
    transform.yield
  }
}

// -----

func.func @get_result_of_op(%arg0: index, %arg1: index) -> index {
  // expected-remark @below {{addi result}}
  // expected-note @below {{value handle points to an op result #0}}
  %r = arith.addi %arg0, %arg1 : index
  return %r : index
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %addi = transform.structured.match ops{["arith.addi"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %result = transform.get_result %addi[0] : (!transform.any_op) -> !transform.any_value
    transform.debug.emit_remark_at %result, "addi result" : !transform.any_value
    transform.yield
  }
}

// -----

func.func @get_out_of_bounds_result_of_op(%arg0: index, %arg1: index) -> index {
  // expected-note @below {{while considering positions of this payload operation}}
  %r = arith.addi %arg0, %arg1 : index
  return %r : index
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %addi = transform.structured.match ops{["arith.addi"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{position overflow 1 (updated from 1) for maximum 1}}
    %result = transform.get_result %addi[1] : (!transform.any_op) -> !transform.any_value
    transform.debug.emit_remark_at %result, "addi result" : !transform.any_value
    transform.yield
  }
}

// -----

func.func @get_result_of_op(%arg0: index, %arg1: index) -> index {
  // expected-remark @below {{matched}}
  %r = arith.addi %arg0, %arg1 : index
  return %r : index
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %addi = transform.structured.match ops{["arith.addi"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %result = transform.get_result %addi[0] : (!transform.any_op) -> !transform.any_value
    %op = transform.get_defining_op %result : (!transform.any_value) -> !transform.any_op
    transform.debug.emit_remark_at %op, "matched" : !transform.any_op
    transform.yield
  }
}

// -----

func.func @get_multiple_result_of_op(%arg0: index, %arg1: index) -> (index, i1) {
  %r, %b = arith.addui_extended %arg0, %arg1 : index, i1
  return %r, %b : index, i1
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %addui = transform.structured.match ops{["arith.addui_extended"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %results = transform.get_result %addui[all] : (!transform.any_op) -> !transform.any_value
    %p = transform.num_associations %results : (!transform.any_value) -> !transform.param<i64>
    // expected-remark @below {{2}}
    transform.debug.emit_param_as_remark %p : !transform.param<i64>
    transform.yield
  }
}

// -----

// expected-note @below {{target value}}
func.func @get_result_of_op_bbarg(%arg0: index, %arg1: index) -> index {
  %r = arith.addi %arg0, %arg1 : index
  return %r : index
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %addi = transform.structured.match ops{["arith.addi"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %bbarg = transform.test_produce_value_handle_to_argument_of_parent_block %addi, 0 : (!transform.any_op) -> !transform.any_value
    // expected-error @below {{cannot get defining op of block argument}}
    %op = transform.get_defining_op %bbarg : (!transform.any_value) -> !transform.any_op
    transform.debug.emit_remark_at %op, "matched" : !transform.any_op
    transform.yield
  }
}

// -----

module @named_inclusion attributes { transform.with_named_sequence } {

  transform.named_sequence @foo(%arg0: !transform.any_op {transform.readonly}) -> () {
    // expected-remark @below {{applying transformation "a"}}
    transform.test_transform_op "a"
    transform.yield
  }

  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    transform.include @foo failures(propagate) (%arg0) : (!transform.any_op) -> ()
    transform.yield
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

  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    transform.include @bar failures(suppress) (%arg0) : (!transform.any_op) -> ()
    transform.yield
  }
}

// -----

// expected-remark @below {{operation}}
module @named_operands attributes { transform.with_named_sequence } {

  transform.named_sequence @foo(%arg0: !transform.any_op {transform.readonly},
                                %arg1: !transform.any_value {transform.readonly}) -> () {
    transform.debug.emit_remark_at %arg0, "operation" : !transform.any_op
    transform.debug.emit_remark_at %arg1, "value" : !transform.any_value
    transform.yield
  }

  // expected-remark @below {{value}}
  // expected-note @below {{value handle points to a block argument #0 in block #0 in region #0}}
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.test_produce_value_handle_to_self_operand %arg0 : (!transform.any_op) -> !transform.any_value
    transform.include @foo failures(propagate) (%arg0, %0) : (!transform.any_op, !transform.any_value) -> ()
    transform.yield
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

  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0:2 = transform.include @foo failures(propagate) (%arg0) : (!transform.any_op) -> (!transform.any_op, !transform.any_value)
    transform.debug.emit_remark_at %0#0, "operation" : !transform.any_op
    transform.debug.emit_remark_at %0#1, "value" : !transform.any_value
    transform.yield
  }
}

// -----

module attributes { transform.with_named_sequence } {
  transform.named_sequence @match1(%current: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
    transform.test_succeed_if_operand_of_op_kind %current, "test.some_op" : !transform.any_op
    transform.yield %current : !transform.any_op
  }

  transform.named_sequence @match2(%current: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
    transform.test_succeed_if_operand_of_op_kind %current, "func.func" : !transform.any_op
    transform.yield %current : !transform.any_op
  }

  transform.named_sequence @action1(%current: !transform.any_op {transform.readonly}) {
    transform.debug.emit_remark_at %current, "matched1" : !transform.any_op
    transform.yield
  }
  transform.named_sequence @action2(%current: !transform.any_op {transform.readonly}) {
    transform.debug.emit_remark_at %current, "matched2" : !transform.any_op
    transform.yield
  }

  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.foreach_match in %root
        @match1 -> @action1,
        @match2 -> @action2
      : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }

  // expected-remark @below {{matched2}}
  func.func private @foo()
  // expected-remark @below {{matched2}}
  func.func private @bar()
  "test.testtest"() : () -> ()
  // expected-remark @below {{matched1}}
  "test.some_op"() : () -> ()
}

// -----

module attributes { transform.with_named_sequence } {
  transform.named_sequence @match(!transform.any_op {transform.readonly})
  transform.named_sequence @action()

  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    // expected-error @below {{unresolved external symbol @match}}
    transform.foreach_match in %root
      @match -> @action : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

module attributes { transform.with_named_sequence } {
  transform.named_sequence @match(%arg: !transform.any_op {transform.readonly}) {
    transform.yield
  }
  transform.named_sequence @action()

  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    // expected-error @below {{unresolved external symbol @action}}
    transform.foreach_match in %root
      @match -> @action : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

module attributes { transform.with_named_sequence } {
  transform.named_sequence @match(%arg: !transform.any_op {transform.readonly}) {
    // expected-error @below {{expected operations in the match part to implement MatchOpInterface}}
    "test.unknown_op"() : () -> ()
    transform.yield
  }
  transform.named_sequence @action() {
    transform.yield
  }

  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.foreach_match in %root
      @match -> @action : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

module attributes { transform.with_named_sequence } {
  transform.named_sequence @match_func(%arg0: !transform.any_op {transform.readonly})
    -> !transform.any_op {
    transform.match.operation_name %arg0 ["func.func"] : !transform.any_op
    transform.yield %arg0 : !transform.any_op
  }

  transform.named_sequence @print_func(%arg0: !transform.any_op {transform.readonly}) {
    transform.debug.emit_remark_at %arg0, "matched func" : !transform.any_op
    transform.yield
  }

  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    transform.foreach_match in %arg0 @match_func -> @print_func : (!transform.any_op) -> !transform.any_op
    transform.yield
  }

  // expected-remark @below {{matched func}}
  func.func @payload() {
    return
  }

  // expected-remark @below {{matched func}}
  func.func private @declaration()

  "test.something_else"() : () -> ()
}

// -----

module attributes { transform.with_named_sequence } {
  transform.named_sequence @eq_1(%arg0: !transform.any_op {transform.readonly})
    -> !transform.any_op {
    transform.match.operation_name %arg0 ["func.func"] : !transform.any_op
    %0 = transform.test_produce_param_with_number_of_test_ops %arg0 : !transform.any_op
    %1 = transform.param.constant 1 : i32 -> !transform.test_dialect_param
    transform.match.param.cmpi eq %0, %1 : !transform.test_dialect_param
    transform.debug.emit_remark_at %arg0, "matched == 1" : !transform.any_op
    transform.yield %arg0 : !transform.any_op
  }

  transform.named_sequence @ne_0(%arg0: !transform.any_op {transform.readonly})
    -> !transform.any_op {
    transform.match.operation_name %arg0 ["func.func"] : !transform.any_op
    %0 = transform.test_produce_param_with_number_of_test_ops %arg0 : !transform.any_op
    %1 = transform.param.constant 0 : i32 -> !transform.test_dialect_param
    transform.match.param.cmpi ne %0, %1 : !transform.test_dialect_param
    transform.debug.emit_remark_at %arg0, "matched != 0" : !transform.any_op
    transform.yield %arg0 : !transform.any_op
  }

  transform.named_sequence @gt_m1(%arg0: !transform.any_op {transform.readonly})
    -> !transform.any_op {
    transform.match.operation_name %arg0 ["func.func"] : !transform.any_op
    %0 = transform.test_produce_param_with_number_of_test_ops %arg0 : !transform.any_op
    %1 = transform.param.constant -1 : i32 -> !transform.test_dialect_param
    transform.match.param.cmpi gt %0, %1 : !transform.test_dialect_param
    transform.debug.emit_remark_at %arg0, "matched > -1" : !transform.any_op
    transform.yield %arg0 : !transform.any_op
  }

  transform.named_sequence @ge_1(%arg0: !transform.any_op {transform.readonly})
    -> !transform.any_op {
    transform.match.operation_name %arg0 ["func.func"] : !transform.any_op
    %0 = transform.test_produce_param_with_number_of_test_ops %arg0 : !transform.any_op
    %1 = transform.param.constant 1 : i32 -> !transform.test_dialect_param
    transform.match.param.cmpi ge %0, %1 : !transform.test_dialect_param
    transform.debug.emit_remark_at %arg0, "matched >= 1" : !transform.any_op
    transform.yield %arg0 : !transform.any_op
  }

  transform.named_sequence @lt_1(%arg0: !transform.any_op {transform.readonly})
    -> !transform.any_op {
    transform.match.operation_name %arg0 ["func.func"] : !transform.any_op
    %0 = transform.test_produce_param_with_number_of_test_ops %arg0 : !transform.any_op
    %1 = transform.param.constant 1 : i32 -> !transform.test_dialect_param
    transform.match.param.cmpi lt %0, %1 : !transform.test_dialect_param
    transform.debug.emit_remark_at %arg0, "matched < 1" : !transform.any_op
    transform.yield %arg0 : !transform.any_op
  }

  transform.named_sequence @le_1(%arg0: !transform.any_op {transform.readonly})
    -> !transform.any_op {
    transform.match.operation_name %arg0 ["func.func"] : !transform.any_op
    %0 = transform.test_produce_param_with_number_of_test_ops %arg0 : !transform.any_op
    %1 = transform.param.constant 1 : i32 -> !transform.test_dialect_param
    transform.match.param.cmpi le %0, %1 : !transform.test_dialect_param
    transform.debug.emit_remark_at %arg0, "matched <= 1" : !transform.any_op
    transform.yield %arg0 : !transform.any_op
  }

  transform.named_sequence @do_nothing(%arg0: !transform.any_op {transform.readonly}) {
    transform.yield
  }

  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.foreach_match in %arg0 @eq_1 -> @do_nothing : (!transform.any_op) -> !transform.any_op
    %1 = transform.foreach_match in %0 @ne_0 -> @do_nothing : (!transform.any_op) -> !transform.any_op
    %2 = transform.foreach_match in %1 @gt_m1 -> @do_nothing : (!transform.any_op) -> !transform.any_op
    %3 = transform.foreach_match in %2 @ge_1 -> @do_nothing : (!transform.any_op) -> !transform.any_op
    %4 = transform.foreach_match in %3 @lt_1 -> @do_nothing : (!transform.any_op) -> !transform.any_op
    %5 = transform.foreach_match in %4 @le_1 -> @do_nothing : (!transform.any_op) -> !transform.any_op
    transform.yield
  }

  // expected-remark @below {{matched > -1}}
  // expected-remark @below {{matched < 1}}
  // expected-remark @below {{matched <= 1}}
  func.func private @declaration()

  // expected-remark @below {{matched == 1}}
  // expected-remark @below {{matched != 0}}
  // expected-remark @below {{matched > -1}}
  // expected-remark @below {{matched >= 1}}
  // expected-remark @below {{matched <= 1}}
  func.func @definition() {
    "test.something"() : () -> ()
    return
  }
}

// -----

// CHECK-LABEL: func @test_tracked_rewrite() {
//  CHECK-NEXT:   transform.test_dummy_payload_op  {new_op} : () -> i1
//  CHECK-NEXT:   transform.test_dummy_payload_op  {new_op} : () -> i1
//  CHECK-NEXT:   return
//  CHECK-NEXT: }
func.func @test_tracked_rewrite() {
  %0 = transform.test_dummy_payload_op {replace_me} : () -> (i1)
  %1 = transform.test_dummy_payload_op {erase_me} : () -> (i1)
  %2 = transform.test_dummy_payload_op {replace_me} : () -> (i1)
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %0 = transform.structured.match ops{["transform.test_dummy_payload_op"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    // expected-remark @below {{2 iterations}}
    transform.test_tracked_rewrite %0 : (!transform.any_op) -> ()
    // One replacement op (test.drop_mapping) is dropped from the mapping.
    %p = transform.num_associations %0 : (!transform.any_op) -> !transform.param<i64>
    // expected-remark @below {{2}}
    transform.debug.emit_param_as_remark  %p : !transform.param<i64>
    transform.yield
  }
}

// -----

// Parameter deduplication happens by value

module attributes {transform.with_named_sequence} {

  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %1 = transform.param.constant 1 -> !transform.param<i64>
    %2 = transform.param.constant 1 -> !transform.param<i64>
    %3 = transform.param.constant 2 -> !transform.param<i64>
    %4 = transform.merge_handles %1, %2 { deduplicate } : !transform.param<i64>
    %p = transform.num_associations %4 : (!transform.param<i64>) -> !transform.param<i64>
    // expected-remark @below {{1}}
    transform.debug.emit_param_as_remark %p : !transform.param<i64>

    %5 = transform.merge_handles %1, %1 { deduplicate } : !transform.param<i64>
    %p2 = transform.num_associations %5 : (!transform.param<i64>) -> !transform.param<i64>
    // expected-remark @below {{1}}
    transform.debug.emit_param_as_remark %p2 : !transform.param<i64>

    %6 = transform.merge_handles %1, %3 { deduplicate } : !transform.param<i64>
    %p3 = transform.num_associations %6 : (!transform.param<i64>) -> !transform.param<i64>
    // expected-remark @below {{2}}
    transform.debug.emit_param_as_remark %p3 : !transform.param<i64>

    %7 = transform.merge_handles %1, %1, %2, %3 : !transform.param<i64>
    %p4 = transform.num_associations %7 : (!transform.param<i64>) -> !transform.param<i64>
    // expected-remark @below {{4}}
    transform.debug.emit_param_as_remark %p4 : !transform.param<i64>
    transform.yield
  }
}

// -----

%0:3 = "test.get_two_results"() : () -> (i32, i32, f32)

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %1 = transform.structured.match ops{["test.get_two_results"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %2 = transform.test_produce_value_handle_to_result %1, 0 : (!transform.any_op) -> !transform.any_value
    %3 = transform.test_produce_value_handle_to_result %1, 1 : (!transform.any_op) -> !transform.any_value

    %4 = transform.merge_handles %2, %2 { deduplicate } : !transform.any_value
    %p = transform.num_associations %4 : (!transform.any_value) -> !transform.param<i64>
    // expected-remark @below {{1}}
    transform.debug.emit_param_as_remark %p : !transform.param<i64>

    %5 = transform.merge_handles %2, %3 { deduplicate } : !transform.any_value
    %p2 = transform.num_associations %5 : (!transform.any_value) -> !transform.param<i64>
    // expected-remark @below {{2}}
    transform.debug.emit_param_as_remark %p2 : !transform.param<i64>

    %6 = transform.test_produce_value_handle_to_result %1, 0 : (!transform.any_op) -> !transform.any_value
    %7 = transform.merge_handles %2, %6 { deduplicate } : !transform.any_value
    %p3 = transform.num_associations %6 : (!transform.any_value) -> !transform.param<i64>
    // expected-remark @below {{1}}
    transform.debug.emit_param_as_remark %p3 : !transform.param<i64>

    %8 = transform.merge_handles %2, %2, %3, %4 : !transform.any_value
    %p4 = transform.num_associations %8 : (!transform.any_value) -> !transform.param<i64>
    // expected-remark @below {{4}}
    transform.debug.emit_param_as_remark %p4 : !transform.param<i64>
    transform.yield
  }
}
// -----

// CHECK-LABEL: func @test_annotation()
//  CHECK-NEXT:   "test.annotate_me"()
//  CHECK-SAME:                        any_attr = "example"
//  CHECK-SAME:                        broadcast_attr = 2 : i64
//  CHECK-SAME:                        new_attr = 1 : i32
//  CHECK-SAME:                        unit_attr
//  CHECK-NEXT:   "test.annotate_me"()
//  CHECK-SAME:                        any_attr = "example"
//  CHECK-SAME:                        broadcast_attr = 2 : i64
//  CHECK-SAME:                        existing_attr = "test"
//  CHECK-SAME:                        new_attr = 1 : i32
//  CHECK-SAME:                        unit_attr
//  CHECK-NEXT:   "test.annotate_me"()
//  CHECK-SAME:                        any_attr = "example"
//  CHECK-SAME:                        broadcast_attr = 2 : i64
//  CHECK-SAME:                        new_attr = 1 : i32
//  CHECK-SAME:                        unit_attr
func.func @test_annotation() {
  %0 = "test.annotate_me"() : () -> (i1)
  %1 = "test.annotate_me"() {existing_attr = "test"} : () -> (i1)
  %2 = "test.annotate_me"() {new_attr = 0} : () -> (i1)
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.structured.match ops{["test.annotate_me"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %1 = transform.test_produce_param_with_number_of_test_ops %0 : !transform.any_op
    transform.annotate %0 "new_attr" = %1 : !transform.any_op, !transform.test_dialect_param

    %2 = transform.param.constant 2 -> !transform.param<i64>
    transform.annotate %0 "broadcast_attr" = %2 : !transform.any_op, !transform.param<i64>
    transform.annotate %0 "unit_attr" : !transform.any_op

    %3 = transform.param.constant "example" -> !transform.any_param
    transform.annotate %0 "any_attr" = %3 : !transform.any_op, !transform.any_param
    transform.yield
  }
}

// -----

func.func @notify_payload_op_replaced(%arg0: index, %arg1: index) {
  %0 = arith.muli %arg0, %arg1 {original} : index
  // expected-remark @below{{updated handle}}
  %1 = arith.muli %arg0, %arg1 {replacement} : index
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %0 = transform.structured.match attributes{original} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.match attributes{replacement} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.test_notify_payload_op_replaced %0, %1 : (!transform.any_op, !transform.any_op) -> ()
    transform.debug.emit_remark_at %0, "updated handle" : !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: func @test_apply_cse()
//       CHECK:   %[[const:.*]] = arith.constant 0 : index
//       CHECK:   %[[ex1:.*]] = scf.execute_region -> index {
//       CHECK:     scf.yield %[[const]]
//       CHECK:   }
//       CHECK:   %[[ex2:.*]] = scf.execute_region -> index {
//       CHECK:     scf.yield %[[const]]
//       CHECK:   }
//       CHECK:   return %[[const]], %[[ex1]], %[[ex2]]
func.func @test_apply_cse() -> (index, index, index) {
  // expected-remark @below{{eliminated 1}}
  // expected-remark @below{{eliminated 2}}
  %0 = arith.constant 0 : index
  %1 = scf.execute_region -> index {
    %2 = arith.constant 0 : index
    scf.yield %2 : index
  } {first}
  %3 = scf.execute_region -> index {
    %4 = arith.constant 0 : index
    scf.yield %4 : index
  } {second}
  return %0, %1, %3 : index, index, index
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %0 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %first = transform.structured.match attributes{first} in %0 : (!transform.any_op) -> !transform.any_op
    %elim_first = transform.structured.match ops{["arith.constant"]} in %first : (!transform.any_op) -> !transform.any_op
    %second = transform.structured.match attributes{first} in %0 : (!transform.any_op) -> !transform.any_op
    %elim_second = transform.structured.match ops{["arith.constant"]} in %first : (!transform.any_op) -> !transform.any_op

    // There are 3 arith.constant ops.
    %all = transform.structured.match ops{["arith.constant"]} in %0 : (!transform.any_op) -> !transform.any_op
    %p = transform.num_associations %all : (!transform.any_op) -> !transform.param<i64>
    // expected-remark @below{{3}}
    transform.debug.emit_param_as_remark %p : !transform.param<i64>
    // "deduplicate" has no effect because these are 3 different ops.
    %merged_before = transform.merge_handles deduplicate %all : !transform.any_op
    %p2 = transform.num_associations %merged_before : (!transform.any_op) -> !transform.param<i64>
    // expected-remark @below{{3}}
    transform.debug.emit_param_as_remark %p2 : !transform.param<i64>

    // Apply CSE.
    transform.apply_cse to %0 : !transform.any_op

    // The handle is still mapped to 3 arith.constant ops.
    %p3 = transform.num_associations %all : (!transform.any_op) -> !transform.param<i64>
    // expected-remark @below{{3}}
    transform.debug.emit_param_as_remark %p3 : !transform.param<i64>
    // But they are all the same op.
    %merged_after = transform.merge_handles deduplicate %all : !transform.any_op
    %p4 = transform.num_associations %merged_after : (!transform.any_op) -> !transform.param<i64>
    // expected-remark @below{{1}}
    transform.debug.emit_param_as_remark %p4 : !transform.param<i64>

    // The other handles were also updated.
    transform.debug.emit_remark_at %elim_first, "eliminated 1" : !transform.any_op
    %p5 = transform.num_associations %elim_first : (!transform.any_op) -> !transform.param<i64>
    // expected-remark @below{{1}}
    transform.debug.emit_param_as_remark %p5 : !transform.param<i64>
    transform.debug.emit_remark_at %elim_second, "eliminated 2" : !transform.any_op
    %p6 = transform.num_associations %elim_second : (!transform.any_op) -> !transform.param<i64>
    // expected-remark @below{{1}}
    transform.debug.emit_param_as_remark %p6 : !transform.param<i64>
    transform.yield
  }
}

// -----

// CHECK-LABEL: func @test_licm(
//       CHECK:   arith.muli
//       CHECK:   scf.for {{.*}} {
//       CHECK:     vector.print
//       CHECK:   }
func.func @test_licm(%arg0: index, %arg1: index, %arg2: index) {
  scf.for %iv = %arg0 to %arg1 step %arg2 {
    %0 = arith.muli %arg0, %arg1 : index
    vector.print %0 : index
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %0 = transform.structured.match ops{["scf.for"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.apply_licm to %0 : !transform.any_op
    transform.yield
  }
}

// -----

// expected-note @below{{when applied to this op}}
module attributes {transform.with_named_sequence} {
  func.func @test_licm_invalid() {
    return
  }

  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    // expected-error @below{{transform applied to the wrong op kind}}
    transform.apply_licm to %arg1 : !transform.any_op
    transform.yield
  }
}

// -----

func.func @get_parent_op() {
  // expected-remark @below{{found test.foo parent}}
  "test.foo"() ({
    // expected-remark @below{{direct parent}}
    "test.bar"() ({
      "test.qux"() : () -> ()
      "test.qux"() : () -> ()
    }) : () -> ()
  }) : () -> ()
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op) {
    %0 = transform.structured.match ops{["test.qux"]} in %arg1 : (!transform.any_op) -> !transform.any_op

    // Get parent by name.
    %1 = transform.get_parent_op %0 {op_name = "test.foo"} : (!transform.any_op) -> !transform.any_op
    transform.debug.emit_remark_at %1, "found test.foo parent" : !transform.any_op

    // Get immediate parent.
    %2 = transform.get_parent_op %0 : (!transform.any_op) -> !transform.any_op
    transform.debug.emit_remark_at %2, "direct parent" : !transform.any_op
    %p = transform.num_associations %2 : (!transform.any_op) -> !transform.param<i64>
    // expected-remark @below{{2}}
    transform.debug.emit_param_as_remark %p : !transform.param<i64>

    // Deduplicate results.
    %3 = transform.structured.match ops{["test.qux"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %4 = transform.get_parent_op %3 {deduplicate} : (!transform.any_op) -> !transform.any_op
    %p2 = transform.num_associations %4 : (!transform.any_op) -> !transform.param<i64>
    // expected-remark @below{{1}}
    transform.debug.emit_param_as_remark %p2 : !transform.param<i64>
    transform.yield
  }
}


// -----

// expected-note @below {{target op}}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    // expected-error @below{{could not find a parent op that matches all requirements}}
    %3 = transform.get_parent_op %arg0 {op_name = "builtin.module"} : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

func.func @cast(%arg0: f32) -> f64 {
  // expected-remark @below{{f64}}
  %0 = arith.extf %arg0 : f32 to f64
  return %0 : f64
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.structured.match ops{["arith.extf"]} in %arg0 : (!transform.any_op) -> !transform.op<"arith.extf">
    %1 = transform.get_result %0[0] : (!transform.op<"arith.extf">) -> !transform.any_value
    %2 = transform.get_type %1 : (!transform.any_value) -> !transform.type
    transform.debug.emit_param_as_remark %2 at %0 : !transform.type, !transform.op<"arith.extf">
    transform.yield
  }
}

// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    // expected-error @below {{expected type attribute, got 0 : i32}}
    transform.test_produce_param (0 : i32) : !transform.type
    transform.yield
  }
}

// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    // expected-error @below {{expected affine map attribute, got 0 : i32}}
    transform.test_produce_param (0 : i32) : !transform.affine_map
    transform.yield
  }
}

// -----

// CHECK-LABEL: @type_param_anchor
func.func private @type_param_anchor()

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    // CHECK: test_produce_param(f32) : !transform.type
    transform.test_produce_param(f32) : !transform.type
    transform.yield
  }
}

// -----

// CHECK-LABEL: @affine_map_param_anchor
func.func private @affine_map_param_anchor()

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    // CHECK: test_produce_param(#{{.*}}) : !transform.affine_map
    transform.test_produce_param(affine_map<(d0) -> ()>) : !transform.affine_map
    transform.yield
  }
}

// -----

func.func @verify_success(%arg0: f64) -> f64 {
  return %arg0 : f64
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.verify %0 : !transform.any_op
    transform.yield
  }
}

// -----

// expected-error @below{{fail_to_verify is set}}
// expected-note @below{{payload op}}
func.func @verify_failure(%arg0: f64) -> f64 {
  return %arg0 : f64
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.test_produce_invalid_ir %0 : !transform.any_op
    // expected-error @below{{failed to verify payload op}}
    transform.verify %0 : !transform.any_op
    transform.yield
  }
}

// -----

func.func @select() {
  // expected-remark @below{{found foo}}
  "test.foo"() : () -> ()
  // expected-remark @below{{found bar}}
  "test.bar"() : () -> ()
  // expected-remark @below{{found foo}}
  "test.foo"() : () -> ()
  func.return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    // Match all ops inside the function (including the function itself).
    %func_op = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %0 = transform.structured.match in %func_op : (!transform.any_op) -> !transform.any_op
    %p = transform.num_associations %0 : (!transform.any_op) -> !transform.param<i64>
    // expected-remark @below{{5}}
    transform.debug.emit_param_as_remark %p : !transform.param<i64>

    // Select "test.foo".
    %foo = transform.select "test.foo" in %0 : (!transform.any_op) -> !transform.any_op
    transform.debug.emit_remark_at %foo, "found foo" : !transform.any_op

    // Select "test.bar".
    %bar = transform.select "test.bar" in %0 : (!transform.any_op) -> !transform.any_op
    transform.debug.emit_remark_at %bar, "found bar" : !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: func @apply_dce(
//  CHECK-NEXT:   memref.store
//  CHECK-NEXT:   return
func.func @apply_dce(%f: f32, %m: memref<5xf32>, %idx: index) {
  // Two dead ops, interleaved with a non-dead op.
  %0 = tensor.empty() : tensor<5xf32>
  memref.store %f, %m[%idx] : memref<5xf32>
  %1 = tensor.insert %f into %0[%idx] : tensor<5xf32>
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %func_op = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %empty_op = transform.structured.match ops{["tensor.empty"]} in %func_op : (!transform.any_op) -> !transform.any_op
    transform.apply_dce to %func_op : !transform.any_op

    %p = transform.num_associations %empty_op : (!transform.any_op) -> !transform.param<i64>
    // expected-remark @below{{0}}
    transform.debug.emit_param_as_remark %p : !transform.param<i64>
    transform.yield
  }
}


// -----

func.func @no_constant_under_loop(%lb: index, %ub: index, %step: index) {
  scf.for %i= %lb to %ub step %step {
    arith.constant 0 : index
  }
  return
}

module @named_inclusion attributes { transform.with_named_sequence } {
// Match `arith.constant`s that are not nested under a `scf.for` and ensure
// there are none in the program

  transform.named_sequence @print(%root: !transform.any_op {transform.readonly}) {
    transform.debug.emit_remark_at %root, "matched func" : !transform.any_op
    transform.yield
  }

  transform.named_sequence @match_constant_not_under_scf_for(%root: !transform.any_op {transform.readonly})
    -> !transform.any_op {
    transform.match.operation_name %root ["arith.constant"] : !transform.any_op
    %for = transform.get_parent_op %root { op_name = "scf.for", allow_empty_results }
      : (!transform.any_op) -> (!transform.any_op)
    transform.match.operation_empty %for : !transform.any_op
    transform.yield %root : !transform.any_op
  }

  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    transform.foreach_match in %arg0
        @match_constant_not_under_scf_for -> @print
      : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
}

// -----

func.func @no_constant_under_loop(%lb: index, %ub: index, %step: index) {
  // expected-remark @below {{no parent scf.for}}
  arith.constant 0 : index
  return
}

module @named_inclusion attributes { transform.with_named_sequence } {
  // Match `arith.constant`s that are not nested under a `scf.for` and ensure
  // there are none in the program

  transform.named_sequence @print(%root: !transform.any_op {transform.readonly}) {
    transform.debug.emit_remark_at %root, "no parent scf.for" : !transform.any_op
    transform.yield
  }

  transform.named_sequence @match_constant_not_under_scf_for(%root: !transform.any_op {transform.readonly})
    -> !transform.any_op {
    transform.match.operation_name %root ["arith.constant"] : !transform.any_op
    %for = transform.get_parent_op %root { op_name = "scf.for", allow_empty_results }
      : (!transform.any_op) -> (!transform.any_op)
    transform.match.operation_empty %for : !transform.any_op
    transform.yield %root : !transform.any_op
  }

  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    transform.foreach_match in %arg0
        @match_constant_not_under_scf_for -> @print
      : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
}

// -----

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    // expected-error @below {{result #0, associated with 2 payload objects, expected 1}}
    transform.collect_matching @matcher in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }

  transform.named_sequence @matcher(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op {
    %0 = transform.merge_handles %arg0, %arg0 : !transform.any_op
    transform.yield %0 : !transform.any_op
  }
}

// -----

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    // expected-error @below {{unresolved external symbol @matcher}}
    transform.collect_matching @matcher in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }

  transform.named_sequence @matcher(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op
}

// -----

module attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    // expected-remark @below {{matched}}
    %0 = transform.collect_matching @matcher in %arg0 : (!transform.any_op) -> !transform.any_op
    // expected-remark @below {{matched}}
    transform.debug.emit_remark_at %0, "matched" : !transform.any_op
    transform.yield
  }

  transform.named_sequence @matcher(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %arg0 ["transform.debug.emit_remark_at", "transform.collect_matching"] : !transform.any_op
    transform.yield %arg0 : !transform.any_op
  }
}
