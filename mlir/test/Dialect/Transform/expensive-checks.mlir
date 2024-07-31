// RUN: mlir-opt --transform-interpreter --split-input-file --verify-diagnostics %s

// expected-note @below {{ancestor payload op}}
func.func @func() {
  // expected-note @below {{nested payload op}}
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.with_pdl_patterns %root : !transform.any_op {
    ^bb0(%arg0: !transform.any_op):
      pdl.pattern @return : benefit(1) {
        %0 = operands
        %1 = types
        %2 = operation "func.return"(%0 : !pdl.range<value>) -> (%1 : !pdl.range<type>)
        rewrite %2 with "transform.dialect"
      }

      sequence %arg0 : !transform.any_op failures(propagate) {
      ^bb1(%arg1: !transform.any_op):
        // expected-note @below {{handle to invalidated ops}}
        %0 = pdl_match @return in %arg1 : (!transform.any_op) -> !transform.any_op
        %1 = get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
        // expected-note @below {{invalidated by this transform op that consumes its operand #0}}
        test_consume_operand %1 : !transform.any_op
        // expected-error @below {{op uses a handle invalidated by a previously executed transform op}}
        transform.debug.emit_remark_at %0, "remark" : !transform.any_op
      }
    }
    transform.yield
  }
}

// -----

func.func @func1() {
  // expected-note @below {{repeated target op}}
  return
}
func.func private @func2()

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {
    transform.with_pdl_patterns %root : !transform.any_op {
    ^bb0(%arg0: !transform.any_op):
      pdl.pattern @func : benefit(1) {
        %0 = operands
        %1 = types
        %2 = operation "func.func"(%0 : !pdl.range<value>) -> (%1 : !pdl.range<type>)
        rewrite %2 with "transform.dialect"
      }
      pdl.pattern @return : benefit(1) {
        %0 = operands
        %1 = types
        %2 = operation "func.return"(%0 : !pdl.range<value>) -> (%1 : !pdl.range<type>)
        rewrite %2 with "transform.dialect"
      }

      sequence %arg0 : !transform.any_op failures(propagate) {
      ^bb1(%arg1: !transform.any_op):
        %0 = pdl_match @func in %arg1 : (!transform.any_op) -> !transform.any_op
        %1 = pdl_match @return in %arg1 : (!transform.any_op) -> !transform.any_op
        %2 = replicate num(%0) %1 : !transform.any_op, !transform.any_op
        // expected-error @below {{a handle passed as operand #0 and consumed by this operation points to a payload entity more than once}}
        test_consume_operand %2 : !transform.any_op
        transform.debug.emit_remark_at %0, "remark" : !transform.any_op
      }
    }
    transform.yield
  }
}


// -----

// expected-note @below {{ancestor payload op}}
// expected-note @below {{nested payload op}}
module attributes {transform.with_named_sequence} {

  transform.named_sequence @__transform_main(%0: !transform.any_op) {
    %1 = transform.test_copy_payload %0 : (!transform.any_op) -> !transform.any_op
    // expected-note @below {{handle to invalidated ops}}
    %2 = transform.test_copy_payload %0 : (!transform.any_op) ->!transform.any_op
    // expected-note @below {{invalidated by this transform op that consumes its operand #0}}
    transform.test_consume_operand %1 : !transform.any_op
    // expected-error @below {{op uses a handle invalidated by a previously executed transform op}}
    transform.test_consume_operand %2 : !transform.any_op
    transform.yield
  }
}

// -----

// expected-note @below {{ancestor payload op}}
// expected-note @below {{nested payload op}}
module attributes {transform.with_named_sequence} {

  transform.named_sequence @__transform_main(%0: !transform.any_op) {
    %1 = transform.test_copy_payload %0 : (!transform.any_op) -> !transform.any_op
    // expected-note @below {{handle to invalidated ops}}
    %2 = transform.test_copy_payload %0 : (!transform.any_op) -> !transform.any_op
    // Consuming two handles in the same operation is invalid if they point
    // to overlapping sets of payload IR ops.
    //
    // expected-error @below {{op uses a handle invalidated by a previously executed transform op}}
    // expected-note @below {{invalidated by this transform op that consumes its operand #0 and invalidates all handles to payload IR entities}}
    transform.test_consume_operand %1, %2 : !transform.any_op, !transform.any_op
    transform.yield
  }
}

// -----

// Deduplication attribute allows "merge_handles" to take repeated operands.

module attributes {transform.with_named_sequence} {

  transform.named_sequence @__transform_main(%0: !transform.any_op) {
    %1 = transform.test_copy_payload %0 : (!transform.any_op) -> !transform.any_op
    %2 = transform.test_copy_payload %0 : (!transform.any_op) -> !transform.any_op
    transform.merge_handles %1, %2 { deduplicate } : !transform.any_op
    transform.yield
  }
}
// -----

// expected-note @below {{payload value}}
%0 = "test.match_anchor"() : () -> (i32)

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %2 = transform.structured.match ops{["test.match_anchor"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %3 = transform.test_produce_value_handle_to_result %2, 0 : (!transform.any_op) -> !transform.any_value
    // expected-note @below {{invalidated handle}}
    %4 = transform.test_produce_value_handle_to_result %2, 0 : (!transform.any_op) -> !transform.any_value
    // expected-note @below {{invalidated by this transform op that consumes its operand #0 and invalidates handles to the same values as associated with it}}
    transform.test_consume_operand %3 : !transform.any_value
    // expected-error @below {{op uses a handle invalidated by a previously executed transform op}}
    transform.test_consume_operand %4 : !transform.any_value
    transform.yield
  }
}

// -----

// expected-note @below {{ancestor op associated with the consumed handle}}
// expected-note @below {{payload value}}
// expected-note @below {{op defining the value as result #0}}
%0 = "test.match_anchor"() : () -> (i32)

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %2 = transform.structured.match ops{["test.match_anchor"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    // expected-note @below {{invalidated handle}}
    %3 = transform.test_produce_value_handle_to_result %2, 0 : (!transform.any_op) -> !transform.any_value
    // expected-note @below {{invalidated by this transform op that consumes its operand #0 and invalidates all handles to payload IR entities associated with this operand and entities nested in them}}
    transform.test_consume_operand %2 : !transform.any_op
    // expected-error @below {{op uses a handle invalidated by a previously executed transform op}}
    transform.test_consume_operand %3 : !transform.any_value
    transform.yield
  }
}

// -----

// expected-note @below {{ancestor op associated with the consumed handle}}
"test.match_anchor_1"() ({
^bb0:
  // expected-note @below {{op defining the value as result #0}}
  // expected-note @below {{payload value}}
  %0 = "test.match_anchor_2"() : () -> (i32)
  "test.region_terminator"() : () -> ()
}) : () -> ()

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %1 = transform.structured.match ops{["test.match_anchor_1"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.match ops{["test.match_anchor_2"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    // expected-note @below {{invalidated handle}}
    %3 = transform.test_produce_value_handle_to_result %2, 0 : (!transform.any_op) -> !transform.any_value
    // expected-note @below {{invalidated by this transform op that consumes its operand #0 and invalidates all handles to payload IR entities associated with this operand and entities nested in them}}
    transform.test_consume_operand %1 : !transform.any_op
    // expected-error @below {{op uses a handle invalidated by a previously executed transform op}}
    transform.test_consume_operand %3 : !transform.any_value
    transform.yield
  }
}

// -----

// expected-note @below {{ancestor op associated with the consumed handle}}
// expected-note @below {{op defining the value as block argument #0 of block #0 in region #0}}
"test.match_anchor_1"() ({
// expected-note @below {{payload value}}
^bb0(%arg0: i32):
  %0 = "test.match_anchor_2"() : () -> (i32)
  "test.region_terminator"() : () -> ()
}) : () -> ()

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %1 = transform.structured.match ops{["test.match_anchor_1"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.match ops{["test.match_anchor_2"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    // expected-note @below {{invalidated handle}}
    %3 = transform.test_produce_value_handle_to_argument_of_parent_block %2, 0 : (!transform.any_op) -> !transform.any_value
    // expected-note @below {{invalidated by this transform op that consumes its operand #0 and invalidates all handles to payload IR entities associated with this operand and entities nested in them}}
    transform.test_consume_operand %1 : !transform.any_op
    // expected-error @below {{op uses a handle invalidated by a previously executed transform op}}
    transform.test_consume_operand %3 : !transform.any_value
    transform.yield
  }
}

// -----

// expected-note @below {{ancestor op associated with the consumed handle}}
"test.match_anchor_1"() ({
^bb:
  // expected-note @below {{op defining the value as block argument #0 of block #0 in region #0}}
  "test.op_with_regions"() ({
  // expected-note @below {{payload value}}
  ^bb0(%arg0: i32):
    %0 = "test.match_anchor_2"() : () -> (i32)
    "test.region_terminator"() : () -> ()
  }): () -> ()
}) : () -> ()

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %1 = transform.structured.match ops{["test.match_anchor_1"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.match ops{["test.match_anchor_2"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    // expected-note @below {{invalidated handle}}
    %3 = transform.test_produce_value_handle_to_argument_of_parent_block %2, 0 : (!transform.any_op) -> !transform.any_value
    // expected-note @below {{invalidated by this transform op that consumes its operand #0 and invalidates all handles to payload IR entities associated with this operand and entities nested in them}}
    transform.test_consume_operand %1 : !transform.any_op
    // expected-error @below {{op uses a handle invalidated by a previously executed transform op}}
    transform.test_consume_operand %3 : !transform.any_value
    transform.yield
  }
}

// -----

// expected-note @below {{ancestor payload op}}
// expected-note @below {{nested payload op}}
// expected-note @below {{consumed handle points to this payload value}}
%0 = "test.match_anchor"() : () -> (i32)

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    // expected-note @below {{handle to invalidated ops}}
    %2 = transform.structured.match ops{["test.match_anchor"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %3 = transform.test_produce_value_handle_to_result %2, 0 : (!transform.any_op) -> !transform.any_value
    // expected-note @below {{invalidated by this transform op that consumes its operand #0 and invalidates all handles to payload IR entities associated with this operand and entities nested in them}}
    transform.test_consume_operand %3 : !transform.any_value
    // expected-error @below {{op uses a handle invalidated by a previously executed transform op}}
    transform.test_consume_operand %2 : !transform.any_op 
    transform.yield
  }
}

// -----

// expected-note @below {{ancestor payload op}}
// expected-note @below {{consumed handle points to this payload value}}
%0 = "test.match_anchor_1"() ({
^bb0:
  // expected-note @below {{nested payload op}}
  "test.match_anchor_2"() : () -> ()
  "test.region_terminator"() : () -> ()
}) : () -> (i32)

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %1 = transform.structured.match ops{["test.match_anchor_1"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    // expected-note @below {{handle to invalidated ops}}
    %2 = transform.structured.match ops{["test.match_anchor_2"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %3 = transform.test_produce_value_handle_to_result %1, 0 : (!transform.any_op) -> !transform.any_value
    // expected-note @below {{invalidated by this transform op that consumes its operand #0 and invalidates all handles to payload IR entities associated with this operand and entities nested in them}}
    transform.test_consume_operand %3 : !transform.any_value
    // expected-error @below {{op uses a handle invalidated by a previously executed transform op}}
    transform.test_consume_operand %2 : !transform.any_op
    transform.yield
  }
}


// -----

"test.match_anchor_1"() ({
// expected-note @below {{consumed handle points to this payload value}}
^bb0(%arg0: f32):
  // expected-note @below {{ancestor payload op}}
  // expected-note @below {{nested payload op}}
  "test.match_anchor_2"() : () -> ()
  "test.region_terminator"() : () -> ()
}) : () -> ()

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    // expected-note @below {{handle to invalidated ops}}
    %2 = transform.structured.match ops{["test.match_anchor_2"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %3 = transform.test_produce_value_handle_to_argument_of_parent_block %2, 0 : (!transform.any_op) -> !transform.any_value
    // expected-note @below {{invalidated by this transform op that consumes its operand #0 and invalidates all handles to payload IR entities associated with this operand and entities nested in them}}
    transform.test_consume_operand %3 : !transform.any_value
    // expected-error @below {{op uses a handle invalidated by a previously executed transform op}}
    transform.test_consume_operand %2 : !transform.any_op
    transform.yield
  }
}

// -----

"test.op_with_regions"() ({
// expected-note @below {{consumed handle points to this payload value}}
^bb(%arg0: i32):
  // expected-note @below {{ancestor payload op}}
  "test.op_with_regions"() ({
  ^bb0:
    // expected-note @below {{nested payload op}}
    "test.match_anchor_2"() : () -> ()
    "test.region_terminator"() : () -> ()
  }): () -> ()
  "test.match_anchor_1"() : () -> ()
}) : () -> ()

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %1 = transform.structured.match ops{["test.match_anchor_1"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    // expected-note @below {{handle to invalidated ops}}
    %2 = transform.structured.match ops{["test.match_anchor_2"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %3 = transform.test_produce_value_handle_to_argument_of_parent_block %1, 0 : (!transform.any_op) -> !transform.any_value
    // expected-note @below {{invalidated by this transform op that consumes its operand #0 and invalidates all handles to payload IR entities associated with this operand and entities nested in them}}
    transform.test_consume_operand %3 : !transform.any_value
    // expected-error @below {{op uses a handle invalidated by a previously executed transform op}}
    transform.test_consume_operand %2 : !transform.any_op
    transform.yield
  }
}

// -----

// Removing a block argument does not invalidate handles to operations in another block.
// Not expecting an error here.

"test.op_with_regions"() ({
^bb1(%arg0: i32):
  "test.match_anchor_1"() : () -> ()
^bb2:
  "test.match_anchor_2"() : () -> ()
}) : () -> ()

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %1 = transform.structured.match ops{["test.match_anchor_1"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.match ops{["test.match_anchor_2"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %3 = transform.test_produce_value_handle_to_argument_of_parent_block %1, 0 : (!transform.any_op) -> !transform.any_value
    transform.test_consume_operand %3 : !transform.any_value
    transform.test_consume_operand %2 : !transform.any_op
    transform.yield
  }
}

// -----

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.test_produce_empty_payload : !transform.any_op
    // expected-note @below {{invalidated by this transform op that consumes its operand #0}}
    transform.test_consume_operand %0 : !transform.any_op
    // expected-error @below {{uses a handle associated with empty payload and invalidated by a previously executed transform op}}
    transform.debug.emit_remark_at %0, "remark" : !transform.any_op
    transform.yield
  }
}

// -----

// Make sure we properly report a use-after-consume error when repeated handles
// are allowed in the consuming op. We still want to report handles consumed by
// _previous_ operations, just not by this one. To bypass the quick static check
// of repeated consumption, create a handle to the transform operation and
// invalidate the handle to the root module thus invalidating all other handles.

// expected-note @below {{ancestor payload op}}
module attributes {transform.with_named_sequence}  {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    // expected-note @below {{handle to invalidated ops}}
    // expected-note @below {{nested payload op}}
    %0 = transform.test_produce_self_handle_or_forward_operand : () -> !transform.any_op
    // expected-note @below {{invalidated by this transform op that consumes its operand #0 and invalidates all handles to payload IR entities associated with this operand and entities nested in them}}
    transform.test_consume_operand %arg0 : !transform.any_op
    // expected-error @below {{uses a handle invalidated by a previously executed transform op}}
    transform.test_consume_operand %0 { allow_repeated_handles } : !transform.any_op
    transform.yield
  }
}

// -----

// Re-entering the region should not trigger the consumption error from previous
// execution of the region.

module attributes {transform.with_named_sequence}  {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    transform.test_re_enter_region {
      %0 = transform.test_produce_self_handle_or_forward_operand : () -> !transform.any_op
      transform.test_consume_operand %0 : !transform.any_op
      transform.yield
    }
    transform.yield
  }
}

// -----

// Re-entering the region should not trigger the consumption error from previous
// execution of the region.

module attributes {transform.with_named_sequence}  {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %0 = transform.test_produce_self_handle_or_forward_operand : () -> !transform.any_op
    transform.test_re_enter_region %0 : !transform.any_op {
    ^bb0(%arg1: !transform.any_op):
      transform.test_consume_operand %arg1 : !transform.any_op
      transform.yield
    }
    transform.yield
  }
}

// -----

// Consuming the same handle repeatedly in the region should trigger an error.
module attributes {transform.with_named_sequence}  {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    // expected-note @below {{payload op}}
    // expected-note @below {{handle to invalidated ops}}
    %0 = transform.test_produce_self_handle_or_forward_operand : () -> !transform.any_op
    transform.test_re_enter_region {
      // expected-error @below {{op uses a handle invalidated by a previously executed transform op}}
      // expected-note @below {{invalidated by this transform op}}
      transform.test_consume_operand %0 : !transform.any_op
      transform.yield
    }
    transform.yield
  }
}

// -----

module @named_inclusion_and_consumption attributes { transform.with_named_sequence } {

  transform.named_sequence @foo(%arg0: !transform.any_op {transform.consumed}) -> () {
    // Consuming this handle removes the mapping from the current stack frame
    // mapping and from the caller's stack frame mapping. (If this were not
    // be the case, the "expensive checks" caching mechanism for op names
    // would throw an error saying that an op is mapped but not in the cache.)
    transform.test_consume_operand %arg0 : !transform.any_op
    transform.yield
  }

  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    transform.include @foo failures(propagate) (%arg0) : (!transform.any_op) -> ()
    transform.yield
  }
}
