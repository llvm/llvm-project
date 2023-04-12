// RUN: mlir-opt %s -split-input-file -verify-diagnostics

// expected-error @below {{expects the entry block to have at least one argument}}
transform.sequence failures(propagate) {
}

// -----

// expected-error @below {{expects the first entry block argument to be of type implementing TransformHandleTypeInterface}}
transform.sequence failures(propagate) {
^bb0(%rag0: i64):
}

// -----

// expected-note @below {{nested in another possible top-level op}}
transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  // expected-error @below {{expects operands to be provided for a nested op}}
  transform.sequence failures(propagate) {
  ^bb1(%arg1: !pdl.operation):
  }
}

// -----

// expected-error @below {{'transform.sequence' op expects trailing entry block arguments to be of type implementing TransformHandleTypeInterface, TransformValueHandleTypeInterface or TransformParamTypeInterface}}
// expected-note @below {{argument #1 does not}}
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op, %arg1: i64):
}

// -----

// expected-error @below {{expected children ops to implement TransformOpInterface}}
transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  // expected-note @below {{op without interface}}
  arith.constant 42.0 : f32
}

// -----

// expected-error @below {{expects the types of the terminator operands to match the types of the result}}
%0 = transform.sequence -> !pdl.operation failures(propagate) {
^bb0(%arg0: !pdl.operation):
  // expected-note @below {{terminator}}
  transform.yield
}

// -----

transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op):
  // expected-error @below {{expects the type of the block argument to match the type of the operand}}
  transform.sequence %arg0: !transform.any_op failures(propagate) {
  ^bb1(%arg1: !pdl.operation):
    transform.yield
  }
}

// -----

transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op, %arg1: !transform.any_op, %arg2: !transform.any_op):
  // expected-error @below {{expected types to be provided for all operands}}
  transform.sequence %arg0, %arg1, %arg2 : (!transform.any_op, !transform.any_op) failures(propagate) {
  ^bb0(%arg3: !transform.any_op, %arg4: !transform.any_op, %arg5: !transform.any_op):
  }
}

// -----

%0 = "test.generate_something"() : () -> !transform.any_op
// expected-error @below {{does not expect extra operands when used as top-level}}
"transform.sequence"(%0) ({
^bb0(%arg0: !transform.any_op):
  "transform.yield"() : () -> ()
}) {failure_propagation_mode = 1 : i32, operand_segment_sizes = array<i32: 0, 1>} : (!transform.any_op) -> ()

// -----

// expected-note @below {{nested in another possible top-level op}}
transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  // expected-error @below {{expects operands to be provided for a nested op}}
  transform.sequence failures(propagate) {
  ^bb1(%arg1: !pdl.operation):
  }
}

// -----

// expected-error @below {{expects only one non-pattern op in its body}}
transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  // expected-note @below {{first non-pattern op}}
  transform.sequence failures(propagate) {
  ^bb1(%arg1: !pdl.operation):
  }
  // expected-note @below {{second non-pattern op}}
  transform.sequence failures(propagate) {
  ^bb1(%arg1: !pdl.operation):
  }
}

// -----

// expected-error @below {{expects only pattern and top-level transform ops in its body}}
transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  // expected-note @below {{offending op}}
  "test.something"() : () -> ()
}

// -----

// expected-note @below {{parent operation}}
transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
   // expected-error @below {{op cannot be nested}}
  transform.with_pdl_patterns %arg0 : !pdl.operation {
  ^bb1(%arg1: !pdl.operation):
  }
}

// -----

// expected-error @below {{op expects at least one non-pattern op}}
transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  pdl.pattern @some : benefit(1) {
    %0 = pdl.operation "test.foo"
    pdl.rewrite %0 with "transform.dialect"
  }
}

// -----

transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  // expected-error @below {{op expects at least one non-pattern op}}
  with_pdl_patterns %arg0 : !pdl.operation {
  ^bb1(%arg1: !pdl.operation):
  }
}


// -----

// expected-error @below {{expects at least one region}}
"transform.test_transform_unrestricted_op_no_interface"() : () -> ()

// -----

// expected-error @below {{expects a single-block region}}
"transform.test_transform_unrestricted_op_no_interface"() ({
^bb0(%arg0: !pdl.operation):
  "test.potential_terminator"() : () -> ()
^bb1:
  "test.potential_terminator"() : () -> ()
}) : () -> ()

// -----

transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  // expected-error @below {{result #0 has more than one potential consumer}}
  %0 = test_produce_self_handle_or_forward_operand
  // expected-note @below {{used here as operand #0}}
  test_consume_operand_of_op_kind_or_fail %0, "transform.test_produce_self_handle_or_forward_operand"
  // expected-note @below {{used here as operand #0}}
  test_consume_operand_of_op_kind_or_fail %0, "transform.test_produce_self_handle_or_forward_operand"
}

// -----

transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  // expected-error @below {{result #0 has more than one potential consumer}}
  %0 = test_produce_self_handle_or_forward_operand
  // expected-note @below {{used here as operand #0}}
  test_consume_operand_of_op_kind_or_fail %0, "transform.test_produce_self_handle_or_forward_operand"
  // expected-note @below {{used here as operand #0}}
  transform.sequence %0 : !pdl.operation failures(propagate) {
  ^bb1(%arg1: !pdl.operation):
    test_consume_operand_of_op_kind_or_fail %arg1, "transform.test_produce_self_handle_or_forward_operand"
  }
}

// -----

transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  // expected-error @below {{result #0 has more than one potential consumer}}
  %0 = test_produce_self_handle_or_forward_operand
  // expected-note @below {{used here as operand #0}}
  test_consume_operand_of_op_kind_or_fail %0, "transform.test_produce_self_handle_or_forward_operand"
  transform.sequence %0 : !pdl.operation failures(propagate) {
  ^bb1(%arg1: !pdl.operation):
    // expected-note @below {{used here as operand #0}}
    test_consume_operand_of_op_kind_or_fail %0, "transform.test_produce_self_handle_or_forward_operand"
  }
}

// -----

transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  // expected-error @below {{result #0 has more than one potential consumer}}
  %0 = test_produce_self_handle_or_forward_operand
  // expected-note @below {{used here as operand #0}}
  test_consume_operand_of_op_kind_or_fail %0, "transform.test_produce_self_handle_or_forward_operand"
  // expected-note @below {{used here as operand #0}}
  transform.sequence %0 : !pdl.operation failures(propagate) {
  ^bb1(%arg1: !pdl.operation):
    transform.sequence %arg1 : !pdl.operation failures(propagate) {
    ^bb2(%arg2: !pdl.operation):
      test_consume_operand_of_op_kind_or_fail %arg2, "transform.test_produce_self_handle_or_forward_operand"
    }
  }
}

// -----

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  // expected-error @below {{expects at least one region}}
  transform.alternatives
}

// -----

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  // expected-error @below {{expects terminator operands to have the same type as results of the operation}}
  %2 = transform.alternatives %arg1 : !pdl.operation -> !pdl.operation {
  ^bb2(%arg2: !pdl.operation):
    transform.yield %arg2 : !pdl.operation
  }, {
  ^bb2(%arg2: !pdl.operation):
    // expected-note @below {{terminator}}
    transform.yield
  }
}

// -----

// expected-error @below {{expects the entry block to have at least one argument}}
transform.alternatives {
^bb0:
  transform.yield
}

// -----

transform.sequence failures(propagate) {
^bb0(%arg0: !pdl.operation):
  // expected-error @below {{result #0 has more than one potential consumer}}
  %0 = test_produce_self_handle_or_forward_operand
  // expected-note @below {{used here as operand #0}}
  transform.foreach %0 : !pdl.operation {
  ^bb1(%arg1: !pdl.operation):
    transform.test_consume_operand %arg1 : !pdl.operation
  }
  // expected-note @below {{used here as operand #0}}
  transform.test_consume_operand %0 : !pdl.operation
}

// -----

transform.sequence failures(suppress) {
^bb0(%arg0: !transform.any_op):
  // expected-error @below {{TransformOpInterface requires memory effects on operands to be specified}}
  // expected-note @below {{no effects specified for operand #0}}
  transform.test_required_memory_effects %arg0 {modifies_payload} : (!transform.any_op) -> !transform.any_op
}

// -----

transform.sequence failures(suppress) {
^bb0(%arg0: !transform.any_op):
  // expected-error @below {{TransformOpInterface requires 'allocate' memory effect to be specified for results}}
  // expected-note @below {{no 'allocate' effect specified for result #0}}
  transform.test_required_memory_effects %arg0 {has_operand_effect, modifies_payload} : (!transform.any_op) -> !transform.any_op
}

// -----

// expected-error @below {{attribute can only be attached to operations with symbol tables}}
"test.unknown_container"() { transform.with_named_sequence } : () -> ()

// -----

module attributes { transform.with_named_sequence } {
  // expected-error @below {{expected a non-empty body block}}
  "transform.named_sequence"() ({
  ^bb0:
  }) { sym_name = "external_named_sequence", function_type = () -> () } : () -> ()

  transform.sequence failures(propagate) {
  ^bb0(%arg0: !transform.any_op):
    transform.include @external_named_sequence failures(propagate) () : () -> ()
  }
}

// -----

module attributes { transform.with_named_sequence } {
  // expected-error @below {{recursion not allowed in named sequences}}
  transform.named_sequence @self_recursion() -> () {
    transform.include @self_recursion failures(suppress) () : () -> ()
  }
}

// -----

module @mutual_recursion attributes { transform.with_named_sequence } {
  // expected-note @below {{operation on recursion stack}}  
  transform.named_sequence @foo(%arg0: !transform.any_op) -> () {
    transform.include @bar failures(suppress) (%arg0) : (!transform.any_op) -> ()
    transform.yield
  }

  // expected-error @below {{recursion not allowed in named sequences}}
  transform.named_sequence @bar(%arg0: !transform.any_op) -> () {
    transform.include @foo failures(propagate) (%arg0) : (!transform.any_op) -> ()
    transform.yield
  }
}

// -----

// expected-error @below {{unknown attribute: "transform.unknown_container"}}
module @unknown_attribute attributes { transform.unknown_container } {}

// -----

module {
  transform.sequence failures(suppress) {
  ^bb0(%arg0: !transform.any_op):
    // expected-error @below {{op does not reference a named transform sequence}}
    transform.include @non_existent failures(propagate) () : () -> ()
  }
}

// -----

module attributes { transform.with_named_sequence } {
  transform.sequence failures(suppress) {
  ^bb0(%arg0: !transform.any_op):
    // expected-error @below {{requires attribute 'target'}}
    "transform.include"() {failure_propagation_mode = 0} : () -> ()
  }
}

// -----

module attributes { transform.with_named_sequence } {
  transform.named_sequence @foo(%arg0: !transform.any_op) -> () {
    transform.yield
  }

  transform.sequence failures(suppress) {
  ^bb0(%arg1: !transform.any_op):
    // expected-error @below {{incorrect number of operands for callee}}
    transform.include @foo failures(suppress) () : () -> ()
  }
}

// -----

module attributes { transform.with_named_sequence } {
  transform.named_sequence @foo(%arg0: !transform.any_op) -> () {
    transform.yield
  }

  transform.sequence failures(suppress) {
  ^bb0(%arg1: !transform.op<"builtin.module">):
    // expected-error @below {{operand type mismatch: expected operand type '!transform.any_op', but provided '!transform.op<"builtin.module">' for operand number 0}}
    transform.include @foo failures(suppress) (%arg1) : (!transform.op<"builtin.module">) -> ()
  }
}

// -----

module attributes { transform.with_named_sequence } {
  transform.named_sequence @foo(%arg0: !transform.any_op) -> (!transform.any_op) {
    transform.yield %arg0 : !transform.any_op
  }

  transform.sequence failures(suppress) {
  ^bb0(%arg1: !transform.any_op):
    // expected-error @below {{incorrect number of results for callee}}
    transform.include @foo failures(suppress) (%arg1) : (!transform.any_op) -> ()
  }
}

// -----

module attributes { transform.with_named_sequence } {
  transform.named_sequence @foo(%arg0: !transform.any_op) -> (!transform.any_op) {
    transform.yield %arg0 : !transform.any_op
  }

  transform.sequence failures(suppress) {
  ^bb0(%arg1: !transform.any_op):
    // expected-error @below {{type of result #0 must implement the same transform dialect interface as the corresponding callee result}}
    transform.include @foo failures(suppress) (%arg1) : (!transform.any_op) -> (!transform.any_value)
  }
}

// -----

// expected-note @below {{symbol table operation}}
module {
  // expected-error @below {{expects the parent symbol table to have the 'transform.with_named_sequence' attribute}}
  transform.named_sequence @parent_has_no_attributes() {
    transform.yield
  }
}

// -----

module attributes { transform.with_named_sequence} {
  // expected-note @below {{ancestor transform op}}
  transform.sequence failures(suppress) {
  ^bb0(%arg0: !transform.any_op):
    // expected-error @below {{cannot be defined inside another transform op}}
    transform.named_sequence @nested() {
      transform.yield
    }
  }
}

// -----

module attributes { transform.with_named_sequence} {
  func.func private @foo()

  // expected-error @below {{expected 'transform.yield' as terminator}}
  transform.named_sequence @nested() {
    // expected-note @below {{terminator}}
    func.call @foo() : () -> ()
  }
}


// -----

module attributes { transform.with_named_sequence} {
  func.func private @foo()

  transform.named_sequence @nested(%arg0: !transform.any_op) {
    // expected-error @below {{expected terminator to have as many operands as the parent op has results}}
    transform.yield %arg0 : !transform.any_op
  }
}

// -----

module attributes { transform.with_named_sequence} {
  func.func private @foo()

  transform.named_sequence @nested(%arg0: !transform.any_op) -> !transform.op<"builtin.module"> {
    // expected-error @below {{the type of the terminator operand #0 must match the type of the corresponding parent op result}}
    transform.yield %arg0 : !transform.any_op
  }
}

// -----

module attributes { transform.with_named_sequence } {
  // expected-error @below {{must provide consumed/readonly status for arguments of external or called ops}}
  transform.named_sequence @foo(%op: !transform.any_op )
}

// -----

module attributes { transform.with_named_sequence } {
  // expected-error @below {{argument #0 cannot be both readonly and consumed}}
  transform.named_sequence @foo(%op: !transform.any_op { transform.readonly, transform.consumed } )
}

// -----

module attributes { transform.with_named_sequence } {
  // expected-error @below {{must provide consumed/readonly status for arguments of external or called ops}}
  transform.named_sequence @foo(%op: !transform.any_op) {
    transform.test_print_remark_at_operand %op, "message" : !transform.any_op
    transform.yield
  }

  transform.sequence failures(propagate) {
  ^bb0(%arg0: !transform.any_op):
    transform.include @foo failures(propagate) (%arg0) : (!transform.any_op) -> ()
    transform.yield
  }
}

// -----

module attributes { transform.with_named_sequence } {
  // expected-error @below {{argument #0 cannot be both readonly and consumed}}
  transform.named_sequence @foo(%op: !transform.any_op {transform.readonly, transform.consumed}) {
    transform.test_print_remark_at_operand %op, "message" : !transform.any_op
    transform.yield
  }

  transform.sequence failures(propagate) {
  ^bb0(%arg0: !transform.any_op):
    transform.include @foo failures(propagate) (%arg0) : (!transform.any_op) -> ()
    transform.yield
  }
}

// -----

module attributes { transform.with_named_sequence } {
  // expected-warning @below {{argument #0 is not consumed in the body but is marked as consume}}
  transform.named_sequence @foo(%op: !transform.any_op {transform.consumed}) {
    transform.test_print_remark_at_operand %op, "message" : !transform.any_op
    transform.yield
  }

  transform.sequence failures(propagate) {
  ^bb0(%arg0: !transform.any_op):
    transform.include @foo failures(propagate) (%arg0) : (!transform.any_op) -> ()
    transform.yield
  }
}

// -----

module attributes { transform.with_named_sequence } {
  // expected-error @below {{argument #0 is consumed in the body but is not marked as such}}
  transform.named_sequence @foo(%op: !transform.any_op {transform.readonly}) {
    transform.test_consume_operand %op : !transform.any_op
    transform.yield
  }

  transform.sequence failures(propagate) {
  ^bb0(%arg0: !transform.any_op):
    transform.include @foo failures(propagate) (%arg0) : (!transform.any_op) -> ()
    transform.yield
  }
}

// -----

// Checking that consumptions annotations are used correctly in invocation checks.
module attributes { transform.with_named_sequence } {
  transform.named_sequence @foo(%op: !transform.any_op { transform.consumed } )

  // expected-error @below {{'transform.sequence' op block argument #0 has more than one potential consumer}}
  transform.sequence failures(propagate) {
  ^bb0(%arg0: !transform.any_op):
    // expected-note @below {{used here as operand #0}}
    transform.include @foo failures(propagate) (%arg0) : (!transform.any_op) -> ()
    // expected-note @below {{used here as operand #0}}
    transform.include @foo failures(propagate) (%arg0) : (!transform.any_op) -> ()
    transform.yield
  }
}

// -----

module attributes { transform.with_named_sequence } {
  transform.sequence failures(propagate) {
  ^bb0(%root: !transform.any_op):
    // expected-error @below {{unresolved matcher symbol @foo}}
    transform.foreach_match in %root
      @foo -> @bar : (!transform.any_op) -> !transform.any_op
  }
}

// -----

module attributes { transform.with_named_sequence } {
  func.func private @foo()

  transform.sequence failures(propagate) {
  ^bb0(%root: !transform.any_op):
    // expected-error @below {{unresolved matcher symbol @foo}}
    transform.foreach_match in %root
      @foo -> @bar : (!transform.any_op) -> !transform.any_op
  }
}

// -----

module attributes { transform.with_named_sequence } {
  transform.named_sequence @match()

  transform.sequence failures(propagate) {
  ^bb0(%root: !transform.any_op):
    // expected-error @below {{unresolved action symbol @bar}}
    transform.foreach_match in %root
      @match -> @bar : (!transform.any_op) -> !transform.any_op
  }
}

// -----

module attributes { transform.with_named_sequence } {
  func.func private @bar()
  transform.named_sequence @match()

  transform.sequence failures(propagate) {
  ^bb0(%root: !transform.any_op):
    // expected-error @below {{unresolved action symbol @bar}}
    transform.foreach_match in %root
      @match -> @bar : (!transform.any_op) -> !transform.any_op
  }
}

// -----

module attributes { transform.with_named_sequence } {
  transform.named_sequence @match() -> !transform.any_op
  transform.named_sequence @action()

  transform.sequence failures(propagate) {
  ^bb0(%root: !transform.any_op):
    // expected-error @below {{mismatching number of matcher results and action arguments between @match (1) and @action (0)}}
    transform.foreach_match in %root
      @match -> @action : (!transform.any_op) -> !transform.any_op
  }
}

// -----

module attributes { transform.with_named_sequence } {
  transform.named_sequence @match(!transform.any_op {transform.readonly})
  // expected-note @below {{symbol declaration}}
  transform.named_sequence @action() -> !transform.any_op

  transform.sequence failures(propagate) {
  ^bb0(%root: !transform.any_op):
    // expected-error @below {{action symbol is not expected to have results}}
    transform.foreach_match in %root
      @match -> @action : (!transform.any_op) -> !transform.any_op
  }
}

// -----

module attributes { transform.with_named_sequence } {
  // expected-note @below {{symbol declaration}}
  transform.named_sequence @match()
  transform.named_sequence @action()

  transform.sequence failures(propagate) {
  ^bb0(%root: !transform.any_op):
    // expected-error @below {{expects matcher symbol to have one argument with the same transform interface as the first operand}}
    transform.foreach_match in %root
      @match -> @action : (!transform.any_op) -> !transform.any_op
  }
}

// -----

module attributes { transform.with_named_sequence } {
  // expected-note @below {{symbol declaration}}
  transform.named_sequence @match(!transform.any_op {transform.consumed})
  transform.named_sequence @action()

  transform.sequence failures(propagate) {
  ^bb0(%root: !transform.any_op):
    // expected-error @below {{'transform.foreach_match' op does not expect matcher symbol to consume its operand}}
    transform.foreach_match in %root
      @match -> @action : (!transform.any_op) -> !transform.any_op
  }
}
