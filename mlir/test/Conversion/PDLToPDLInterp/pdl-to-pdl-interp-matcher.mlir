// RUN: mlir-opt -split-input-file -convert-pdl-to-pdl-interp %s | FileCheck %s

// CHECK-LABEL: module @empty_module
module @empty_module {
// CHECK: func @matcher(%{{.*}}: !pdl.operation)
// CHECK-NEXT: pdl_interp.finalize
}

// -----

// CHECK-LABEL: module @simple
module @simple {
  // CHECK: func @matcher(%[[ROOT:.*]]: !pdl.operation)
  // CHECK:   pdl_interp.check_operation_name of %[[ROOT]] is "foo.op" -> ^bb2, ^bb1
  // CHECK: ^bb1:
  // CHECK:   pdl_interp.finalize
  // CHECK: ^bb2:
  // CHECK:   pdl_interp.check_operand_count of %[[ROOT]] is 0 -> ^bb3, ^bb1
  // CHECK: ^bb3:
  // CHECK:   pdl_interp.check_result_count of %[[ROOT]] is 0 -> ^bb4, ^bb1
  // CHECK: ^bb4:
  // CHECK:   pdl_interp.record_match @rewriters::@pdl_generated_rewriter
  // CHECK-SAME: benefit(1), loc([%[[ROOT]]]), root("foo.op") -> ^bb1

  // CHECK: module @rewriters
  // CHECK:   func @pdl_generated_rewriter(%[[REWRITE_ROOT:.*]]: !pdl.operation)
  // CHECK:     pdl_interp.apply_rewrite "rewriter"(%[[REWRITE_ROOT]]
  // CHECK:     pdl_interp.finalize
  pdl.pattern : benefit(1) {
    %root = operation "foo.op"
    rewrite %root with "rewriter"
  }
}

// -----

// CHECK-LABEL: module @attributes
module @attributes {
  // CHECK: func @matcher(%[[ROOT:.*]]: !pdl.operation)
  // Check the value of "attr".
  // CHECK-DAG:   %[[ATTR:.*]] = pdl_interp.get_attribute "attr" of %[[ROOT]]
  // CHECK-DAG:   pdl_interp.is_not_null %[[ATTR]] : !pdl.attribute
  // CHECK-DAG:   pdl_interp.check_attribute %[[ATTR]] is 10 : i64

  // Check the type of "attr1".
  // CHECK-DAG:   %[[ATTR1:.*]] = pdl_interp.get_attribute "attr1" of %[[ROOT]]
  // CHECK-DAG:   pdl_interp.is_not_null %[[ATTR1]] : !pdl.attribute
  // CHECK-DAG:   %[[ATTR1_TYPE:.*]] = pdl_interp.get_attribute_type of %[[ATTR1]]
  // CHECK-DAG:   pdl_interp.check_type %[[ATTR1_TYPE]] is i64
  pdl.pattern : benefit(1) {
    %type = type : i64
    %attr = attribute = 10 : i64
    %attr1 = attribute : %type
    %root = operation {"attr" = %attr, "attr1" = %attr1}
    rewrite %root with "rewriter"
  }
}

// -----

// CHECK-LABEL: module @constraints
module @constraints {
  // CHECK: func @matcher(%[[ROOT:.*]]: !pdl.operation)
  // CHECK-DAG:   %[[INPUT:.*]] = pdl_interp.get_operand 0 of %[[ROOT]]
  // CHECK-DAG:   %[[INPUT1:.*]] = pdl_interp.get_operand 1 of %[[ROOT]]
  // CHECK-DAG:   %[[RESULT:.*]] = pdl_interp.get_result 0 of %[[ROOT]]
  // CHECK:       pdl_interp.apply_constraint "multi_constraint"(%[[INPUT]], %[[INPUT1]], %[[RESULT]]

  pdl.pattern : benefit(1) {
    %input0 = operand
    %input1 = operand
    %root = operation(%input0, %input1 : !pdl.value, !pdl.value)
    %result0 = result 0 of %root

    pdl.apply_native_constraint "multi_constraint"(%input0, %input1, %result0 : !pdl.value, !pdl.value, !pdl.value)
    rewrite %root with "rewriter"
  }
}

// -----

// CHECK-LABEL: module @constraint_with_result
module @constraint_with_result {
  // CHECK: func @matcher(%[[ROOT:.*]]: !pdl.operation)
  // CHECK: %[[ATTR:.*]] = pdl_interp.apply_constraint "check_op_and_get_attr_constr"(%[[ROOT]]
  // CHECK: pdl_interp.record_match @rewriters::@pdl_generated_rewriter(%[[ROOT]], %[[ATTR]] : !pdl.operation, !pdl.attribute)
  pdl.pattern : benefit(1) {
    %root = operation
    %attr = pdl.apply_native_constraint "check_op_and_get_attr_constr"(%root : !pdl.operation) : !pdl.attribute
    rewrite %root with "rewriter"(%attr : !pdl.attribute)
  }
}

// -----

// CHECK-LABEL: module @constraint_with_unused_result
module @constraint_with_unused_result {
  // CHECK: func @matcher(%[[ROOT:.*]]: !pdl.operation)
  // CHECK: %[[ATTR:.*]] = pdl_interp.apply_constraint "check_op_and_get_attr_constr"(%[[ROOT]]
  // CHECK: pdl_interp.record_match @rewriters::@pdl_generated_rewriter(%[[ROOT]] : !pdl.operation)
  pdl.pattern : benefit(1) {
    %root = operation
    %attr = pdl.apply_native_constraint "check_op_and_get_attr_constr"(%root : !pdl.operation) : !pdl.attribute
    rewrite %root with "rewriter"
  }
}

// -----

// CHECK-LABEL: module @constraint_with_result_multiple
module @constraint_with_result_multiple {
  // check that native constraints work as expected even when multiple identical constraints are fused

  // CHECK: func @matcher(%[[ROOT:.*]]: !pdl.operation)
  // CHECK: %[[ATTR:.*]] = pdl_interp.apply_constraint "check_op_and_get_attr_constr"(%[[ROOT]]
  // CHECK-NOT: pdl_interp.apply_constraint "check_op_and_get_attr_constr"
  // CHECK: pdl_interp.record_match @rewriters::@pdl_generated_rewriter_0(%[[ROOT]], %[[ATTR]]  : !pdl.operation, !pdl.attribute)
  // CHECK: pdl_interp.record_match @rewriters::@pdl_generated_rewriter(%[[ROOT]], %[[ATTR]] : !pdl.operation, !pdl.attribute)
  pdl.pattern : benefit(1) {
    %root = operation
    %attr = pdl.apply_native_constraint "check_op_and_get_attr_constr"(%root : !pdl.operation) : !pdl.attribute
    rewrite %root with "rewriter"(%attr : !pdl.attribute)
  }
  pdl.pattern : benefit(1) {
    %root = operation
    %attr = pdl.apply_native_constraint "check_op_and_get_attr_constr"(%root : !pdl.operation) : !pdl.attribute
    rewrite %root with "rewriter"(%attr : !pdl.attribute)
  }
}

// -----

// CHECK-LABEL: module @negated_constraint
module @negated_constraint {
  // CHECK: func @matcher(%[[ROOT:.*]]: !pdl.operation)
  // CHECK: pdl_interp.apply_constraint "constraint"(%[[ROOT]] : !pdl.operation) {isNegated = true}
  // CHECK: pdl_interp.record_match @rewriters::@pdl_generated_rewriter(%[[ROOT]] : !pdl.operation)
  pdl.pattern : benefit(1) {
    %root = operation
    pdl.apply_native_constraint "constraint"(%root : !pdl.operation) {isNegated = true}
    rewrite %root with "rewriter"
  }
}

// -----

// CHECK-LABEL: module @inputs
module @inputs {
  // CHECK: func @matcher(%[[ROOT:.*]]: !pdl.operation)
  // CHECK-DAG: pdl_interp.check_operand_count of %[[ROOT]] is 2

  // Get the input and check the type.
  // CHECK-DAG:   %[[INPUT:.*]] = pdl_interp.get_operand 0 of %[[ROOT]]
  // CHECK-DAG:   pdl_interp.is_not_null %[[INPUT]] : !pdl.value
  // CHECK-DAG:   %[[INPUT_TYPE:.*]] = pdl_interp.get_value_type of %[[INPUT]]
  // CHECK-DAG:   pdl_interp.check_type %[[INPUT_TYPE]] is i64

  // Get the second operand and check that it is equal to the first.
  // CHECK-DAG:  %[[INPUT1:.*]] = pdl_interp.get_operand 1 of %[[ROOT]]
  // CHECK-DAG:  pdl_interp.are_equal %[[INPUT]], %[[INPUT1]] : !pdl.value
  pdl.pattern : benefit(1) {
    %type = type : i64
    %input = operand : %type
    %root = operation(%input, %input : !pdl.value, !pdl.value)
    rewrite %root with "rewriter"
  }
}

// -----

// CHECK-LABEL: module @variadic_inputs
module @variadic_inputs {
  // CHECK: func @matcher(%[[ROOT:.*]]: !pdl.operation)
  // CHECK-DAG: pdl_interp.check_operand_count of %[[ROOT]] is at_least 2

  // The first operand has a known index.
  // CHECK-DAG:   %[[INPUT:.*]] = pdl_interp.get_operand 0 of %[[ROOT]]
  // CHECK-DAG:   pdl_interp.is_not_null %[[INPUT]] : !pdl.value

  // The second operand is a group of unknown size, with a type constraint.
  // CHECK-DAG:   %[[VAR_INPUTS:.*]] = pdl_interp.get_operands 1 of %[[ROOT]] : !pdl.range<value>
  // CHECK-DAG:   pdl_interp.is_not_null %[[VAR_INPUTS]] : !pdl.range<value>

  // CHECK-DAG:   %[[INPUT_TYPE:.*]] = pdl_interp.get_value_type of %[[VAR_INPUTS]] : !pdl.range<type>
  // CHECK-DAG:   pdl_interp.check_types %[[INPUT_TYPE]] are [i64]

  // The third operand is at an unknown offset due to operand 2, but is expected
  // to be of size 1.
  // CHECK-DAG:  %[[INPUT2:.*]] = pdl_interp.get_operands 2 of %[[ROOT]] : !pdl.value
  // CHECK-DAG:  pdl_interp.are_equal %[[INPUT]], %[[INPUT2]] : !pdl.value
  pdl.pattern : benefit(1) {
    %types = types : [i64]
    %inputs = operands : %types
    %input = operand
    %root = operation(%input, %inputs, %input : !pdl.value, !pdl.range<value>, !pdl.value)
    rewrite %root with "rewriter"
  }
}

// -----

// CHECK-LABEL: module @single_operand_range
module @single_operand_range {
  // CHECK: func @matcher(%[[ROOT:.*]]: !pdl.operation)

  // Check that the operand range is treated as all of the operands of the
  // operation.
  // CHECK-DAG:   %[[RESULTS:.*]] = pdl_interp.get_operands of %[[ROOT]]
  // CHECK-DAG:   %[[RESULT_TYPES:.*]] = pdl_interp.get_value_type of %[[RESULTS]] : !pdl.range<type>
  // CHECK-DAG:   pdl_interp.check_types %[[RESULT_TYPES]] are [i64]

  // The operand count is unknown, so there is no need to check for it.
  // CHECK-NOT: pdl_interp.check_operand_count
  pdl.pattern : benefit(1) {
    %types = types : [i64]
    %operands = operands : %types
    %root = operation(%operands : !pdl.range<value>)
    rewrite %root with "rewriter"
  }
}

// -----

// CHECK-LABEL: module @results
module @results {
  // CHECK: func @matcher(%[[ROOT:.*]]: !pdl.operation)
  // CHECK:   pdl_interp.check_result_count of %[[ROOT]] is 2

  // Get the result and check the type.
  // CHECK-DAG:   %[[RESULT:.*]] = pdl_interp.get_result 0 of %[[ROOT]]
  // CHECK-DAG:   pdl_interp.is_not_null %[[RESULT]] : !pdl.value
  // CHECK-DAG:   %[[RESULT_TYPE:.*]] = pdl_interp.get_value_type of %[[RESULT]]
  // CHECK-DAG:   pdl_interp.check_type %[[RESULT_TYPE]] is i32

  // The second result doesn't have any constraints, so we don't generate an
  // access for it.
  // CHECK-NOT:   pdl_interp.get_result 1 of %[[ROOT]]
  pdl.pattern : benefit(1) {
    %type1 = type : i32
    %type2 = type
    %root = operation -> (%type1, %type2 : !pdl.type, !pdl.type)
    rewrite %root with "rewriter"
  }
}

// -----

// CHECK-LABEL: module @variadic_results
module @variadic_results {
  // CHECK: func @matcher(%[[ROOT:.*]]: !pdl.operation)
  // CHECK-DAG: pdl_interp.check_result_count of %[[ROOT]] is at_least 2

  // The first result has a known index.
  // CHECK-DAG:   %[[RESULT:.*]] = pdl_interp.get_result 0 of %[[ROOT]]
  // CHECK-DAG:   pdl_interp.is_not_null %[[RESULT]] : !pdl.value

  // The second result is a group of unknown size, with a type constraint.
  // CHECK-DAG:   %[[VAR_RESULTS:.*]] = pdl_interp.get_results 1 of %[[ROOT]] : !pdl.range<value>
  // CHECK-DAG:   pdl_interp.is_not_null %[[VAR_RESULTS]] : !pdl.range<value>

  // CHECK-DAG:   %[[RESULT_TYPE:.*]] = pdl_interp.get_value_type of %[[VAR_RESULTS]] : !pdl.range<type>
  // CHECK-DAG:   pdl_interp.check_types %[[RESULT_TYPE]] are [i64]

  // The third result is at an unknown offset due to result 1, but is expected
  // to be of size 1.
  // CHECK-DAG:  %[[RESULT2:.*]] = pdl_interp.get_results 2 of %[[ROOT]] : !pdl.value
  // CHECK-DAG:   pdl_interp.is_not_null %[[RESULT2]] : !pdl.value
  pdl.pattern : benefit(1) {
    %types = types : [i64]
    %type = type
    %root = operation -> (%type, %types, %type : !pdl.type, !pdl.range<type>, !pdl.type)
    rewrite %root with "rewriter"
  }
}

// -----

// CHECK-LABEL: module @single_result_range
module @single_result_range {
  // CHECK: func @matcher(%[[ROOT:.*]]: !pdl.operation)

  // Check that the result range is treated as all of the results of the
  // operation.
  // CHECK-DAG:   %[[RESULTS:.*]] = pdl_interp.get_results of %[[ROOT]]
  // CHECK-DAG:   %[[RESULT_TYPES:.*]] = pdl_interp.get_value_type of %[[RESULTS]] : !pdl.range<type>
  // CHECK-DAG:   pdl_interp.check_types %[[RESULT_TYPES]] are [i64]

  // The result count is unknown, so there is no need to check for it.
  // CHECK-NOT: pdl_interp.check_result_count
  pdl.pattern : benefit(1) {
    %types = types : [i64]
    %root = operation -> (%types : !pdl.range<type>)
    rewrite %root with "rewriter"
  }
}

// -----

// CHECK-LABEL: module @results_as_operands
module @results_as_operands {
  // CHECK: func @matcher(%[[ROOT:.*]]: !pdl.operation)

  // Get the first result and check it matches the first operand.
  // CHECK-DAG:   %[[OPERAND_0:.*]] = pdl_interp.get_operand 0 of %[[ROOT]]
  // CHECK-DAG:   %[[DEF_OP_0:.*]] = pdl_interp.get_defining_op of %[[OPERAND_0]]
  // CHECK-DAG:   %[[RESULT_0:.*]] = pdl_interp.get_result 0 of %[[DEF_OP_0]]
  // CHECK-DAG:   pdl_interp.are_equal %[[RESULT_0]], %[[OPERAND_0]]

  // Get the second result and check it matches the second operand.
  // CHECK-DAG:   %[[OPERAND_1:.*]] = pdl_interp.get_operand 1 of %[[ROOT]]
  // CHECK-DAG:   %[[DEF_OP_1:.*]] = pdl_interp.get_defining_op of %[[OPERAND_1]]
  // CHECK-DAG:   %[[RESULT_1:.*]] = pdl_interp.get_result 1 of %[[DEF_OP_1]]
  // CHECK-DAG:   pdl_interp.are_equal %[[RESULT_1]], %[[OPERAND_1]]

  // Check that the parent operation of both results is the same.
  // CHECK-DAG:   pdl_interp.are_equal %[[DEF_OP_0]], %[[DEF_OP_1]]

  pdl.pattern : benefit(1) {
    %type1 = type : i32
    %type2 = type
    %inputOp = operation -> (%type1, %type2 : !pdl.type, !pdl.type)
    %result1 = result 0 of %inputOp
    %result2 = result 1 of %inputOp

    %root = operation(%result1, %result2 : !pdl.value, !pdl.value)
    rewrite %root with "rewriter"
  }
}

// -----

// CHECK-LABEL: module @single_result_range_as_operands
module @single_result_range_as_operands {
  // CHECK: func @matcher(%[[ROOT:.*]]: !pdl.operation)
  // CHECK-DAG:  %[[OPERANDS:.*]] = pdl_interp.get_operands of %[[ROOT]] : !pdl.range<value>
  // CHECK-DAG:  %[[OP:.*]] = pdl_interp.get_defining_op of %[[OPERANDS]] : !pdl.range<value>
  // CHECK-DAG:  pdl_interp.is_not_null %[[OP]]
  // CHECK-DAG:  %[[RESULTS:.*]] = pdl_interp.get_results of %[[OP]] : !pdl.range<value>
  // CHECK-DAG:  pdl_interp.are_equal %[[RESULTS]], %[[OPERANDS]] : !pdl.range<value>

  pdl.pattern : benefit(1) {
    %types = types
    %inputOp = operation -> (%types : !pdl.range<type>)
    %results = results of %inputOp

    %root = operation(%results : !pdl.range<value>)
    rewrite %root with "rewriter"
  }
}

// -----

// CHECK-LABEL: module @switch_single_result_type
module @switch_single_result_type {
  // CHECK: func @matcher(%[[ROOT:.*]]: !pdl.operation)
  // CHECK:   %[[RESULT:.*]] = pdl_interp.get_result 0 of %[[ROOT]]
  // CHECK:   %[[RESULT_TYPE:.*]] = pdl_interp.get_value_type of %[[RESULT]]
  // CHECK:   pdl_interp.switch_type %[[RESULT_TYPE]] to [i32, i64]
  pdl.pattern : benefit(1) {
    %type = type : i32
    %root = operation -> (%type : !pdl.type)
    rewrite %root with "rewriter"
  }
  pdl.pattern : benefit(1) {
    %type = type : i64
    %root = operation -> (%type : !pdl.type)
    rewrite %root with "rewriter"
  }
}

// -----

// CHECK-LABEL: module @switch_result_types
module @switch_result_types {
  // CHECK: func @matcher(%[[ROOT:.*]]: !pdl.operation)
  // CHECK:   %[[RESULTS:.*]] = pdl_interp.get_results of %[[ROOT]]
  // CHECK:   %[[RESULT_TYPES:.*]] = pdl_interp.get_value_type of %[[RESULTS]]
  // CHECK:   pdl_interp.switch_types %[[RESULT_TYPES]] to {{\[\[}}i32], [i64, i32]]
  pdl.pattern : benefit(1) {
    %types = types : [i32]
    %root = operation -> (%types : !pdl.range<type>)
    rewrite %root with "rewriter"
  }
  pdl.pattern : benefit(1) {
    %types = types : [i64, i32]
    %root = operation -> (%types : !pdl.range<type>)
    rewrite %root with "rewriter"
  }
}

// -----

// CHECK-LABEL: module @switch_operand_count_at_least
module @switch_operand_count_at_least {
  // Check that when there are multiple "at_least" checks, the failure branch
  // goes to the next one in increasing order.

  // CHECK: func @matcher(%[[ROOT:.*]]: !pdl.operation)
  // CHECK: pdl_interp.check_operand_count of %[[ROOT]] is at_least 1 -> ^[[PATTERN_1_NEXT_BLOCK:.*]],
  // CHECK: ^bb2:
  // CHECK-NEXT: pdl_interp.check_operand_count of %[[ROOT]] is at_least 2
  // CHECK: ^[[PATTERN_1_NEXT_BLOCK]]:
  // CHECK-NEXT: {{.*}} -> ^{{.*}}, ^bb2
  pdl.pattern : benefit(1) {
    %operand = operand
    %operands = operands
    %root = operation(%operand, %operands : !pdl.value, !pdl.range<value>)
    rewrite %root with "rewriter"
  }
  pdl.pattern : benefit(1) {
    %operand = operand
    %operand2 = operand
    %operands = operands
    %root = operation(%operand, %operand2, %operands : !pdl.value, !pdl.value, !pdl.range<value>)
    rewrite %root with "rewriter"
  }
}

// -----

// CHECK-LABEL: module @switch_result_count_at_least
module @switch_result_count_at_least {
  // Check that when there are multiple "at_least" checks, the failure branch
  // goes to the next one in increasing order.

  // CHECK: func @matcher(%[[ROOT:.*]]: !pdl.operation)
  // CHECK: pdl_interp.check_result_count of %[[ROOT]] is at_least 1 -> ^[[PATTERN_1_NEXT_BLOCK:.*]],
  // CHECK: ^[[PATTERN_2_BLOCK:[a-zA-Z_0-9]*]]:
  // CHECK: pdl_interp.check_result_count of %[[ROOT]] is at_least 2
  // CHECK: ^[[PATTERN_1_NEXT_BLOCK]]:
  // CHECK-NEXT: pdl_interp.get_result
  // CHECK-NEXT: pdl_interp.is_not_null {{.*}} -> ^{{.*}}, ^[[PATTERN_2_BLOCK]]
  pdl.pattern : benefit(1) {
    %type = type
    %types = types
    %root = operation -> (%type, %types : !pdl.type, !pdl.range<type>)
    rewrite %root with "rewriter"
  }
  pdl.pattern : benefit(1) {
    %type = type
    %type2 = type
    %types = types
    %root = operation -> (%type, %type2, %types : !pdl.type, !pdl.type, !pdl.range<type>)
    rewrite %root with "rewriter"
  }
}


// -----

// CHECK-LABEL: module @predicate_ordering
module @predicate_ordering {
  // Check that the result is checked for null first, before applying the
  // constraint. The null check is prevalent in both patterns, so should be
  // prioritized first.

  // CHECK: func @matcher(%[[ROOT:.*]]: !pdl.operation)
  // CHECK:   %[[RESULT:.*]] = pdl_interp.get_result 0 of %[[ROOT]]
  // CHECK-NEXT: pdl_interp.is_not_null %[[RESULT]]
  // CHECK:   %[[RESULT_TYPE:.*]] = pdl_interp.get_value_type of %[[RESULT]]
  // CHECK: pdl_interp.apply_constraint "typeConstraint"(%[[RESULT_TYPE]]

  pdl.pattern : benefit(1) {
    %resultType = type
    pdl.apply_native_constraint "typeConstraint"(%resultType : !pdl.type)
    %root = operation -> (%resultType : !pdl.type)
    rewrite %root with "rewriter"
  }

  pdl.pattern : benefit(1) {
    %resultType = type
    %apply = operation -> (%resultType : !pdl.type)
    rewrite %apply with "rewriter"
  }
}


// -----

// CHECK-LABEL: module @multi_root
module @multi_root {
  // Check the lowering of a simple two-root pattern.
  // This checks that we correctly generate the pdl_interp.choose_op operation
  // and tie the break between %root1 and %root2 in favor of %root1.

  // CHECK: func @matcher(%[[ROOT1:.*]]: !pdl.operation)
  // CHECK-DAG: %[[VAL1:.*]] = pdl_interp.get_operand 0 of %[[ROOT1]]
  // CHECK-DAG: %[[OP1:.*]] = pdl_interp.get_defining_op of %[[VAL1]]
  // CHECK-DAG: %[[OPS:.*]] = pdl_interp.get_users of %[[VAL1]] : !pdl.value
  // CHECK-DAG: pdl_interp.foreach %[[ROOT2:.*]] : !pdl.operation in %[[OPS]]
  // CHECK-DAG:   %[[OPERANDS:.*]] = pdl_interp.get_operand 0 of %[[ROOT2]]
  // CHECK-DAG:   pdl_interp.are_equal %[[OPERANDS]], %[[VAL1]] : !pdl.value -> ^{{.*}}, ^[[CONTINUE:.*]]
  // CHECK-DAG:   pdl_interp.continue
  // CHECK-DAG:   %[[VAL2:.*]] = pdl_interp.get_operand 1 of %[[ROOT2]]
  // CHECK-DAG:   %[[OP2:.*]] = pdl_interp.get_defining_op of %[[VAL2]]
  // CHECK-DAG:   pdl_interp.is_not_null %[[OP1]] : !pdl.operation -> ^{{.*}}, ^[[CONTINUE]]
  // CHECK-DAG:   pdl_interp.is_not_null %[[OP2]] : !pdl.operation
  // CHECK-DAG:   pdl_interp.is_not_null %[[VAL1]] : !pdl.value
  // CHECK-DAG:   pdl_interp.is_not_null %[[VAL2]] : !pdl.value
  // CHECK-DAG:   pdl_interp.is_not_null %[[ROOT2]] : !pdl.operation

  pdl.pattern @rewrite_multi_root : benefit(1) {
    %input1 = operand
    %input2 = operand
    %type = type
    %op1 = operation(%input1 : !pdl.value) -> (%type : !pdl.type)
    %val1 = result 0 of %op1
    %root1 = operation(%val1 : !pdl.value)
    %op2 = operation(%input2 : !pdl.value) -> (%type : !pdl.type)
    %val2 = result 0 of %op2
    %root2 = operation(%val1, %val2 : !pdl.value, !pdl.value)
    rewrite %root1 with "rewriter"(%root2 : !pdl.operation)
  }
}


// -----

// CHECK-LABEL: module @overlapping_roots
module @overlapping_roots {
  // Check the lowering of a degenerate two-root pattern, where one root
  // is in the subtree rooted at another.

  // CHECK: func @matcher(%[[ROOT:.*]]: !pdl.operation)
  // CHECK-DAG: %[[VAL:.*]] = pdl_interp.get_operand 0 of %[[ROOT]]
  // CHECK-DAG: %[[OP:.*]] = pdl_interp.get_defining_op of %[[VAL]]
  // CHECK-DAG: %[[INPUT1:.*]] = pdl_interp.get_operand 0 of %[[OP]]
  // CHECK-DAG: %[[INPUT2:.*]] = pdl_interp.get_operand 1 of %[[OP]]
  // CHECK-DAG: pdl_interp.is_not_null %[[VAL]] : !pdl.value
  // CHECK-DAG: pdl_interp.is_not_null %[[OP]] : !pdl.operation
  // CHECK-DAG: pdl_interp.is_not_null %[[INPUT1]] : !pdl.value
  // CHECK-DAG: pdl_interp.is_not_null %[[INPUT2]] : !pdl.value

  pdl.pattern @rewrite_overlapping_roots : benefit(1) {
    %input1 = operand
    %input2 = operand
    %type = type
    %op = operation(%input1, %input2 : !pdl.value, !pdl.value) -> (%type : !pdl.type)
    %val = result 0 of %op
    %root = operation(%val : !pdl.value)
    rewrite with "rewriter"(%root : !pdl.operation)
  }
}

// -----

// CHECK-LABEL: module @force_overlapped_root
module @force_overlapped_root {
  // Check the lowering of a degenerate two-root pattern, where one root
  // is in the subtree rooted at another, and we are forced to use this
  // root as the root of the search tree.

  // CHECK: func @matcher(%[[ROOT:.*]]: !pdl.operation)
  // CHECK-DAG: %[[VAL:.*]] = pdl_interp.get_result 0 of %[[ROOT]]
  // CHECK-DAG: pdl_interp.check_operand_count of %[[ROOT]] is 2
  // CHECK-DAG: pdl_interp.check_result_count of %[[ROOT]] is 1
  // CHECK-DAG: %[[INPUT2:.*]] = pdl_interp.get_operand 1 of %[[ROOT]]
  // CHECK-DAG: pdl_interp.is_not_null %[[INPUT2]] : !pdl.value
  // CHECK-DAG: %[[INPUT1:.*]] = pdl_interp.get_operand 0 of %[[ROOT]]
  // CHECK-DAG: pdl_interp.is_not_null %[[INPUT1]] : !pdl.value
  // CHECK-DAG: %[[OPS:.*]] = pdl_interp.get_users of %[[VAL]] : !pdl.value
  // CHECK-DAG: pdl_interp.foreach %[[OP:.*]] : !pdl.operation in %[[OPS]]
  // CHECK-DAG:   pdl_interp.is_not_null %[[OP]] : !pdl.operation
  // CHECK-DAG:   pdl_interp.check_operand_count of %[[OP]] is 1

  pdl.pattern @rewrite_forced_overlapped_root : benefit(1) {
    %input1 = operand
    %input2 = operand
    %type = type
    %root = operation(%input1, %input2 : !pdl.value, !pdl.value) -> (%type : !pdl.type)
    %val = result 0 of %root
    %op = operation(%val : !pdl.value)
    rewrite %root with "rewriter"(%op : !pdl.operation)
  }
}

// -----

// CHECK-LABEL: module @variadic_results_all
module @variadic_results_all {
  // Check the correct lowering when using all results of an operation
  // and passing it them as operands to another operation.

  // CHECK: func @matcher(%[[ROOT:.*]]: !pdl.operation)
  // CHECK-DAG: pdl_interp.check_operand_count of %[[ROOT]] is 0
  // CHECK-DAG: %[[VALS:.*]] = pdl_interp.get_results of %[[ROOT]] : !pdl.range<value>
  // CHECK-DAG: %[[VAL0:.*]] = pdl_interp.extract 0 of %[[VALS]]
  // CHECK-DAG: %[[OPS:.*]] = pdl_interp.get_users of %[[VAL0]] : !pdl.value
  // CHECK-DAG: pdl_interp.foreach %[[OP:.*]] : !pdl.operation in %[[OPS]]
  // CHECK-DAG:   %[[OPERANDS:.*]] = pdl_interp.get_operands of %[[OP]]
  // CHECK-DAG:    pdl_interp.are_equal %[[OPERANDS]], %[[VALS]] : !pdl.range<value> -> ^{{.*}}, ^[[CONTINUE:.*]]
  // CHECK-DAG:   pdl_interp.is_not_null %[[OP]]
  // CHECK-DAG:   pdl_interp.check_result_count of %[[OP]] is 0
  pdl.pattern @variadic_results_all : benefit(1) {
    %types = types
    %root = operation -> (%types : !pdl.range<type>)
    %vals = results of %root
    %op = operation(%vals : !pdl.range<value>)
    rewrite %root with "rewriter"(%op : !pdl.operation)
  }
}

// -----

// CHECK-LABEL: module @variadic_results_at
module @variadic_results_at {
  // Check the correct lowering when using selected results of an operation
  // and passing it them as an operand to another operation.

  // CHECK: func @matcher(%[[ROOT1:.*]]: !pdl.operation)
  // CHECK-DAG: %[[VALS:.*]] = pdl_interp.get_operands 0 of %[[ROOT1]] : !pdl.range<value>
  // CHECK-DAG: %[[OP:.*]] = pdl_interp.get_defining_op of %[[VALS]] : !pdl.range<value>
  // CHECK-DAG: pdl_interp.is_not_null %[[OP]] : !pdl.operation
  // CHECK-DAG: pdl_interp.check_operand_count of %[[ROOT1]] is at_least 1
  // CHECK-DAG: pdl_interp.check_result_count of %[[ROOT1]] is 0
  // CHECK-DAG: %[[VAL:.*]] = pdl_interp.get_operands 1 of %[[ROOT1]] : !pdl.value
  // CHECK-DAG: pdl_interp.is_not_null %[[VAL]]
  // CHECK-DAG: pdl_interp.is_not_null %[[VALS]]
  // CHECK-DAG: %[[VAL0:.*]] = pdl_interp.extract 0 of %[[VALS]]
  // CHECK-DAG: %[[ROOTS2:.*]] = pdl_interp.get_users of %[[VAL0]] : !pdl.value
  // CHECK-DAG: pdl_interp.foreach %[[ROOT2:.*]] : !pdl.operation in %[[ROOTS2]] {
  // CHECK-DAG:   %[[OPERANDS:.*]] = pdl_interp.get_operands 1 of %[[ROOT2]]
  // CHECK-DAG:   pdl_interp.are_equal %[[OPERANDS]], %[[VALS]] : !pdl.range<value> -> ^{{.*}}, ^[[CONTINUE:.*]]
  // CHECK-DAG:   pdl_interp.is_not_null %[[ROOT2]]
  // CHECK-DAG:   pdl_interp.check_operand_count of %[[ROOT2]] is at_least 1
  // CHECK-DAG:   pdl_interp.check_result_count of %[[ROOT2]] is 0
  // CHECK-DAG:   pdl_interp.check_operand_count of %[[OP]] is 0
  // CHECK-DAG:   pdl_interp.check_result_count of %[[OP]] is at_least 1
  pdl.pattern @variadic_results_at : benefit(1) {
    %type = type
    %types = types
    %val = operand
    %op = operation -> (%types, %type : !pdl.range<type>, !pdl.type)
    %vals = results 0 of %op -> !pdl.range<value>
    %root1 = operation(%vals, %val : !pdl.range<value>, !pdl.value)
    %root2 = operation(%val, %vals : !pdl.value, !pdl.range<value>)
    rewrite with "rewriter"(%root1, %root2 : !pdl.operation, !pdl.operation)
  }
}

// -----

// CHECK-LABEL: module @attribute_literal
module @attribute_literal {
  // CHECK: func @matcher(%{{.*}}: !pdl.operation)
  // CHECK: %[[ATTR:.*]] = pdl_interp.create_attribute 10 : i64
  // CHECK: pdl_interp.apply_constraint "constraint"(%[[ATTR]] : !pdl.attribute)

  // Check the correct lowering of an attribute that hasn't been bound.
  pdl.pattern : benefit(1) {
    %attr = attribute = 10
    pdl.apply_native_constraint "constraint"(%attr: !pdl.attribute)

    %root = operation
    rewrite %root with "rewriter"
  }
}

// -----

// CHECK-LABEL: module @type_literal
module @type_literal {
  // CHECK: func @matcher(%{{.*}}: !pdl.operation)
  // CHECK: %[[TYPE:.*]] = pdl_interp.create_type i32
  // CHECK: %[[TYPES:.*]] = pdl_interp.create_types [i32, i64]
  // CHECK: pdl_interp.apply_constraint "constraint"(%[[TYPE]], %[[TYPES]] : !pdl.type, !pdl.range<type>)

  // Check the correct lowering of a type that hasn't been bound.
  pdl.pattern : benefit(1) {
    %type = type : i32
    %types = types : [i32, i64]
    pdl.apply_native_constraint "constraint"(%type, %types: !pdl.type, !pdl.range<type>)

    %root = operation
    rewrite %root with "rewriter"
  }
}

// -----

// CHECK-LABEL: module @common_connector
module @common_connector {
  // Check the correct lowering when multiple roots are using the same
  // connector.

  // CHECK: func @matcher(%[[ROOTC:.*]]: !pdl.operation)
  // CHECK-DAG: %[[VAL2:.*]] = pdl_interp.get_operand 0 of %[[ROOTC]]
  // CHECK-DAG: %[[INTER:.*]] = pdl_interp.get_defining_op of %[[VAL2]] : !pdl.value
  // CHECK-DAG: pdl_interp.is_not_null %[[INTER]] : !pdl.operation -> ^bb2, ^bb1
  // CHECK-DAG: %[[VAL1:.*]] = pdl_interp.get_operand 0 of %[[INTER]]
  // CHECK-DAG: %[[OP:.*]] = pdl_interp.get_defining_op of %[[VAL1]] : !pdl.value
  // CHECK-DAG: pdl_interp.is_not_null %[[OP]]
  // CHECK-DAG: %[[VAL0:.*]] = pdl_interp.get_result 0 of %[[OP]]
  // CHECK-DAG: %[[ROOTS:.*]] = pdl_interp.get_users of %[[VAL0]] : !pdl.value
  // CHECK-DAG: pdl_interp.foreach %[[ROOTA:.*]] : !pdl.operation in %[[ROOTS]] {
  // CHECK-DAG:   pdl_interp.is_not_null %[[ROOTA]] : !pdl.operation -> ^{{.*}}, ^[[CONTA:.*]]
  // CHECK-DAG:   pdl_interp.continue
  // CHECK-DAG:   pdl_interp.foreach %[[ROOTB:.*]] : !pdl.operation in %[[ROOTS]] {
  // CHECK-DAG:     pdl_interp.is_not_null %[[ROOTB]] : !pdl.operation -> ^{{.*}}, ^[[CONTB:.*]]
  // CHECK-DAG:     %[[ROOTA_OP:.*]] = pdl_interp.get_operand 0 of %[[ROOTA]]
  // CHECK-DAG:     pdl_interp.are_equal %[[ROOTA_OP]], %[[VAL0]] : !pdl.value
  // CHECK-DAG:     %[[ROOTB_OP:.*]] = pdl_interp.get_operand 0 of %[[ROOTB]]
  // CHECK-DAG:     pdl_interp.are_equal %[[ROOTB_OP]], %[[VAL0]] : !pdl.value
  // CHECK-DAG:    } -> ^[[CONTA:.*]]
  pdl.pattern @common_connector : benefit(1) {
      %type = type
      %op = operation -> (%type, %type : !pdl.type, !pdl.type)
      %val0 = result 0 of %op
      %val1 = result 1 of %op
      %rootA = operation (%val0 : !pdl.value)
      %rootB = operation (%val0 : !pdl.value)
      %inter = operation (%val1 : !pdl.value) -> (%type : !pdl.type)
      %val2 = result 0 of %inter
      %rootC = operation (%val2 : !pdl.value)
      rewrite with "rewriter"(%rootA, %rootB, %rootC : !pdl.operation, !pdl.operation, !pdl.operation)
  }
}

// -----

// CHECK-LABEL: module @common_connector_range
module @common_connector_range {
  // Check the correct lowering when multiple roots are using the same
  // connector range.

  // CHECK: func @matcher(%[[ROOTC:.*]]: !pdl.operation)
  // CHECK-DAG: %[[VALS2:.*]] = pdl_interp.get_operands of %[[ROOTC]] : !pdl.range<value>
  // CHECK-DAG: %[[INTER:.*]] = pdl_interp.get_defining_op of %[[VALS2]] : !pdl.range<value>
  // CHECK-DAG: pdl_interp.is_not_null %[[INTER]] : !pdl.operation -> ^bb2, ^bb1
  // CHECK-DAG: %[[VALS1:.*]] = pdl_interp.get_operands of %[[INTER]] : !pdl.range<value>
  // CHECK-DAG: %[[OP:.*]] = pdl_interp.get_defining_op of %[[VALS1]] : !pdl.range<value>
  // CHECK-DAG: pdl_interp.is_not_null %[[OP]]
  // CHECK-DAG: %[[VALS0:.*]] = pdl_interp.get_results 0 of %[[OP]]
  // CHECK-DAG: %[[VAL0:.*]] = pdl_interp.extract 0 of %[[VALS0]] : !pdl.value
  // CHECK-DAG: %[[ROOTS:.*]] = pdl_interp.get_users of %[[VAL0]] : !pdl.value
  // CHECK-DAG: pdl_interp.foreach %[[ROOTA:.*]] : !pdl.operation in %[[ROOTS]] {
  // CHECK-DAG:   pdl_interp.is_not_null %[[ROOTA]] : !pdl.operation -> ^{{.*}}, ^[[CONTA:.*]]
  // CHECK-DAG:   pdl_interp.continue
  // CHECK-DAG:   pdl_interp.foreach %[[ROOTB:.*]] : !pdl.operation in %[[ROOTS]] {
  // CHECK-DAG:     pdl_interp.is_not_null %[[ROOTB]] : !pdl.operation -> ^{{.*}}, ^[[CONTB:.*]]
  // CHECK-DAG:     %[[ROOTA_OPS:.*]] = pdl_interp.get_operands of %[[ROOTA]]
  // CHECK-DAG:     pdl_interp.are_equal %[[ROOTA_OPS]], %[[VALS0]] : !pdl.range<value>
  // CHECK-DAG:     %[[ROOTB_OPS:.*]] = pdl_interp.get_operands of %[[ROOTB]]
  // CHECK-DAG:     pdl_interp.are_equal %[[ROOTB_OPS]], %[[VALS0]] : !pdl.range<value>
  // CHECK-DAG:    } -> ^[[CONTA:.*]]
  pdl.pattern @common_connector_range : benefit(1) {
    %types = types
    %op = operation -> (%types, %types : !pdl.range<type>, !pdl.range<type>)
    %vals0 = results 0 of %op -> !pdl.range<value>
    %vals1 = results 1 of %op -> !pdl.range<value>
    %rootA = operation (%vals0 : !pdl.range<value>)
    %rootB = operation (%vals0 : !pdl.range<value>)
    %inter = operation (%vals1 : !pdl.range<value>) -> (%types : !pdl.range<type>)
    %vals2 = results of %inter
    %rootC = operation (%vals2 : !pdl.range<value>)
    rewrite with "rewriter"(%rootA, %rootB, %rootC : !pdl.operation, !pdl.operation, !pdl.operation)
  }
}
