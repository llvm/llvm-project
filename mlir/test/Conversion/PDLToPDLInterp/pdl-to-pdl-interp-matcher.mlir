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
    %root = pdl.operation "foo.op"
    pdl.rewrite %root with "rewriter"
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
    %type = pdl.type : i64
    %attr = pdl.attribute 10 : i64
    %attr1 = pdl.attribute : %type
    %root = pdl.operation {"attr" = %attr, "attr1" = %attr1}
    pdl.rewrite %root with "rewriter"
  }
}

// -----

// CHECK-LABEL: module @constraints
module @constraints {
  // CHECK: func @matcher(%[[ROOT:.*]]: !pdl.operation)
  // CHECK-DAG:   %[[INPUT:.*]] = pdl_interp.get_operand 0 of %[[ROOT]]
  // CHECK-DAG:   %[[INPUT1:.*]] = pdl_interp.get_operand 1 of %[[ROOT]]
  // CHECK-DAG:   %[[RESULT:.*]] = pdl_interp.get_result 0 of %[[ROOT]]
  // CHECK:       pdl_interp.apply_constraint "multi_constraint" [true](%[[INPUT]], %[[INPUT1]], %[[RESULT]]

  pdl.pattern : benefit(1) {
    %input0 = pdl.operand
    %input1 = pdl.operand
    %root = pdl.operation(%input0, %input1 : !pdl.value, !pdl.value)
    %result0 = pdl.result 0 of %root

    pdl.apply_native_constraint "multi_constraint"[true](%input0, %input1, %result0 : !pdl.value, !pdl.value, !pdl.value)
    pdl.rewrite %root with "rewriter"
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
    %type = pdl.type : i64
    %input = pdl.operand : %type
    %root = pdl.operation(%input, %input : !pdl.value, !pdl.value)
    pdl.rewrite %root with "rewriter"
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
    %types = pdl.types : [i64]
    %inputs = pdl.operands : %types
    %input = pdl.operand
    %root = pdl.operation(%input, %inputs, %input : !pdl.value, !pdl.range<value>, !pdl.value)
    pdl.rewrite %root with "rewriter"
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
    %types = pdl.types : [i64]
    %operands = pdl.operands : %types
    %root = pdl.operation(%operands : !pdl.range<value>)
    pdl.rewrite %root with "rewriter"
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
    %type1 = pdl.type : i32
    %type2 = pdl.type
    %root = pdl.operation -> (%type1, %type2 : !pdl.type, !pdl.type)
    pdl.rewrite %root with "rewriter"
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
    %types = pdl.types : [i64]
    %type = pdl.type
    %root = pdl.operation -> (%type, %types, %type : !pdl.type, !pdl.range<type>, !pdl.type)
    pdl.rewrite %root with "rewriter"
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
    %types = pdl.types : [i64]
    %root = pdl.operation -> (%types : !pdl.range<type>)
    pdl.rewrite %root with "rewriter"
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
    %type1 = pdl.type : i32
    %type2 = pdl.type
    %inputOp = pdl.operation -> (%type1, %type2 : !pdl.type, !pdl.type)
    %result1 = pdl.result 0 of %inputOp
    %result2 = pdl.result 1 of %inputOp

    %root = pdl.operation(%result1, %result2 : !pdl.value, !pdl.value)
    pdl.rewrite %root with "rewriter"
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
    %types = pdl.types
    %inputOp = pdl.operation -> (%types : !pdl.range<type>)
    %results = pdl.results of %inputOp

    %root = pdl.operation(%results : !pdl.range<value>)
    pdl.rewrite %root with "rewriter"
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
    %type = pdl.type : i32
    %root = pdl.operation -> (%type : !pdl.type)
    pdl.rewrite %root with "rewriter"
  }
  pdl.pattern : benefit(1) {
    %type = pdl.type : i64
    %root = pdl.operation -> (%type : !pdl.type)
    pdl.rewrite %root with "rewriter"
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
    %types = pdl.types : [i32]
    %root = pdl.operation -> (%types : !pdl.range<type>)
    pdl.rewrite %root with "rewriter"
  }
  pdl.pattern : benefit(1) {
    %types = pdl.types : [i64, i32]
    %root = pdl.operation -> (%types : !pdl.range<type>)
    pdl.rewrite %root with "rewriter"
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
    %operand = pdl.operand
    %operands = pdl.operands
    %root = pdl.operation(%operand, %operands : !pdl.value, !pdl.range<value>)
    pdl.rewrite %root with "rewriter"
  }
  pdl.pattern : benefit(1) {
    %operand = pdl.operand
    %operand2 = pdl.operand
    %operands = pdl.operands
    %root = pdl.operation(%operand, %operand2, %operands : !pdl.value, !pdl.value, !pdl.range<value>)
    pdl.rewrite %root with "rewriter"
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
    %type = pdl.type
    %types = pdl.types
    %root = pdl.operation -> (%type, %types : !pdl.type, !pdl.range<type>)
    pdl.rewrite %root with "rewriter"
  }
  pdl.pattern : benefit(1) {
    %type = pdl.type
    %type2 = pdl.type
    %types = pdl.types
    %root = pdl.operation -> (%type, %type2, %types : !pdl.type, !pdl.type, !pdl.range<type>)
    pdl.rewrite %root with "rewriter"
  }
}


// -----

// CHECK-LABEL: module @predicate_ordering
module @predicate_ordering  {
  // Check that the result is checked for null first, before applying the
  // constraint. The null check is prevalent in both patterns, so should be
  // prioritized first.

  // CHECK: func @matcher(%[[ROOT:.*]]: !pdl.operation)
  // CHECK:   %[[RESULT:.*]] = pdl_interp.get_result 0 of %[[ROOT]]
  // CHECK-NEXT: pdl_interp.is_not_null %[[RESULT]]
  // CHECK:   %[[RESULT_TYPE:.*]] = pdl_interp.get_value_type of %[[RESULT]]
  // CHECK: pdl_interp.apply_constraint "typeConstraint" [](%[[RESULT_TYPE]]

  pdl.pattern : benefit(1) {
    %resultType = pdl.type
    pdl.apply_native_constraint "typeConstraint"[](%resultType : !pdl.type)
    %root = pdl.operation -> (%resultType : !pdl.type)
    pdl.rewrite %root with "rewriter"
  }

  pdl.pattern : benefit(1) {
    %resultType = pdl.type
    %apply = pdl.operation -> (%resultType : !pdl.type)
    pdl.rewrite %apply with "rewriter"
  }
}
