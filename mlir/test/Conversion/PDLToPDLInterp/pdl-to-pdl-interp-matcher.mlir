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
  // CHECK:     pdl_interp.apply_rewrite "rewriter" on %[[REWRITE_ROOT]]
  // CHECK:     pdl_interp.finalize
  pdl.pattern : benefit(1) {
    %root = pdl.operation "foo.op"()
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
  // CHECK:       pdl_interp.apply_constraint "multi_constraint" [true](%[[INPUT]], %[[INPUT1]] : !pdl.value, !pdl.value)

  pdl.pattern : benefit(1) {
    %input0 = pdl.operand
    %input1 = pdl.operand

    pdl.apply_constraint "multi_constraint"[true](%input0, %input1 : !pdl.value, !pdl.value)

    %root = pdl.operation(%input0, %input1)
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
    %root = pdl.operation(%input, %input)
    pdl.rewrite %root with "rewriter"
  }
}

// -----

// CHECK-LABEL: module @results
module @results {
  // CHECK: func @matcher(%[[ROOT:.*]]: !pdl.operation)
  // CHECK:   pdl_interp.check_result_count of %[[ROOT]] is 2

  // Get the input and check the type.
  // CHECK-DAG:   %[[RESULT:.*]] = pdl_interp.get_result 0 of %[[ROOT]]
  // CHECK-DAG:   pdl_interp.is_not_null %[[RESULT]] : !pdl.value
  // CHECK-DAG:   %[[RESULT_TYPE:.*]] = pdl_interp.get_value_type of %[[RESULT]]
  // CHECK-DAG:   pdl_interp.check_type %[[RESULT_TYPE]] is i32

  // Get the second operand and check that it is equal to the first.
  // CHECK-DAG:  %[[RESULT1:.*]] = pdl_interp.get_result 1 of %[[ROOT]]
  // CHECK-NOT: pdl_interp.get_value_type of %[[RESULT1]]
  pdl.pattern : benefit(1) {
    %type1 = pdl.type : i32
    %type2 = pdl.type
    %root, %results:2 = pdl.operation -> %type1, %type2
    pdl.rewrite %root with "rewriter"
  }
}

// -----

// CHECK-LABEL: module @switch_result_types
module @switch_result_types {
  // CHECK: func @matcher(%[[ROOT:.*]]: !pdl.operation)
  // CHECK:   %[[RESULT:.*]] = pdl_interp.get_result 0 of %[[ROOT]]
  // CHECK:   %[[RESULT_TYPE:.*]] = pdl_interp.get_value_type of %[[RESULT]]
  // CHECK:   pdl_interp.switch_type %[[RESULT_TYPE]] to [i32, i64]
  pdl.pattern : benefit(1) {
    %type = pdl.type : i32
    %root, %result = pdl.operation -> %type
    pdl.rewrite %root with "rewriter"
  }
  pdl.pattern : benefit(1) {
    %type = pdl.type : i64
    %root, %result = pdl.operation -> %type
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
    pdl.apply_constraint "typeConstraint"[](%resultType : !pdl.type)
    %root, %result = pdl.operation -> %resultType
    pdl.rewrite %root with "rewriter"
  }

  pdl.pattern : benefit(1) {
    %resultType = pdl.type
    %apply, %applyRes = pdl.operation -> %resultType
    pdl.rewrite %apply with "rewriter"
  }
}
