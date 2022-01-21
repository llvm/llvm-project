// RUN: mlir-opt -split-input-file -convert-pdl-to-pdl-interp %s | FileCheck %s

// -----

// CHECK-LABEL: module @external
module @external {
  // CHECK: module @rewriters
  // CHECK:   func @pdl_generated_rewriter(%[[ROOT:.*]]: !pdl.operation, %[[INPUT:.*]]: !pdl.value)
  // CHECK:     pdl_interp.apply_rewrite "rewriter" [true](%[[ROOT]], %[[INPUT]] : !pdl.operation, !pdl.value)
  pdl.pattern : benefit(1) {
    %input = operand
    %root = operation "foo.op"(%input : !pdl.value)
    rewrite %root with "rewriter"[true](%input : !pdl.value)
  }
}

// -----

// CHECK-LABEL: module @erase
module @erase {
  // CHECK: module @rewriters
  // CHECK:   func @pdl_generated_rewriter(%[[ROOT:.*]]: !pdl.operation)
  // CHECK:     pdl_interp.erase %[[ROOT]]
  // CHECK:     pdl_interp.finalize
  pdl.pattern : benefit(1) {
    %root = operation "foo.op"
    rewrite %root {
      erase %root
    }
  }
}

// -----

// CHECK-LABEL: module @operation_attributes
module @operation_attributes {
  // CHECK: module @rewriters
  // CHECK:   func @pdl_generated_rewriter(%[[ATTR:.*]]: !pdl.attribute, %[[ROOT:.*]]: !pdl.operation)
  // CHECK:     %[[ATTR1:.*]] = pdl_interp.create_attribute true
  // CHECK:     pdl_interp.create_operation "foo.op" {"attr" = %[[ATTR]], "attr1" = %[[ATTR1]]}
  pdl.pattern : benefit(1) {
    %attr = attribute
    %root = operation "foo.op" {"attr" = %attr}
    rewrite %root {
      %attr1 = attribute true
      %newOp = operation "foo.op" {"attr" = %attr, "attr1" = %attr1}
      erase %root
    }
  }
}

// -----

// CHECK-LABEL: module @operation_operands
module @operation_operands {
  // CHECK: module @rewriters
  // CHECK:   func @pdl_generated_rewriter(%[[OPERAND:.*]]: !pdl.value, %[[ROOT:.*]]: !pdl.operation)
  // CHECK:     %[[NEWOP:.*]] = pdl_interp.create_operation "foo.op"(%[[OPERAND]] : !pdl.value)
  // CHECK:     %[[OPERAND1:.*]] = pdl_interp.get_result 0 of %[[NEWOP]]
  // CHECK:     pdl_interp.create_operation "foo.op2"(%[[OPERAND1]] : !pdl.value)
  pdl.pattern : benefit(1) {
    %operand = operand
    %root = operation "foo.op"(%operand : !pdl.value)
    rewrite %root {
      %type = type : i32
      %newOp = operation "foo.op"(%operand : !pdl.value) -> (%type : !pdl.type)
      %result = result 0 of %newOp
      %newOp1 = operation "foo.op2"(%result : !pdl.value)
      erase %root
    }
  }
}

// -----

// CHECK-LABEL: module @operation_infer_types_from_replaceop
module @operation_infer_types_from_replaceop {
  // CHECK: module @rewriters
  // CHECK:   func @pdl_generated_rewriter(%[[ROOT:.*]]: !pdl.operation
  // CHECK:     %[[RESULTS:.*]] = pdl_interp.get_results of %[[ROOT]]
  // CHECK:     %[[RESULT_TYPES:.*]] = pdl_interp.get_value_type of %[[RESULTS]]
  // CHECK:     pdl_interp.create_operation "foo.op" -> (%[[RESULT_TYPES]] : !pdl.range<type>)
  pdl.pattern : benefit(1) {
    %rootType = type
    %rootType1 = type
    %root = operation "foo.op" -> (%rootType, %rootType1 : !pdl.type, !pdl.type)
    rewrite %root {
      %newType1 = type
      %newOp = operation "foo.op" -> (%rootType, %newType1 : !pdl.type, !pdl.type)
      replace %root with %newOp
    }
  }
}

// -----

// CHECK-LABEL: module @operation_infer_types_from_otherop_individual_results
module @operation_infer_types_from_otherop_individual_results {
  // CHECK: module @rewriters
  // CHECK:   func @pdl_generated_rewriter(%[[TYPE:.*]]: !pdl.type, %[[TYPES:.*]]: !pdl.range<type>
  // CHECK:     pdl_interp.create_operation "foo.op" -> (%[[TYPE]], %[[TYPES]] : !pdl.type, !pdl.range<type>)
  pdl.pattern : benefit(1) {
    %rootType = type
    %rootTypes = types
    %root = operation "foo.op" -> (%rootType, %rootTypes : !pdl.type, !pdl.range<type>)
    rewrite %root {
      %newOp = operation "foo.op" -> (%rootType, %rootTypes : !pdl.type, !pdl.range<type>)
    }
  }
}

// -----

// CHECK-LABEL: module @operation_infer_types_from_otherop_results
module @operation_infer_types_from_otherop_results {
  // CHECK: module @rewriters
  // CHECK:   func @pdl_generated_rewriter(%[[TYPES:.*]]: !pdl.range<type>
  // CHECK:     pdl_interp.create_operation "foo.op" -> (%[[TYPES]] : !pdl.range<type>)
  pdl.pattern : benefit(1) {
    %rootTypes = types
    %root = operation "foo.op" -> (%rootTypes : !pdl.range<type>)
    rewrite %root {
      %newOp = operation "foo.op" -> (%rootTypes : !pdl.range<type>)
    }
  }
}

// -----

// CHECK-LABEL: module @replace_with_op
module @replace_with_op {
  // CHECK: module @rewriters
  // CHECK:   func @pdl_generated_rewriter(%[[ROOT:.*]]: !pdl.operation)
  // CHECK:     %[[NEWOP:.*]] = pdl_interp.create_operation
  // CHECK:     %[[RESULTS:.*]] = pdl_interp.get_results of %[[NEWOP]]
  // CHECK:     pdl_interp.replace %[[ROOT]] with (%[[RESULTS]] : !pdl.range<value>)
  pdl.pattern : benefit(1) {
    %type = type : i32
    %root = operation "foo.op" -> (%type : !pdl.type)
    rewrite %root {
      %newOp = operation "foo.op" -> (%type : !pdl.type)
      replace %root with %newOp
    }
  }
}

// -----

// CHECK-LABEL: module @replace_with_values
module @replace_with_values {
  // CHECK: module @rewriters
  // CHECK:   func @pdl_generated_rewriter({{.*}}, %[[ROOT:.*]]: !pdl.operation)
  // CHECK:     %[[NEWOP:.*]] = pdl_interp.create_operation
  // CHECK:     %[[RESULT:.*]] = pdl_interp.get_result 0 of %[[NEWOP]]
  // CHECK:     %[[RESULTS:.*]] = pdl_interp.get_results 1 of %[[NEWOP]] : !pdl.range<value>
  // CHECK:     %[[RESULTS_2:.*]] = pdl_interp.get_results 2 of %[[NEWOP]] : !pdl.value
  // CHECK:     pdl_interp.replace %[[ROOT]] with (%[[RESULT]], %[[RESULTS]], %[[RESULTS_2]] : !pdl.value, !pdl.range<value>, !pdl.value)
  pdl.pattern : benefit(1) {
    %types = types
    %root = operation "foo.op" -> (%types : !pdl.range<type>)
    rewrite %root {
      %newOp = operation "foo.op" -> (%types : !pdl.range<type>)
      %newResult = result 0 of %newOp
      %newResults = results 1 of %newOp -> !pdl.range<value>
      %newResults2 = results 2 of %newOp -> !pdl.value
      replace %root with (%newResult, %newResults, %newResults2 : !pdl.value, !pdl.range<value>, !pdl.value)
    }
  }
}

// -----

// CHECK-LABEL: module @replace_with_no_results
module @replace_with_no_results {
  // CHECK: module @rewriters
  // CHECK:   func @pdl_generated_rewriter(%[[ROOT:.*]]: !pdl.operation)
  // CHECK:     pdl_interp.create_operation "foo.op"
  // CHECK:     pdl_interp.erase %[[ROOT]]
  pdl.pattern : benefit(1) {
    %root = operation "foo.op"
    rewrite %root {
      %newOp = operation "foo.op"
      replace %root with %newOp
    }
  }
}

// -----

// CHECK-LABEL: module @apply_native_rewrite
module @apply_native_rewrite {
  // CHECK: module @rewriters
  // CHECK:   func @pdl_generated_rewriter(%[[ROOT:.*]]: !pdl.operation)
  // CHECK:     %[[TYPE:.*]] = pdl_interp.apply_rewrite "functor" [true](%[[ROOT]] : !pdl.operation) : !pdl.type
  // CHECK:     pdl_interp.create_operation "foo.op" -> (%[[TYPE]] : !pdl.type)
  pdl.pattern : benefit(1) {
    %type = type
    %root = operation "foo.op" -> (%type : !pdl.type)
    rewrite %root {
      %newType = apply_native_rewrite "functor"[true](%root : !pdl.operation) : !pdl.type
      %newOp = operation "foo.op" -> (%newType : !pdl.type)
    }
  }
}

// -----

// CHECK-LABEL: module @unbound_rewrite_op
module @unbound_rewrite_op {
  // CHECK: module @rewriters
  // CHECK:   func @pdl_generated_rewriter()
  // CHECK:     %[[UNUSED:.*]] = pdl_interp.create_operation "bar.op"
  // CHECK:     pdl_interp.finalize
  pdl.pattern : benefit(1) {
    %root = operation "foo.op"
    rewrite %root {
      %unused = operation "bar.op"
    }
  }
}

// -----
