// RUN: mlir-opt -split-input-file -convert-pdl-to-pdl-interp %s | FileCheck %s

// -----

// CHECK-LABEL: module @external
module @external {
  // CHECK: module @rewriters
  // CHECK:   func @pdl_generated_rewriter(%[[ROOT:.*]]: !pdl.operation, %[[INPUT:.*]]: !pdl.value)
  // CHECK:     pdl_interp.apply_rewrite "rewriter" [true](%[[ROOT]], %[[INPUT]] : !pdl.operation, !pdl.value)
  pdl.pattern : benefit(1) {
    %input = pdl.operand
    %root = pdl.operation "foo.op"(%input : !pdl.value)
    pdl.rewrite %root with "rewriter"[true](%input : !pdl.value)
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
    %root = pdl.operation "foo.op"
    pdl.rewrite %root {
      pdl.erase %root
    }
  }
}

// -----

// CHECK-LABEL: module @operation_attributes
module @operation_attributes {
  // CHECK: module @rewriters
  // CHECK:   func @pdl_generated_rewriter(%[[ATTR:.*]]: !pdl.attribute, %[[ROOT:.*]]: !pdl.operation)
  // CHECK:     %[[ATTR1:.*]] = pdl_interp.create_attribute true
  // CHECK:     pdl_interp.create_operation "foo.op"() {"attr" = %[[ATTR]], "attr1" = %[[ATTR1]]}
  pdl.pattern : benefit(1) {
    %attr = pdl.attribute
    %root = pdl.operation "foo.op" {"attr" = %attr}
    pdl.rewrite %root {
      %attr1 = pdl.attribute true
      %newOp = pdl.operation "foo.op" {"attr" = %attr, "attr1" = %attr1}
      pdl.erase %root
    }
  }
}

// -----

// CHECK-LABEL: module @operation_operands
module @operation_operands {
  // CHECK: module @rewriters
  // CHECK:   func @pdl_generated_rewriter(%[[OPERAND:.*]]: !pdl.value, %[[ROOT:.*]]: !pdl.operation)
  // CHECK:     %[[NEWOP:.*]] = pdl_interp.create_operation "foo.op"(%[[OPERAND]])
  // CHECK:     %[[OPERAND1:.*]] = pdl_interp.get_result 0 of %[[NEWOP]]
  // CHECK:     pdl_interp.create_operation "foo.op2"(%[[OPERAND1]])
  pdl.pattern : benefit(1) {
    %operand = pdl.operand
    %root = pdl.operation "foo.op"(%operand : !pdl.value)
    pdl.rewrite %root {
      %type = pdl.type : i32
      %newOp = pdl.operation "foo.op"(%operand : !pdl.value) -> (%type : !pdl.type)
      %result = pdl.result 0 of %newOp
      %newOp1 = pdl.operation "foo.op2"(%result : !pdl.value)
      pdl.erase %root
    }
  }
}

// -----

// CHECK-LABEL: module @operation_operands
module @operation_operands {
  // CHECK: module @rewriters
  // CHECK:   func @pdl_generated_rewriter(%[[OPERAND:.*]]: !pdl.value, %[[ROOT:.*]]: !pdl.operation)
  // CHECK:     %[[NEWOP:.*]] = pdl_interp.create_operation "foo.op"(%[[OPERAND]])
  // CHECK:     %[[OPERAND1:.*]] = pdl_interp.get_result 0 of %[[NEWOP]]
  // CHECK:     pdl_interp.create_operation "foo.op2"(%[[OPERAND1]])
  pdl.pattern : benefit(1) {
    %operand = pdl.operand
    %root = pdl.operation "foo.op"(%operand : !pdl.value)
    pdl.rewrite %root {
      %type = pdl.type : i32
      %newOp = pdl.operation "foo.op"(%operand : !pdl.value) -> (%type : !pdl.type)
      %result = pdl.result 0 of %newOp
      %newOp1 = pdl.operation "foo.op2"(%result : !pdl.value)
      pdl.erase %root
    }
  }
}

// -----

// CHECK-LABEL: module @operation_result_types
module @operation_result_types {
  // CHECK: module @rewriters
  // CHECK:   func @pdl_generated_rewriter(%[[TYPE:.*]]: !pdl.type, %[[TYPE1:.*]]: !pdl.type
  // CHECK:     pdl_interp.create_operation "foo.op"() -> %[[TYPE]], %[[TYPE1]]
  pdl.pattern : benefit(1) {
    %rootType = pdl.type
    %rootType1 = pdl.type
    %root = pdl.operation "foo.op" -> (%rootType, %rootType1 : !pdl.type, !pdl.type)
    pdl.rewrite %root {
      %newType1 = pdl.type
      %newOp = pdl.operation "foo.op" -> (%rootType, %newType1 : !pdl.type, !pdl.type)
      pdl.replace %root with %newOp
    }
  }
}

// -----

// CHECK-LABEL: module @replace_with_op
module @replace_with_op {
  // CHECK: module @rewriters
  // CHECK:   func @pdl_generated_rewriter(%[[ROOT:.*]]: !pdl.operation)
  // CHECK:     %[[NEWOP:.*]] = pdl_interp.create_operation
  // CHECK:     %[[OP_RESULT:.*]] = pdl_interp.get_result 0 of %[[NEWOP]]
  // CHECK:     pdl_interp.replace %[[ROOT]] with(%[[OP_RESULT]])
  pdl.pattern : benefit(1) {
    %type = pdl.type : i32
    %root = pdl.operation "foo.op" -> (%type : !pdl.type)
    pdl.rewrite %root {
      %newOp = pdl.operation "foo.op" -> (%type : !pdl.type)
      pdl.replace %root with %newOp
    }
  }
}

// -----

// CHECK-LABEL: module @replace_with_values
module @replace_with_values {
  // CHECK: module @rewriters
  // CHECK:   func @pdl_generated_rewriter(%[[ROOT:.*]]: !pdl.operation)
  // CHECK:     %[[NEWOP:.*]] = pdl_interp.create_operation
  // CHECK:     %[[OP_RESULT:.*]] = pdl_interp.get_result 0 of %[[NEWOP]]
  // CHECK:     pdl_interp.replace %[[ROOT]] with(%[[OP_RESULT]])
  pdl.pattern : benefit(1) {
    %type = pdl.type : i32
    %root = pdl.operation "foo.op" -> (%type : !pdl.type)
    pdl.rewrite %root {
      %newOp = pdl.operation "foo.op" -> (%type : !pdl.type)
      %newResult = pdl.result 0 of %newOp
      pdl.replace %root with (%newResult : !pdl.value)
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
    %root = pdl.operation "foo.op"
    pdl.rewrite %root {
      %newOp = pdl.operation "foo.op"
      pdl.replace %root with %newOp
    }
  }
}

// -----

// CHECK-LABEL: module @apply_native_rewrite
module @apply_native_rewrite {
  // CHECK: module @rewriters
  // CHECK:   func @pdl_generated_rewriter(%[[ROOT:.*]]: !pdl.operation)
  // CHECK:     %[[TYPE:.*]] = pdl_interp.apply_rewrite "functor" [true](%[[ROOT]] : !pdl.operation) : !pdl.type
  // CHECK:     pdl_interp.create_operation "foo.op"() -> %[[TYPE]]
  pdl.pattern : benefit(1) {
    %type = pdl.type
    %root = pdl.operation "foo.op" -> (%type : !pdl.type)
    pdl.rewrite %root {
      %newType = pdl.apply_native_rewrite "functor"[true](%root : !pdl.operation) : !pdl.type
      %newOp = pdl.operation "foo.op" -> (%newType : !pdl.type)
      pdl.replace %root with %newOp
    }
  }
}
