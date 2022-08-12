// RUN: mlir-opt %s -test-pdl-bytecode-pass -split-input-file | FileCheck %s

// Note: Tests here are written using the PDL Interpreter dialect to avoid
// unnecessarily testing unnecessary aspects of the pattern compilation
// pipeline. These tests are written such that we can focus solely on the
// lowering/execution of the bytecode itself.

//===----------------------------------------------------------------------===//
// pdl_interp::ApplyConstraintOp
//===----------------------------------------------------------------------===//

module @patterns {
  pdl_interp.func @matcher(%root : !pdl.operation) {
    pdl_interp.apply_constraint "multi_entity_constraint"(%root, %root : !pdl.operation, !pdl.operation) -> ^pat, ^end

  ^pat:
    pdl_interp.apply_constraint "single_entity_constraint"(%root : !pdl.operation) -> ^pat2, ^end

  ^pat2:
    pdl_interp.record_match @rewriters::@success(%root : !pdl.operation) : benefit(1), loc([%root]) -> ^end

  ^end:
    pdl_interp.finalize
  }

  module @rewriters {
    pdl_interp.func @success(%root : !pdl.operation) {
      %op = pdl_interp.create_operation "test.replaced_by_pattern"
      pdl_interp.erase %root
      pdl_interp.finalize
    }
  }
}

// CHECK-LABEL: test.apply_constraint_1
// CHECK: "test.replaced_by_pattern"
module @ir attributes { test.apply_constraint_1 } {
  "test.op"() { test_attr } : () -> ()
}

// -----

module @patterns {
  pdl_interp.func @matcher(%root : !pdl.operation) {
    %results = pdl_interp.get_results of %root : !pdl.range<value>
    %types = pdl_interp.get_value_type of %results : !pdl.range<type>
    pdl_interp.apply_constraint "multi_entity_var_constraint"(%results, %types : !pdl.range<value>, !pdl.range<type>) -> ^pat, ^end

  ^pat:
    pdl_interp.record_match @rewriters::@success(%root : !pdl.operation) : benefit(1), loc([%root]) -> ^end

  ^end:
    pdl_interp.finalize
  }

  module @rewriters {
    pdl_interp.func @success(%root : !pdl.operation) {
      %op = pdl_interp.create_operation "test.replaced_by_pattern"
      pdl_interp.erase %root
      pdl_interp.finalize
    }
  }
}

// CHECK-LABEL: test.apply_constraint_2
// CHECK-NOT: "test.replaced_by_pattern"
// CHECK: "test.replaced_by_pattern"
module @ir attributes { test.apply_constraint_2 } {
  "test.failure_op"() { test_attr } : () -> ()
  "test.success_op"() : () -> (i32, i64)
}

// -----

//===----------------------------------------------------------------------===//
// pdl_interp::ApplyRewriteOp
//===----------------------------------------------------------------------===//

module @patterns {
  pdl_interp.func @matcher(%root : !pdl.operation) {
    pdl_interp.check_operation_name of %root is "test.op" -> ^pat, ^end

  ^pat:
    pdl_interp.record_match @rewriters::@success(%root : !pdl.operation) : benefit(1), loc([%root]) -> ^end

  ^end:
    pdl_interp.finalize
  }

  module @rewriters {
    pdl_interp.func @success(%root : !pdl.operation) {
      %operand = pdl_interp.get_operand 0 of %root
      pdl_interp.apply_rewrite "rewriter"(%root, %operand : !pdl.operation, !pdl.value)
      pdl_interp.finalize
    }
  }
}

// CHECK-LABEL: test.apply_rewrite_1
// CHECK: %[[INPUT:.*]] = "test.op_input"
// CHECK-NOT: "test.op"
// CHECK: "test.success"(%[[INPUT]])
module @ir attributes { test.apply_rewrite_1 } {
  %input = "test.op_input"() : () -> i32
  "test.op"(%input) : (i32) -> ()
}

// -----

module @patterns {
  pdl_interp.func @matcher(%root : !pdl.operation) {
    pdl_interp.check_operation_name of %root is "test.op" -> ^pat, ^end

  ^pat:
    pdl_interp.record_match @rewriters::@success(%root : !pdl.operation) : benefit(1), loc([%root]) -> ^end

  ^end:
    pdl_interp.finalize
  }

  module @rewriters {
    pdl_interp.func @success(%root : !pdl.operation) {
      %op = pdl_interp.apply_rewrite "creator"(%root : !pdl.operation) : !pdl.operation
      pdl_interp.erase %root
      pdl_interp.finalize
    }
  }
}

// CHECK-LABEL: test.apply_rewrite_2
// CHECK: "test.success"
module @ir attributes { test.apply_rewrite_2 } {
  "test.op"() : () -> ()
}

// -----

module @patterns {
  pdl_interp.func @matcher(%root : !pdl.operation) {
    pdl_interp.check_operation_name of %root is "test.op" -> ^pat, ^end

  ^pat:
    pdl_interp.record_match @rewriters::@success(%root : !pdl.operation) : benefit(1), loc([%root]) -> ^end

  ^end:
    pdl_interp.finalize
  }

  module @rewriters {
    pdl_interp.func @success(%root : !pdl.operation) {
      %operands, %types = pdl_interp.apply_rewrite "var_creator"(%root : !pdl.operation) : !pdl.range<value>, !pdl.range<type>
      %op = pdl_interp.create_operation "test.success"(%operands : !pdl.range<value>) -> (%types : !pdl.range<type>)
      pdl_interp.replace %root with (%operands : !pdl.range<value>)
      pdl_interp.finalize
    }
  }
}

// CHECK-LABEL: test.apply_rewrite_3
// CHECK: %[[OPERAND:.*]] = "test.producer"
// CHECK: "test.success"(%[[OPERAND]]) : (i32) -> i32
// CHECK: "test.consumer"(%[[OPERAND]])
module @ir attributes { test.apply_rewrite_3 } {
  %first_operand = "test.producer"() : () -> (i32)
  %operand = "test.op"(%first_operand) : (i32) -> (i32)
  "test.consumer"(%operand) : (i32) -> ()
}

// -----

module @patterns {
  pdl_interp.func @matcher(%root : !pdl.operation) {
    pdl_interp.check_operation_name of %root is "test.op" -> ^pat, ^end

  ^pat:
    pdl_interp.record_match @rewriters::@success(%root : !pdl.operation) : benefit(1), loc([%root]) -> ^end

  ^end:
    pdl_interp.finalize
  }

  module @rewriters {
    pdl_interp.func @success(%root : !pdl.operation) {
      %attr = pdl_interp.apply_rewrite "str_creator" : !pdl.attribute
      %type = pdl_interp.apply_rewrite "type_creator" : !pdl.type
      %newOp = pdl_interp.create_operation "test.success" {"attr" = %attr} -> (%type : !pdl.type)
      pdl_interp.erase %root
      pdl_interp.finalize
    }
  }
}

// CHECK-LABEL: test.apply_rewrite_4
// CHECK: "test.success"() {attr = "test.str"} : () -> f32
module @ir attributes { test.apply_rewrite_4 } {
  "test.op"() : () -> ()
}

// -----

//===----------------------------------------------------------------------===//
// pdl_interp::AreEqualOp
//===----------------------------------------------------------------------===//

module @patterns {
  pdl_interp.func @matcher(%root : !pdl.operation) {
    %test_attr = pdl_interp.create_attribute unit
    %attr = pdl_interp.get_attribute "test_attr" of %root
    pdl_interp.are_equal %test_attr, %attr : !pdl.attribute -> ^pat, ^end

  ^pat:
    pdl_interp.record_match @rewriters::@success(%root : !pdl.operation) : benefit(1), loc([%root]) -> ^end

  ^end:
    pdl_interp.finalize
  }

  module @rewriters {
    pdl_interp.func @success(%root : !pdl.operation) {
      %op = pdl_interp.create_operation "test.success"
      pdl_interp.erase %root
      pdl_interp.finalize
    }
  }
}

// CHECK-LABEL: test.are_equal_1
// CHECK: "test.success"
module @ir attributes { test.are_equal_1 } {
  "test.op"() { test_attr } : () -> ()
}

// -----

module @patterns {
  pdl_interp.func @matcher(%root : !pdl.operation) {
    %const_types = pdl_interp.create_types [i32, i64]
    %results = pdl_interp.get_results of %root : !pdl.range<value>
    %result_types = pdl_interp.get_value_type of %results : !pdl.range<type>
    pdl_interp.are_equal %result_types, %const_types : !pdl.range<type> -> ^pat, ^end

  ^pat:
    pdl_interp.record_match @rewriters::@success(%root : !pdl.operation) : benefit(1), loc([%root]) -> ^end

  ^end:
    pdl_interp.finalize
  }

  module @rewriters {
    pdl_interp.func @success(%root : !pdl.operation) {
      %op = pdl_interp.create_operation "test.success"
      pdl_interp.erase %root
      pdl_interp.finalize
    }
  }
}

// CHECK-LABEL: test.are_equal_2
// CHECK: "test.not_equal"
// CHECK: "test.success"
// CHECK-NOT: "test.op"
module @ir attributes { test.are_equal_2 } {
  "test.not_equal"() : () -> (i32)
  "test.op"() : () -> (i32, i64)
}

// -----

//===----------------------------------------------------------------------===//
// pdl_interp::BranchOp
//===----------------------------------------------------------------------===//

module @patterns {
  pdl_interp.func @matcher(%root : !pdl.operation) {
    pdl_interp.check_operation_name of %root is "test.op" -> ^pat1, ^end

  ^pat1:
    pdl_interp.branch ^pat2

  ^pat2:
    pdl_interp.record_match @rewriters::@success(%root : !pdl.operation) : benefit(2), loc([%root]) -> ^end

  ^end:
    pdl_interp.finalize
  }

  module @rewriters {
    pdl_interp.func @success(%root : !pdl.operation) {
      %op = pdl_interp.create_operation "test.success"
      pdl_interp.erase %root
      pdl_interp.finalize
    }
  }
}

// CHECK-LABEL: test.branch_1
// CHECK: "test.success"
module @ir attributes { test.branch_1 } {
  "test.op"() : () -> ()
}

// -----

//===----------------------------------------------------------------------===//
// pdl_interp::CheckAttributeOp
//===----------------------------------------------------------------------===//

module @patterns {
  pdl_interp.func @matcher(%root : !pdl.operation) {
    %attr = pdl_interp.get_attribute "test_attr" of %root
    pdl_interp.check_attribute %attr is unit -> ^pat, ^end

  ^pat:
    pdl_interp.record_match @rewriters::@success(%root : !pdl.operation) : benefit(1), loc([%root]) -> ^end

  ^end:
    pdl_interp.finalize
  }

  module @rewriters {
    pdl_interp.func @success(%root : !pdl.operation) {
      %op = pdl_interp.create_operation "test.success"
      pdl_interp.erase %root
      pdl_interp.finalize
    }
  }
}

// CHECK-LABEL: test.check_attribute_1
// CHECK: "test.success"
module @ir attributes { test.check_attribute_1 } {
  "test.op"() { test_attr } : () -> ()
}

// -----

//===----------------------------------------------------------------------===//
// pdl_interp::CheckOperandCountOp
//===----------------------------------------------------------------------===//

module @patterns {
  pdl_interp.func @matcher(%root : !pdl.operation) {
    pdl_interp.check_operand_count of %root is at_least 1 -> ^exact_check, ^end

  ^exact_check:
    pdl_interp.check_operand_count of %root is 2 -> ^pat, ^end

  ^pat:
    pdl_interp.record_match @rewriters::@success(%root : !pdl.operation) : benefit(1), loc([%root]) -> ^end

  ^end:
    pdl_interp.finalize
  }

  module @rewriters {
    pdl_interp.func @success(%root : !pdl.operation) {
      %op = pdl_interp.create_operation "test.success"
      pdl_interp.erase %root
      pdl_interp.finalize
    }
  }
}

// CHECK-LABEL: test.check_operand_count_1
// CHECK: "test.op"() : () -> i32
// CHECK: "test.success"
module @ir attributes { test.check_operand_count_1 } {
  %operand = "test.op"() : () -> i32
  "test.op"(%operand, %operand) : (i32, i32) -> ()
}

// -----

//===----------------------------------------------------------------------===//
// pdl_interp::CheckOperationNameOp
//===----------------------------------------------------------------------===//

module @patterns {
  pdl_interp.func @matcher(%root : !pdl.operation) {
    pdl_interp.check_operation_name of %root is "test.op" -> ^pat, ^end

  ^pat:
    pdl_interp.record_match @rewriters::@success(%root : !pdl.operation) : benefit(1), loc([%root]) -> ^end

  ^end:
    pdl_interp.finalize
  }

  module @rewriters {
    pdl_interp.func @success(%root : !pdl.operation) {
      %op = pdl_interp.create_operation "test.success"
      pdl_interp.erase %root
      pdl_interp.finalize
    }
  }
}

// CHECK-LABEL: test.check_operation_name_1
// CHECK: "test.success"
module @ir attributes { test.check_operation_name_1 } {
  "test.op"() : () -> ()
}

// -----

//===----------------------------------------------------------------------===//
// pdl_interp::CheckResultCountOp
//===----------------------------------------------------------------------===//

module @patterns {
  pdl_interp.func @matcher(%root : !pdl.operation) {
    pdl_interp.check_result_count of %root is at_least 1 -> ^exact_check, ^end

  ^exact_check:
    pdl_interp.check_result_count of %root is 2 -> ^pat, ^end

  ^pat:
    pdl_interp.record_match @rewriters::@success(%root : !pdl.operation) : benefit(1), loc([%root]) -> ^end

  ^end:
    pdl_interp.finalize
  }

  module @rewriters {
    pdl_interp.func @success(%root : !pdl.operation) {
      %op = pdl_interp.create_operation "test.success"
      pdl_interp.erase %root
      pdl_interp.finalize
    }
  }
}

// CHECK-LABEL: test.check_result_count_1
// CHECK: "test.op"() : () -> i32
// CHECK: "test.success"() : () -> ()
// CHECK-NOT: "test.op"() : () -> (i32, i32)
module @ir attributes { test.check_result_count_1 } {
  "test.op"() : () -> i32
  "test.op"() : () -> (i32, i32)
}

// -----

//===----------------------------------------------------------------------===//
// pdl_interp::CheckTypeOp
//===----------------------------------------------------------------------===//

module @patterns {
  pdl_interp.func @matcher(%root : !pdl.operation) {
    %attr = pdl_interp.get_attribute "test_attr" of %root
    pdl_interp.is_not_null %attr : !pdl.attribute -> ^pat1, ^end

  ^pat1:
    %type = pdl_interp.get_attribute_type of %attr
    pdl_interp.check_type %type is i32 -> ^pat2, ^end

  ^pat2:
    pdl_interp.record_match @rewriters::@success(%root : !pdl.operation) : benefit(1), loc([%root]) -> ^end

  ^end:
    pdl_interp.finalize
  }

  module @rewriters {
    pdl_interp.func @success(%root : !pdl.operation) {
      %op = pdl_interp.create_operation "test.success"
      pdl_interp.erase %root
      pdl_interp.finalize
    }
  }
}

// CHECK-LABEL: test.check_type_1
// CHECK: "test.success"
module @ir attributes { test.check_type_1 } {
  "test.op"() { test_attr = 10 : i32 } : () -> ()
}

// -----

//===----------------------------------------------------------------------===//
// pdl_interp::CheckTypesOp
//===----------------------------------------------------------------------===//

module @patterns {
  pdl_interp.func @matcher(%root : !pdl.operation) {
    %results = pdl_interp.get_results of %root : !pdl.range<value>
    %result_types = pdl_interp.get_value_type of %results : !pdl.range<type>
    pdl_interp.check_types %result_types are [i32] -> ^pat2, ^end

  ^pat2:
    pdl_interp.record_match @rewriters::@success(%root : !pdl.operation) : benefit(1), loc([%root]) -> ^end

  ^end:
    pdl_interp.finalize
  }

  module @rewriters {
    pdl_interp.func @success(%root : !pdl.operation) {
      %op = pdl_interp.create_operation "test.success"
      pdl_interp.erase %root
      pdl_interp.finalize
    }
  }
}

// CHECK-LABEL: test.check_types_1
// CHECK: "test.op"() : () -> (i32, i64)
// CHECK: "test.success"
// CHECK-NOT: "test.op"() : () -> i32
module @ir attributes { test.check_types_1 } {
  "test.op"() : () -> (i32, i64)
  "test.op"() : () -> i32
}

// -----

//===----------------------------------------------------------------------===//
// pdl_interp::ContinueOp
//===----------------------------------------------------------------------===//

// Fully tested within the tests for other operations.

//===----------------------------------------------------------------------===//
// pdl_interp::CreateAttributeOp
//===----------------------------------------------------------------------===//

// Fully tested within the tests for other operations.

//===----------------------------------------------------------------------===//
// pdl_interp::CreateOperationOp
//===----------------------------------------------------------------------===//

// Unused operation to force loading the `arithmetic` dialect for the
// test of type inferrence.
arith.constant 10

// Test support for inferring the types of an operation.
module @patterns {
  pdl_interp.func @matcher(%root : !pdl.operation) {
    pdl_interp.check_operation_name of %root is "test.op" -> ^pat, ^end

  ^pat:
    pdl_interp.record_match @rewriters::@success(%root : !pdl.operation) : benefit(1), loc([%root]) -> ^end

  ^end:
    pdl_interp.finalize
  }

  module @rewriters {
    pdl_interp.func @success(%root : !pdl.operation) {
      %attr = pdl_interp.create_attribute true
      %cst = pdl_interp.create_operation "arith.constant" {"value" = %attr} -> <inferred>
      %cstResults = pdl_interp.get_results of %cst : !pdl.range<value>
      %op = pdl_interp.create_operation "test.success"(%cstResults : !pdl.range<value>)
      pdl_interp.erase %root
      pdl_interp.finalize
    }
  }
}

// CHECK-LABEL: test.create_op_infer_results
// CHECK: %[[CST:.*]] = arith.constant true
// CHECK: "test.success"(%[[CST]])
module @ir attributes { test.create_op_infer_results } {
  %results:2 = "test.op"() : () -> (i64, i64)
}

// -----

//===----------------------------------------------------------------------===//
// pdl_interp::CreateTypeOp
//===----------------------------------------------------------------------===//

module @patterns {
  pdl_interp.func @matcher(%root : !pdl.operation) {
    %attr = pdl_interp.get_attribute "test_attr" of %root
    pdl_interp.is_not_null %attr : !pdl.attribute -> ^pat1, ^end

  ^pat1:
    %test_type = pdl_interp.create_type i32
    %type = pdl_interp.get_attribute_type of %attr
    pdl_interp.are_equal %type, %test_type : !pdl.type -> ^pat2, ^end

  ^pat2:
    pdl_interp.record_match @rewriters::@success(%root : !pdl.operation) : benefit(1), loc([%root]) -> ^end

  ^end:
    pdl_interp.finalize
  }

  module @rewriters {
    pdl_interp.func @success(%root : !pdl.operation) {
      %op = pdl_interp.create_operation "test.success"
      pdl_interp.erase %root
      pdl_interp.finalize
    }
  }
}

// CHECK-LABEL: test.create_type_1
// CHECK: "test.success"
module @ir attributes { test.create_type_1 } {
  "test.op"() { test_attr = 0 : i32 } : () -> ()
}

// -----

//===----------------------------------------------------------------------===//
// pdl_interp::CreateTypesOp
//===----------------------------------------------------------------------===//

// Fully tested within the tests for other operations.

//===----------------------------------------------------------------------===//
// pdl_interp::EraseOp
//===----------------------------------------------------------------------===//

// Fully tested within the tests for other operations.

//===----------------------------------------------------------------------===//
// pdl_interp::ExtractOp
//===----------------------------------------------------------------------===//

module @patterns {
  pdl_interp.func @matcher(%root : !pdl.operation) {
    %val = pdl_interp.get_result 0 of %root
    %ops = pdl_interp.get_users of %val : !pdl.value
    %op1 = pdl_interp.extract 1 of %ops : !pdl.operation
    pdl_interp.is_not_null %op1 : !pdl.operation -> ^success, ^end
  ^success:
    pdl_interp.record_match @rewriters::@success(%op1 : !pdl.operation) : benefit(1), loc([%root]) -> ^end
  ^end:
    pdl_interp.finalize
  }

  module @rewriters {
    pdl_interp.func @success(%matched : !pdl.operation) {
      %op = pdl_interp.create_operation "test.success"
      pdl_interp.erase %matched
      pdl_interp.finalize
    }
  }
}

// CHECK-LABEL: test.extract_op
// CHECK: "test.success"
// CHECK: %[[OPERAND:.*]] = "test.op"
// CHECK: "test.op"(%[[OPERAND]])
module @ir attributes { test.extract_op } {
  %operand = "test.op"() : () -> i32
  "test.op"(%operand) : (i32) -> (i32)
  "test.op"(%operand, %operand) : (i32, i32) -> (i32)
}

// -----

module @patterns {
  pdl_interp.func @matcher(%root : !pdl.operation) {
    %vals = pdl_interp.get_results of %root : !pdl.range<value>
    %types = pdl_interp.get_value_type of %vals : !pdl.range<type>
    %type1 = pdl_interp.extract 1 of %types : !pdl.type
    pdl_interp.is_not_null %type1 : !pdl.type -> ^success, ^end
  ^success:
    pdl_interp.record_match @rewriters::@success(%root : !pdl.operation) : benefit(1), loc([%root]) -> ^end
  ^end:
    pdl_interp.finalize
  }

  module @rewriters {
    pdl_interp.func @success(%matched : !pdl.operation) {
      %op = pdl_interp.create_operation "test.success"
      pdl_interp.erase %matched
      pdl_interp.finalize
    }
  }
}

// CHECK-LABEL: test.extract_type
// CHECK: %[[OPERAND:.*]] = "test.op"
// CHECK: "test.success"
// CHECK: "test.op"(%[[OPERAND]])
module @ir attributes { test.extract_type } {
  %operand = "test.op"() : () -> i32
  "test.op"(%operand) : (i32) -> (i32, i32)
  "test.op"(%operand) : (i32) -> (i32)
}

// -----

module @patterns {
  pdl_interp.func @matcher(%root : !pdl.operation) {
    %vals = pdl_interp.get_results of %root : !pdl.range<value>
    %val1 = pdl_interp.extract 1 of %vals : !pdl.value
    pdl_interp.is_not_null %val1 : !pdl.value -> ^success, ^end
  ^success:
    pdl_interp.record_match @rewriters::@success(%root : !pdl.operation) : benefit(1), loc([%root]) -> ^end
  ^end:
    pdl_interp.finalize
  }

  module @rewriters {
    pdl_interp.func @success(%matched : !pdl.operation) {
      %op = pdl_interp.create_operation "test.success"
      pdl_interp.erase %matched
      pdl_interp.finalize
    }
  }
}

// CHECK-LABEL: test.extract_value
// CHECK: %[[OPERAND:.*]] = "test.op"
// CHECK: "test.success"
// CHECK: "test.op"(%[[OPERAND]])
module @ir attributes { test.extract_value } {
  %operand = "test.op"() : () -> i32
  "test.op"(%operand) : (i32) -> (i32, i32)
  "test.op"(%operand) : (i32) -> (i32)
}

// -----

//===----------------------------------------------------------------------===//
// pdl_interp::FinalizeOp
//===----------------------------------------------------------------------===//

// Fully tested within the tests for other operations.

//===----------------------------------------------------------------------===//
// pdl_interp::ForEachOp
//===----------------------------------------------------------------------===//

module @patterns {
  pdl_interp.func @matcher(%root : !pdl.operation) {
    %val1 = pdl_interp.get_result 0 of %root
    %ops1 = pdl_interp.get_users of %val1 : !pdl.value
    pdl_interp.foreach %op1 : !pdl.operation in %ops1 {
      %val2 = pdl_interp.get_result 0 of %op1
      %ops2 = pdl_interp.get_users of %val2 : !pdl.value
      pdl_interp.foreach %op2 : !pdl.operation in %ops2 {
        pdl_interp.record_match @rewriters::@success(%op2 : !pdl.operation) : benefit(1), loc([%root]) -> ^cont
      ^cont:
        pdl_interp.continue
      } -> ^cont
    ^cont:
      pdl_interp.continue
    } -> ^end
  ^end:
    pdl_interp.finalize
  }

  module @rewriters {
    pdl_interp.func @success(%matched : !pdl.operation) {
      %op = pdl_interp.create_operation "test.success"
      pdl_interp.erase %matched
      pdl_interp.finalize
    }
  }
}

// CHECK-LABEL: test.foreach
// CHECK: "test.success"
// CHECK: "test.success"
// CHECK: "test.success"
// CHECK: "test.success"
// CHECK: %[[ROOT:.*]] = "test.op"
// CHECK: %[[VALA:.*]] = "test.op"(%[[ROOT]])
// CHECK: %[[VALB:.*]] = "test.op"(%[[ROOT]])
module @ir attributes { test.foreach } {
  %root = "test.op"() : () -> i32
  %valA = "test.op"(%root) : (i32) -> (i32)
  "test.op"(%valA) : (i32) -> (i32)
  "test.op"(%valA) : (i32) -> (i32)
  %valB = "test.op"(%root) : (i32) -> (i32)
  "test.op"(%valB) : (i32) -> (i32)
  "test.op"(%valB) : (i32) -> (i32)
}

// -----

//===----------------------------------------------------------------------===//
// pdl_interp::GetUsersOp
//===----------------------------------------------------------------------===//

module @patterns {
  pdl_interp.func @matcher(%root : !pdl.operation) {
    %val = pdl_interp.get_result 0 of %root
    %ops = pdl_interp.get_users of %val : !pdl.value
    pdl_interp.foreach %op : !pdl.operation in %ops {
      pdl_interp.record_match @rewriters::@success(%op : !pdl.operation) : benefit(1), loc([%root]) -> ^cont
    ^cont:
      pdl_interp.continue
    } -> ^end
  ^end:
    pdl_interp.finalize
  }

  module @rewriters {
    pdl_interp.func @success(%matched : !pdl.operation) {
      %op = pdl_interp.create_operation "test.success"
      pdl_interp.erase %matched
      pdl_interp.finalize
    }
  }
}

// CHECK-LABEL: test.get_users_of_value
// CHECK: "test.success"
// CHECK: "test.success"
// CHECK: %[[OPERAND:.*]] = "test.op"
module @ir attributes { test.get_users_of_value } {
  %operand = "test.op"() : () -> i32
  "test.op"(%operand) : (i32) -> (i32)
  "test.op"(%operand, %operand) : (i32, i32) -> (i32)
}

// -----

module @patterns {
  pdl_interp.func @matcher(%root : !pdl.operation) {
    pdl_interp.check_result_count of %root is at_least 2 -> ^next, ^end
  ^next:
    %vals = pdl_interp.get_results of %root : !pdl.range<value>
    %ops = pdl_interp.get_users of %vals : !pdl.range<value>
    pdl_interp.foreach %op : !pdl.operation in %ops {
      pdl_interp.record_match @rewriters::@success(%op : !pdl.operation) : benefit(1), loc([%root]) -> ^cont
    ^cont:
      pdl_interp.continue
    } -> ^end
  ^end:
    pdl_interp.finalize
  }

  module @rewriters {
    pdl_interp.func @success(%matched : !pdl.operation) {
      %op = pdl_interp.create_operation "test.success"
      pdl_interp.erase %matched
      pdl_interp.finalize
    }
  }
}

// CHECK-LABEL: test.get_all_users_of_range
// CHECK: "test.success"
// CHECK: "test.success"
// CHECK: %[[OPERANDS:.*]]:2 = "test.op"
module @ir attributes { test.get_all_users_of_range } {
  %operands:2 = "test.op"() : () -> (i32, i32)
  "test.op"(%operands#0) : (i32) -> (i32)
  "test.op"(%operands#1) : (i32) -> (i32)
}

// -----

module @patterns {
  pdl_interp.func @matcher(%root : !pdl.operation) {
    pdl_interp.check_result_count of %root is at_least 2 -> ^next, ^end
  ^next:
    %vals = pdl_interp.get_results of %root : !pdl.range<value>
    %val = pdl_interp.extract 0 of %vals : !pdl.value
    %ops = pdl_interp.get_users of %val : !pdl.value
    pdl_interp.foreach %op : !pdl.operation in %ops {
      pdl_interp.record_match @rewriters::@success(%op : !pdl.operation) : benefit(1), loc([%root]) -> ^cont
    ^cont:
      pdl_interp.continue
    } -> ^end
  ^end:
    pdl_interp.finalize
  }

  module @rewriters {
    pdl_interp.func @success(%matched : !pdl.operation) {
      %op = pdl_interp.create_operation "test.success"
      pdl_interp.erase %matched
      pdl_interp.finalize
    }
  }
}

// CHECK-LABEL: test.get_first_users_of_range
// CHECK: "test.success"
// CHECK: %[[OPERANDS:.*]]:2 = "test.op"
// CHECK: "test.op"
module @ir attributes { test.get_first_users_of_range } {
  %operands:2 = "test.op"() : () -> (i32, i32)
  "test.op"(%operands#0) : (i32) -> (i32)
  "test.op"(%operands#1) : (i32) -> (i32)
}

// -----

//===----------------------------------------------------------------------===//
// pdl_interp::GetAttributeOp
//===----------------------------------------------------------------------===//

// Fully tested within the tests for other operations.

//===----------------------------------------------------------------------===//
// pdl_interp::GetAttributeTypeOp
//===----------------------------------------------------------------------===//

// Fully tested within the tests for other operations.

//===----------------------------------------------------------------------===//
// pdl_interp::GetDefiningOpOp
//===----------------------------------------------------------------------===//

module @patterns {
  pdl_interp.func @matcher(%root : !pdl.operation) {
    pdl_interp.check_operand_count of %root is 5 -> ^pat1, ^end

  ^pat1:
    %operand0 = pdl_interp.get_operand 0 of %root
    %operand4 = pdl_interp.get_operand 4 of %root
    %defOp0 = pdl_interp.get_defining_op of %operand0 : !pdl.value
    %defOp4 = pdl_interp.get_defining_op of %operand4 : !pdl.value
    pdl_interp.are_equal %defOp0, %defOp4 : !pdl.operation -> ^pat2, ^end

  ^pat2:
    pdl_interp.record_match @rewriters::@success(%root : !pdl.operation) : benefit(1), loc([%root]) -> ^end

  ^end:
    pdl_interp.finalize
  }

  module @rewriters {
    pdl_interp.func @success(%root : !pdl.operation) {
      %op = pdl_interp.create_operation "test.success"
      pdl_interp.erase %root
      pdl_interp.finalize
    }
  }
}

// CHECK-LABEL: test.get_defining_op_1
// CHECK: %[[OPERAND0:.*]] = "test.op"
// CHECK: %[[OPERAND1:.*]] = "test.op"
// CHECK: "test.success"
// CHECK: "test.op"(%[[OPERAND0]], %[[OPERAND0]], %[[OPERAND0]], %[[OPERAND0]], %[[OPERAND1]])
module @ir attributes { test.get_defining_op_1 } {
  %operand = "test.op"() : () -> i32
  %other_operand = "test.op"() : () -> i32
  "test.op"(%operand, %operand, %operand, %operand, %operand) : (i32, i32, i32, i32, i32) -> ()
  "test.op"(%operand, %operand, %operand, %operand, %other_operand) : (i32, i32, i32, i32, i32) -> ()
}

// -----

//===----------------------------------------------------------------------===//
// pdl_interp::GetOperandOp
//===----------------------------------------------------------------------===//

// Fully tested within the tests for other operations.

//===----------------------------------------------------------------------===//
// pdl_interp::GetOperandsOp
//===----------------------------------------------------------------------===//

module @patterns {
  pdl_interp.func @matcher(%root : !pdl.operation) {
    pdl_interp.check_operand_count of %root is 2 -> ^pat1, ^end

  ^pat1:
    %operands = pdl_interp.get_operands 0 of %root : !pdl.range<value>
    %full_operands = pdl_interp.get_operands of %root : !pdl.range<value>
    pdl_interp.are_equal %operands, %full_operands : !pdl.range<value> -> ^pat2, ^end

  ^pat2:
    pdl_interp.record_match @rewriters::@success(%root : !pdl.operation) : benefit(1), loc([%root]) -> ^end

  ^end:
    pdl_interp.finalize
  }

  module @rewriters {
    pdl_interp.func @success(%root : !pdl.operation) {
      %op = pdl_interp.create_operation "test.success"
      pdl_interp.erase %root
      pdl_interp.finalize
    }
  }
}

// CHECK-LABEL: test.get_operands_1
// CHECK: "test.success"
module @ir attributes { test.get_operands_1 } {
  %inputs:2 = "test.producer"() : () -> (i32, i32)
  "test.op"(%inputs#0, %inputs#1) : (i32, i32) -> ()
}

// -----

// Test all of the various combinations related to `AttrSizedOperandSegments`.
module @patterns {
  pdl_interp.func @matcher(%root : !pdl.operation) {
    pdl_interp.check_operation_name of %root is "test.attr_sized_operands" -> ^pat1, ^end

  ^pat1:
    %operands_0 = pdl_interp.get_operands 0 of %root : !pdl.range<value>
    pdl_interp.is_not_null %operands_0 : !pdl.range<value> -> ^pat2, ^end

  ^pat2:
    %operands_0_single = pdl_interp.get_operands 0 of %root : !pdl.value
    pdl_interp.is_not_null %operands_0_single : !pdl.value -> ^end, ^pat3

  ^pat3:
    %operands_1 = pdl_interp.get_operands 1 of %root : !pdl.range<value>
    pdl_interp.is_not_null %operands_1 : !pdl.range<value> -> ^pat4, ^end

  ^pat4:
    %operands_1_single = pdl_interp.get_operands 1 of %root : !pdl.value
    pdl_interp.is_not_null %operands_1_single : !pdl.value -> ^end, ^pat5

  ^pat5:
    %operands_2 = pdl_interp.get_operands 2 of %root : !pdl.range<value>
    pdl_interp.is_not_null %operands_2 : !pdl.range<value> -> ^pat6, ^end

  ^pat6:
    %operands_2_single = pdl_interp.get_operands 2 of %root : !pdl.value
    pdl_interp.is_not_null %operands_2_single : !pdl.value -> ^pat7, ^end

  ^pat7:
    %invalid_operands = pdl_interp.get_operands 50 of %root : !pdl.value
    pdl_interp.is_not_null %invalid_operands : !pdl.value -> ^end, ^pat8

  ^pat8:
    pdl_interp.record_match @rewriters::@success(%root, %operands_0, %operands_1, %operands_2, %operands_2_single : !pdl.operation, !pdl.range<value>, !pdl.range<value>, !pdl.range<value>, !pdl.value) : benefit(1), loc([%root]) -> ^end


  ^end:
    pdl_interp.finalize
  }

  module @rewriters {
    pdl_interp.func @success(%root: !pdl.operation, %operands_0: !pdl.range<value>, %operands_1: !pdl.range<value>, %operands_2: !pdl.range<value>, %operands_2_single: !pdl.value) {
      %op0 = pdl_interp.create_operation "test.success"(%operands_0 : !pdl.range<value>)
      %op1 = pdl_interp.create_operation "test.success"(%operands_1 : !pdl.range<value>)
      %op2 = pdl_interp.create_operation "test.success"(%operands_2 : !pdl.range<value>)
      %op3 = pdl_interp.create_operation "test.success"(%operands_2_single : !pdl.value)
      pdl_interp.erase %root
      pdl_interp.finalize
    }
  }
}

// CHECK-LABEL: test.get_operands_2
// CHECK-NEXT:  %[[INPUTS:.*]]:5 = "test.producer"() : () -> (i32, i32, i32, i32, i32)
// CHECK-NEXT:  "test.success"() : () -> ()
// CHECK-NEXT:  "test.success"(%[[INPUTS]]#0, %[[INPUTS]]#1, %[[INPUTS]]#2, %[[INPUTS]]#3) : (i32, i32, i32, i32) -> ()
// CHECK-NEXT:  "test.success"(%[[INPUTS]]#4) : (i32) -> ()
// CHECK-NEXT:  "test.success"(%[[INPUTS]]#4) : (i32) -> ()
module @ir attributes { test.get_operands_2 } {
  %inputs:5 = "test.producer"() : () -> (i32, i32, i32, i32, i32)
  "test.attr_sized_operands"(%inputs#0, %inputs#1, %inputs#2, %inputs#3, %inputs#4) {operand_segment_sizes = dense<[0, 4, 1, 0]> : vector<4xi32>} : (i32, i32, i32, i32, i32) -> ()
}

// -----

//===----------------------------------------------------------------------===//
// pdl_interp::GetResultOp
//===----------------------------------------------------------------------===//

module @patterns {
  pdl_interp.func @matcher(%root : !pdl.operation) {
    pdl_interp.check_result_count of %root is 5 -> ^pat1, ^end

  ^pat1:
    %result0 = pdl_interp.get_result 0 of %root
    %result4 = pdl_interp.get_result 4 of %root
    %result0_type = pdl_interp.get_value_type of %result0 : !pdl.type
    %result4_type = pdl_interp.get_value_type of %result4 : !pdl.type
    pdl_interp.are_equal %result0_type, %result4_type : !pdl.type -> ^pat2, ^end

  ^pat2:
    pdl_interp.record_match @rewriters::@success(%root : !pdl.operation) : benefit(1), loc([%root]) -> ^end

  ^end:
    pdl_interp.finalize
  }

  module @rewriters {
    pdl_interp.func @success(%root : !pdl.operation) {
      %op = pdl_interp.create_operation "test.success"
      pdl_interp.erase %root
      pdl_interp.finalize
    }
  }
}

// CHECK-LABEL: test.get_result_1
// CHECK: "test.success"
// CHECK: "test.op"() : () -> (i32, i32, i32, i32, i64)
module @ir attributes { test.get_result_1 } {
  %a:5 = "test.op"() : () -> (i32, i32, i32, i32, i32)
  %b:5 = "test.op"() : () -> (i32, i32, i32, i32, i64)
}

// -----

//===----------------------------------------------------------------------===//
// pdl_interp::GetResultsOp
//===----------------------------------------------------------------------===//

module @patterns {
  pdl_interp.func @matcher(%root : !pdl.operation) {
    pdl_interp.check_result_count of %root is 5 -> ^pat1, ^end

  ^pat1:
    %results = pdl_interp.get_results 0 of %root : !pdl.range<value>
    %full_results = pdl_interp.get_results of %root : !pdl.range<value>
    pdl_interp.are_equal %results, %full_results : !pdl.range<value> -> ^pat2, ^end

  ^pat2:
    pdl_interp.record_match @rewriters::@success(%root : !pdl.operation) : benefit(1), loc([%root]) -> ^end

  ^end:
    pdl_interp.finalize
  }

  module @rewriters {
    pdl_interp.func @success(%root : !pdl.operation) {
      %op = pdl_interp.create_operation "test.success"
      pdl_interp.erase %root
      pdl_interp.finalize
    }
  }
}

// CHECK-LABEL: test.get_results_1
// CHECK: "test.success"
module @ir attributes { test.get_results_1 } {
  %a:5 = "test.producer"() : () -> (i32, i32, i32, i32, i32)
}

// -----

// Test all of the various combinations related to `AttrSizedResultSegments`.
module @patterns {
  pdl_interp.func @matcher(%root : !pdl.operation) {
    pdl_interp.check_operation_name of %root is "test.attr_sized_results" -> ^pat1, ^end

  ^pat1:
    %results_0 = pdl_interp.get_results 0 of %root : !pdl.range<value>
    pdl_interp.is_not_null %results_0 : !pdl.range<value> -> ^pat2, ^end

  ^pat2:
    %results_0_single = pdl_interp.get_results 0 of %root : !pdl.value
    pdl_interp.is_not_null %results_0_single : !pdl.value -> ^end, ^pat3

  ^pat3:
    %results_1 = pdl_interp.get_results 1 of %root : !pdl.range<value>
    pdl_interp.is_not_null %results_1 : !pdl.range<value> -> ^pat4, ^end

  ^pat4:
    %results_1_single = pdl_interp.get_results 1 of %root : !pdl.value
    pdl_interp.is_not_null %results_1_single : !pdl.value -> ^end, ^pat5

  ^pat5:
    %results_2 = pdl_interp.get_results 2 of %root : !pdl.range<value>
    pdl_interp.is_not_null %results_2 : !pdl.range<value> -> ^pat6, ^end

  ^pat6:
    %results_2_single = pdl_interp.get_results 2 of %root : !pdl.value
    pdl_interp.is_not_null %results_2_single : !pdl.value -> ^pat7, ^end

  ^pat7:
    %invalid_results = pdl_interp.get_results 50 of %root : !pdl.value
    pdl_interp.is_not_null %invalid_results : !pdl.value -> ^end, ^pat8

  ^pat8:
    pdl_interp.record_match @rewriters::@success(%root, %results_0, %results_1, %results_2, %results_2_single : !pdl.operation, !pdl.range<value>, !pdl.range<value>, !pdl.range<value>, !pdl.value) : benefit(1), loc([%root]) -> ^end


  ^end:
    pdl_interp.finalize
  }

  module @rewriters {
    pdl_interp.func @success(%root: !pdl.operation, %results_0: !pdl.range<value>, %results_1: !pdl.range<value>, %results_2: !pdl.range<value>, %results_2_single: !pdl.value) {
      %results_0_types = pdl_interp.get_value_type of %results_0 : !pdl.range<type>
      %results_1_types = pdl_interp.get_value_type of %results_1 : !pdl.range<type>
      %results_2_types = pdl_interp.get_value_type of %results_2 : !pdl.range<type>
      %results_2_single_types = pdl_interp.get_value_type of %results_2_single : !pdl.type

      %op0 = pdl_interp.create_operation "test.success" -> (%results_0_types : !pdl.range<type>)
      %op1 = pdl_interp.create_operation "test.success" -> (%results_1_types : !pdl.range<type>)
      %op2 = pdl_interp.create_operation "test.success" -> (%results_2_types : !pdl.range<type>)
      %op3 = pdl_interp.create_operation "test.success" -> (%results_2_single_types : !pdl.type)

      %new_results_0 = pdl_interp.get_results of %op0 : !pdl.range<value>
      %new_results_1 = pdl_interp.get_results of %op1 : !pdl.range<value>
      %new_results_2 = pdl_interp.get_results of %op2 : !pdl.range<value>

      pdl_interp.replace %root with (%new_results_0, %new_results_1, %new_results_2 : !pdl.range<value>, !pdl.range<value>, !pdl.range<value>)
      pdl_interp.finalize
    }
  }
}

// CHECK-LABEL: test.get_results_2
// CHECK: "test.success"() : () -> ()
// CHECK: %[[RESULTS_1:.*]]:4 = "test.success"() : () -> (i32, i32, i32, i32)
// CHECK: %[[RESULTS_2:.*]] = "test.success"() : () -> i32
// CHECK: %[[RESULTS_2_SINGLE:.*]] = "test.success"() : () -> i32
// CHECK: "test.consumer"(%[[RESULTS_1]]#0, %[[RESULTS_1]]#1, %[[RESULTS_1]]#2, %[[RESULTS_1]]#3, %[[RESULTS_2]]) : (i32, i32, i32, i32, i32) -> ()
module @ir attributes { test.get_results_2 } {
  %results:5 = "test.attr_sized_results"() {result_segment_sizes = dense<[0, 4, 1, 0]> : vector<4xi32>} : () -> (i32, i32, i32, i32, i32)
  "test.consumer"(%results#0, %results#1, %results#2, %results#3, %results#4) : (i32, i32, i32, i32, i32) -> ()
}

// -----

//===----------------------------------------------------------------------===//
// pdl_interp::GetValueTypeOp
//===----------------------------------------------------------------------===//

// Fully tested within the tests for other operations.

//===----------------------------------------------------------------------===//
// pdl_interp::IsNotNullOp
//===----------------------------------------------------------------------===//

// Fully tested within the tests for other operations.

//===----------------------------------------------------------------------===//
// pdl_interp::RecordMatchOp
//===----------------------------------------------------------------------===//

// Check that the highest benefit pattern is selected.
module @patterns {
  pdl_interp.func @matcher(%root : !pdl.operation) {
    pdl_interp.check_operation_name of %root is "test.op" -> ^pat1, ^end

  ^pat1:
    pdl_interp.record_match @rewriters::@failure(%root : !pdl.operation) : benefit(1), loc([%root]) -> ^pat2

  ^pat2:
    pdl_interp.record_match @rewriters::@success(%root : !pdl.operation) : benefit(2), loc([%root]) -> ^end

  ^end:
    pdl_interp.finalize
  }

  module @rewriters {
    pdl_interp.func @failure(%root : !pdl.operation) {
      pdl_interp.erase %root
      pdl_interp.finalize
    }
    pdl_interp.func @success(%root : !pdl.operation) {
      %op = pdl_interp.create_operation "test.success"
      pdl_interp.erase %root
      pdl_interp.finalize
    }
  }
}

// CHECK-LABEL: test.record_match_1
// CHECK: "test.success"
module @ir attributes { test.record_match_1 } {
  "test.op"() : () -> ()
}

// -----

// Check that ranges are properly forwarded to the result.
module @patterns {
  pdl_interp.func @matcher(%root : !pdl.operation) {
    pdl_interp.check_operation_name of %root is "test.op" -> ^pat1, ^end

  ^pat1:
    %operands = pdl_interp.get_operands of %root : !pdl.range<value>
    %results = pdl_interp.get_results of %root : !pdl.range<value>
    %types = pdl_interp.get_value_type of %results : !pdl.range<type>
    pdl_interp.record_match @rewriters::@success(%operands, %types, %root : !pdl.range<value>, !pdl.range<type>, !pdl.operation) : benefit(1), loc([%root]) -> ^end

  ^end:
    pdl_interp.finalize
  }

  module @rewriters {
    pdl_interp.func @success(%operands: !pdl.range<value>, %types: !pdl.range<type>, %root: !pdl.operation) {
      %op = pdl_interp.create_operation "test.success"(%operands : !pdl.range<value>) -> (%types : !pdl.range<type>)
      %results = pdl_interp.get_results of %op : !pdl.range<value>
      pdl_interp.replace %root with (%results : !pdl.range<value>)
      pdl_interp.finalize
    }
  }
}

// CHECK-LABEL: test.record_match_2
// CHECK: %[[OPERAND:.*]] = "test.producer"() : () -> i32
// CHECK: %[[RESULTS:.*]]:2 = "test.success"(%[[OPERAND]]) : (i32) -> (i64, i32)
// CHECK: "test.consumer"(%[[RESULTS]]#0, %[[RESULTS]]#1) : (i64, i32) -> ()
module @ir attributes { test.record_match_2 } {
  %input = "test.producer"() : () -> i32
  %results:2 = "test.op"(%input) : (i32) -> (i64, i32)
  "test.consumer"(%results#0, %results#1) : (i64, i32) -> ()
}

// -----

//===----------------------------------------------------------------------===//
// pdl_interp::ReplaceOp
//===----------------------------------------------------------------------===//

module @patterns {
  pdl_interp.func @matcher(%root : !pdl.operation) {
    pdl_interp.check_operation_name of %root is "test.op" -> ^pat, ^end

  ^pat:
    pdl_interp.record_match @rewriters::@success(%root : !pdl.operation) : benefit(1), loc([%root]) -> ^end

  ^end:
    pdl_interp.finalize
  }

  module @rewriters {
    pdl_interp.func @success(%root : !pdl.operation) {
      %operand = pdl_interp.get_operand 0 of %root
      pdl_interp.replace %root with (%operand : !pdl.value)
      pdl_interp.finalize
    }
  }
}

// CHECK-LABEL: test.replace_op_1
// CHECK: %[[INPUT:.*]] = "test.op_input"
// CHECK-NOT: "test.op"
// CHECK: "test.op_consumer"(%[[INPUT]])
module @ir attributes { test.replace_op_1 } {
  %input = "test.op_input"() : () -> i32
  %result = "test.op"(%input) : (i32) -> i32
  "test.op_consumer"(%result) : (i32) -> ()
}

// -----

//===----------------------------------------------------------------------===//
// pdl_interp::SwitchAttributeOp
//===----------------------------------------------------------------------===//

module @patterns {
  pdl_interp.func @matcher(%root : !pdl.operation) {
    %attr = pdl_interp.get_attribute "test_attr" of %root
    pdl_interp.switch_attribute %attr to [0, unit](^end, ^pat) -> ^end

  ^pat:
    %attr_2 = pdl_interp.get_attribute "test_attr_2" of %root
    pdl_interp.switch_attribute %attr_2 to [0, unit](^end, ^end) -> ^pat2

  ^pat2:
    pdl_interp.record_match @rewriters::@success(%root : !pdl.operation) : benefit(1), loc([%root]) -> ^end

  ^end:
    pdl_interp.finalize
  }

  module @rewriters {
    pdl_interp.func @success(%root : !pdl.operation) {
      %op = pdl_interp.create_operation "test.success"
      pdl_interp.erase %root
      pdl_interp.finalize
    }
  }
}

// CHECK-LABEL: test.switch_attribute_1
// CHECK: "test.success"
module @ir attributes { test.switch_attribute_1 } {
  "test.op"() { test_attr } : () -> ()
}

// -----

//===----------------------------------------------------------------------===//
// pdl_interp::SwitchOperandCountOp
//===----------------------------------------------------------------------===//

module @patterns {
  pdl_interp.func @matcher(%root : !pdl.operation) {
    pdl_interp.switch_operand_count of %root to dense<[0, 1]> : vector<2xi32>(^end, ^pat) -> ^end

  ^pat:
    pdl_interp.switch_operand_count of %root to dense<[0, 2]> : vector<2xi32>(^end, ^end) -> ^pat2

  ^pat2:
    pdl_interp.record_match @rewriters::@success(%root : !pdl.operation) : benefit(1), loc([%root]) -> ^end

  ^end:
    pdl_interp.finalize
  }

  module @rewriters {
    pdl_interp.func @success(%root : !pdl.operation) {
      %op = pdl_interp.create_operation "test.success"
      pdl_interp.erase %root
      pdl_interp.finalize
    }
  }
}

// CHECK-LABEL: test.switch_operand_1
// CHECK: "test.success"
module @ir attributes { test.switch_operand_1 } {
  %input = "test.op_input"() : () -> i32
  "test.op"(%input) : (i32) -> ()
}

// -----

//===----------------------------------------------------------------------===//
// pdl_interp::SwitchOperationNameOp
//===----------------------------------------------------------------------===//

module @patterns {
  pdl_interp.func @matcher(%root : !pdl.operation) {
    pdl_interp.switch_operation_name of %root to ["foo.op", "test.op"](^end, ^pat1) -> ^end

  ^pat1:
    pdl_interp.switch_operation_name of %root to ["foo.op", "bar.op"](^end, ^end) -> ^pat2

  ^pat2:
    pdl_interp.record_match @rewriters::@success(%root : !pdl.operation) : benefit(1), loc([%root]) -> ^end

  ^end:
    pdl_interp.finalize
  }

  module @rewriters {
    pdl_interp.func @success(%root : !pdl.operation) {
      %op = pdl_interp.create_operation "test.success"
      pdl_interp.erase %root
      pdl_interp.finalize
    }
  }
}

// CHECK-LABEL: test.switch_operation_name_1
// CHECK: "test.success"
module @ir attributes { test.switch_operation_name_1 } {
  "test.op"() : () -> ()
}

// -----

//===----------------------------------------------------------------------===//
// pdl_interp::SwitchResultCountOp
//===----------------------------------------------------------------------===//

module @patterns {
  pdl_interp.func @matcher(%root : !pdl.operation) {
    pdl_interp.switch_result_count of %root to dense<[0, 1]> : vector<2xi32>(^end, ^pat) -> ^end

  ^pat:
    pdl_interp.switch_result_count of %root to dense<[0, 2]> : vector<2xi32>(^end, ^end) -> ^pat2

  ^pat2:
    pdl_interp.record_match @rewriters::@success(%root : !pdl.operation) : benefit(1), loc([%root]) -> ^end

  ^end:
    pdl_interp.finalize
  }

  module @rewriters {
    pdl_interp.func @success(%root : !pdl.operation) {
      %op = pdl_interp.create_operation "test.success"
      pdl_interp.erase %root
      pdl_interp.finalize
    }
  }
}

// CHECK-LABEL: test.switch_result_1
// CHECK: "test.success"
module @ir attributes { test.switch_result_1 } {
  "test.op"() : () -> i32
}

// -----

//===----------------------------------------------------------------------===//
// pdl_interp::SwitchTypeOp
//===----------------------------------------------------------------------===//

module @patterns {
  pdl_interp.func @matcher(%root : !pdl.operation) {
    %attr = pdl_interp.get_attribute "test_attr" of %root
    pdl_interp.is_not_null %attr : !pdl.attribute -> ^pat1, ^end

  ^pat1:
    %type = pdl_interp.get_attribute_type of %attr
    pdl_interp.switch_type %type to [i32, i64](^pat2, ^end) -> ^end

  ^pat2:
    pdl_interp.switch_type %type to [i16, i64](^end, ^end) -> ^pat3

  ^pat3:
    pdl_interp.record_match @rewriters::@success(%root : !pdl.operation) : benefit(1), loc([%root]) -> ^end

  ^end:
    pdl_interp.finalize
  }

  module @rewriters {
    pdl_interp.func @success(%root : !pdl.operation) {
      %op = pdl_interp.create_operation "test.success"
      pdl_interp.erase %root
      pdl_interp.finalize
    }
  }
}

// CHECK-LABEL: test.switch_type_1
// CHECK: "test.success"
module @ir attributes { test.switch_type_1 } {
  "test.op"() { test_attr = 10 : i32 } : () -> ()
}

// -----

//===----------------------------------------------------------------------===//
// pdl_interp::SwitchTypesOp
//===----------------------------------------------------------------------===//

module @patterns {
  pdl_interp.func @matcher(%root : !pdl.operation) {
    %results = pdl_interp.get_results of %root : !pdl.range<value>
    %types = pdl_interp.get_value_type of %results : !pdl.range<type>
    pdl_interp.switch_types %types to [[i64, i64], [i32]](^pat2, ^end) -> ^end

  ^pat2:
    pdl_interp.switch_types %types to [[i32], [i64, i32]](^end, ^end) -> ^pat3

  ^pat3:
    pdl_interp.record_match @rewriters::@success(%root : !pdl.operation) : benefit(1), loc([%root]) -> ^end

  ^end:
    pdl_interp.finalize
  }

  module @rewriters {
    pdl_interp.func @success(%root : !pdl.operation) {
      %op = pdl_interp.create_operation "test.success"
      pdl_interp.erase %root
      pdl_interp.finalize
    }
  }
}

// CHECK-LABEL: test.switch_types_1
// CHECK: "test.success"
module @ir attributes { test.switch_types_1 } {
  %results:2 = "test.op"() : () -> (i64, i64)
}
