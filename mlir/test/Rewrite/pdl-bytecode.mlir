// RUN: mlir-opt %s -test-pdl-bytecode-pass -split-input-file | FileCheck %s

// Note: Tests here are written using the PDL Interpreter dialect to avoid
// unnecessarily testing unnecessary aspects of the pattern compilation
// pipeline. These tests are written such that we can focus solely on the
// lowering/execution of the bytecode itself.

//===----------------------------------------------------------------------===//
// pdl_interp::ApplyConstraintOp
//===----------------------------------------------------------------------===//

module @patterns {
  func @matcher(%root : !pdl.operation) {
    pdl_interp.apply_constraint "multi_entity_constraint"(%root, %root : !pdl.operation, !pdl.operation) -> ^pat, ^end

  ^pat:
    pdl_interp.apply_constraint "single_entity_constraint"(%root : !pdl.operation) -> ^pat2, ^end

  ^pat2:
    pdl_interp.record_match @rewriters::@success(%root : !pdl.operation) : benefit(1), loc([%root]) -> ^end

  ^end:
    pdl_interp.finalize
  }

  module @rewriters {
    func @success(%root : !pdl.operation) {
      %op = pdl_interp.create_operation "test.replaced_by_pattern"() -> ()
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

//===----------------------------------------------------------------------===//
// pdl_interp::ApplyRewriteOp
//===----------------------------------------------------------------------===//

module @patterns {
  func @matcher(%root : !pdl.operation) {
    pdl_interp.check_operation_name of %root is "test.op" -> ^pat, ^end

  ^pat:
    pdl_interp.record_match @rewriters::@success(%root : !pdl.operation) : benefit(1), loc([%root]) -> ^end

  ^end:
    pdl_interp.finalize
  }

  module @rewriters {
    func @success(%root : !pdl.operation) {
      %operand = pdl_interp.get_operand 0 of %root
      pdl_interp.apply_rewrite "rewriter"[42](%operand : !pdl.value) on %root
      pdl_interp.finalize
    }
  }
}

// CHECK-LABEL: test.apply_rewrite_1
// CHECK: %[[INPUT:.*]] = "test.op_input"
// CHECK-NOT: "test.op"
// CHECK: "test.success"(%[[INPUT]]) {constantParams = [42]}
module @ir attributes { test.apply_rewrite_1 } {
  %input = "test.op_input"() : () -> i32
  "test.op"(%input) : (i32) -> ()
}
// -----

//===----------------------------------------------------------------------===//
// pdl_interp::AreEqualOp
//===----------------------------------------------------------------------===//

module @patterns {
  func @matcher(%root : !pdl.operation) {
    %test_attr = pdl_interp.create_attribute unit
    %attr = pdl_interp.get_attribute "test_attr" of %root
    pdl_interp.are_equal %test_attr, %attr : !pdl.attribute -> ^pat, ^end

  ^pat:
    pdl_interp.record_match @rewriters::@success(%root : !pdl.operation) : benefit(1), loc([%root]) -> ^end

  ^end:
    pdl_interp.finalize
  }

  module @rewriters {
    func @success(%root : !pdl.operation) {
      %op = pdl_interp.create_operation "test.success"() -> ()
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

//===----------------------------------------------------------------------===//
// pdl_interp::BranchOp
//===----------------------------------------------------------------------===//

module @patterns {
  func @matcher(%root : !pdl.operation) {
    pdl_interp.check_operation_name of %root is "test.op" -> ^pat1, ^end

  ^pat1:
    pdl_interp.branch ^pat2

  ^pat2:
    pdl_interp.record_match @rewriters::@success(%root : !pdl.operation) : benefit(2), loc([%root]) -> ^end

  ^end:
    pdl_interp.finalize
  }

  module @rewriters {
    func @success(%root : !pdl.operation) {
      %op = pdl_interp.create_operation "test.success"() -> ()
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
  func @matcher(%root : !pdl.operation) {
    %attr = pdl_interp.get_attribute "test_attr" of %root
    pdl_interp.check_attribute %attr is unit -> ^pat, ^end

  ^pat:
    pdl_interp.record_match @rewriters::@success(%root : !pdl.operation) : benefit(1), loc([%root]) -> ^end

  ^end:
    pdl_interp.finalize
  }

  module @rewriters {
    func @success(%root : !pdl.operation) {
      %op = pdl_interp.create_operation "test.success"() -> ()
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
  func @matcher(%root : !pdl.operation) {
    pdl_interp.check_operand_count of %root is 1 -> ^pat, ^end

  ^pat:
    pdl_interp.record_match @rewriters::@success(%root : !pdl.operation) : benefit(1), loc([%root]) -> ^end

  ^end:
    pdl_interp.finalize
  }

  module @rewriters {
    func @success(%root : !pdl.operation) {
      %op = pdl_interp.create_operation "test.success"() -> ()
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
  "test.op"(%operand) : (i32) -> ()
}

// -----

//===----------------------------------------------------------------------===//
// pdl_interp::CheckOperationNameOp
//===----------------------------------------------------------------------===//

module @patterns {
  func @matcher(%root : !pdl.operation) {
    pdl_interp.check_operation_name of %root is "test.op" -> ^pat, ^end

  ^pat:
    pdl_interp.record_match @rewriters::@success(%root : !pdl.operation) : benefit(1), loc([%root]) -> ^end

  ^end:
    pdl_interp.finalize
  }

  module @rewriters {
    func @success(%root : !pdl.operation) {
      %op = pdl_interp.create_operation "test.success"() -> ()
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
  func @matcher(%root : !pdl.operation) {
    pdl_interp.check_result_count of %root is 1 -> ^pat, ^end

  ^pat:
    pdl_interp.record_match @rewriters::@success(%root : !pdl.operation) : benefit(1), loc([%root]) -> ^end

  ^end:
    pdl_interp.finalize
  }

  module @rewriters {
    func @success(%root : !pdl.operation) {
      %op = pdl_interp.create_operation "test.success"() -> ()
      pdl_interp.erase %root
      pdl_interp.finalize
    }
  }
}

// CHECK-LABEL: test.check_result_count_1
// CHECK: "test.success"() : () -> ()
module @ir attributes { test.check_result_count_1 } {
  "test.op"() : () -> i32
}

// -----

//===----------------------------------------------------------------------===//
// pdl_interp::CheckTypeOp
//===----------------------------------------------------------------------===//

module @patterns {
  func @matcher(%root : !pdl.operation) {
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
    func @success(%root : !pdl.operation) {
      %op = pdl_interp.create_operation "test.success"() -> ()
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
// pdl_interp::CreateAttributeOp
//===----------------------------------------------------------------------===//

// Fully tested within the tests for other operations.

//===----------------------------------------------------------------------===//
// pdl_interp::CreateNativeOp
//===----------------------------------------------------------------------===//

// -----

module @patterns {
  func @matcher(%root : !pdl.operation) {
    pdl_interp.check_operation_name of %root is "test.op" -> ^pat, ^end

  ^pat:
    pdl_interp.record_match @rewriters::@success(%root : !pdl.operation) : benefit(1), loc([%root]) -> ^end

  ^end:
    pdl_interp.finalize
  }

  module @rewriters {
    func @success(%root : !pdl.operation) {
      %op = pdl_interp.create_native "creator"(%root : !pdl.operation) : !pdl.operation
      pdl_interp.erase %root
      pdl_interp.finalize
    }
  }
}

// CHECK-LABEL: test.create_native_1
// CHECK: "test.success"
module @ir attributes { test.create_native_1 } {
  "test.op"() : () -> ()
}

//===----------------------------------------------------------------------===//
// pdl_interp::CreateOperationOp
//===----------------------------------------------------------------------===//

// -----

//===----------------------------------------------------------------------===//
// pdl_interp::CreateTypeOp
//===----------------------------------------------------------------------===//

module @patterns {
  func @matcher(%root : !pdl.operation) {
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
    func @success(%root : !pdl.operation) {
      %op = pdl_interp.create_operation "test.success"() -> ()
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
// pdl_interp::EraseOp
//===----------------------------------------------------------------------===//

// Fully tested within the tests for other operations.

//===----------------------------------------------------------------------===//
// pdl_interp::FinalizeOp
//===----------------------------------------------------------------------===//

// Fully tested within the tests for other operations.

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
  func @matcher(%root : !pdl.operation) {
    pdl_interp.check_operand_count of %root is 5 -> ^pat1, ^end

  ^pat1:
    %operand0 = pdl_interp.get_operand 0 of %root
    %operand4 = pdl_interp.get_operand 4 of %root
    %defOp0 = pdl_interp.get_defining_op of %operand0
    %defOp4 = pdl_interp.get_defining_op of %operand4
    pdl_interp.are_equal %defOp0, %defOp4 : !pdl.operation -> ^pat2, ^end

  ^pat2:
    pdl_interp.record_match @rewriters::@success(%root : !pdl.operation) : benefit(1), loc([%root]) -> ^end

  ^end:
    pdl_interp.finalize
  }

  module @rewriters {
    func @success(%root : !pdl.operation) {
      %op = pdl_interp.create_operation "test.success"() -> ()
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
// pdl_interp::GetResultOp
//===----------------------------------------------------------------------===//

module @patterns {
  func @matcher(%root : !pdl.operation) {
    pdl_interp.check_result_count of %root is 5 -> ^pat1, ^end

  ^pat1:
    %result0 = pdl_interp.get_result 0 of %root
    %result4 = pdl_interp.get_result 4 of %root
    %result0_type = pdl_interp.get_value_type of %result0
    %result4_type = pdl_interp.get_value_type of %result4
    pdl_interp.are_equal %result0_type, %result4_type : !pdl.type -> ^pat2, ^end

  ^pat2:
    pdl_interp.record_match @rewriters::@success(%root : !pdl.operation) : benefit(1), loc([%root]) -> ^end

  ^end:
    pdl_interp.finalize
  }

  module @rewriters {
    func @success(%root : !pdl.operation) {
      %op = pdl_interp.create_operation "test.success"() -> ()
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
// pdl_interp::GetValueTypeOp
//===----------------------------------------------------------------------===//

// Fully tested within the tests for other operations.

//===----------------------------------------------------------------------===//
// pdl_interp::InferredTypeOp
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
  func @matcher(%root : !pdl.operation) {
    pdl_interp.check_operation_name of %root is "test.op" -> ^pat1, ^end

  ^pat1:
    pdl_interp.record_match @rewriters::@failure(%root : !pdl.operation) : benefit(1), loc([%root]) -> ^pat2

  ^pat2:
    pdl_interp.record_match @rewriters::@success(%root : !pdl.operation) : benefit(2), loc([%root]) -> ^end

  ^end:
    pdl_interp.finalize
  }

  module @rewriters {
    func @failure(%root : !pdl.operation) {
      pdl_interp.erase %root
      pdl_interp.finalize
    }
    func @success(%root : !pdl.operation) {
      %op = pdl_interp.create_operation "test.success"() -> ()
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

//===----------------------------------------------------------------------===//
// pdl_interp::ReplaceOp
//===----------------------------------------------------------------------===//

module @patterns {
  func @matcher(%root : !pdl.operation) {
    pdl_interp.check_operation_name of %root is "test.op" -> ^pat, ^end

  ^pat:
    pdl_interp.record_match @rewriters::@success(%root : !pdl.operation) : benefit(1), loc([%root]) -> ^end

  ^end:
    pdl_interp.finalize
  }

  module @rewriters {
    func @success(%root : !pdl.operation) {
      %operand = pdl_interp.get_operand 0 of %root
      pdl_interp.replace %root with (%operand)
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
  func @matcher(%root : !pdl.operation) {
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
    func @success(%root : !pdl.operation) {
      %op = pdl_interp.create_operation "test.success"() -> ()
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
  func @matcher(%root : !pdl.operation) {
    pdl_interp.switch_operand_count of %root to dense<[0, 1]> : vector<2xi32>(^end, ^pat) -> ^end

  ^pat:
    pdl_interp.switch_operand_count of %root to dense<[0, 2]> : vector<2xi32>(^end, ^end) -> ^pat2

  ^pat2:
    pdl_interp.record_match @rewriters::@success(%root : !pdl.operation) : benefit(1), loc([%root]) -> ^end

  ^end:
    pdl_interp.finalize
  }

  module @rewriters {
    func @success(%root : !pdl.operation) {
      %op = pdl_interp.create_operation "test.success"() -> ()
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
  func @matcher(%root : !pdl.operation) {
    pdl_interp.switch_operation_name of %root to ["foo.op", "test.op"](^end, ^pat1) -> ^end

  ^pat1:
    pdl_interp.switch_operation_name of %root to ["foo.op", "bar.op"](^end, ^end) -> ^pat2

  ^pat2:
    pdl_interp.record_match @rewriters::@success(%root : !pdl.operation) : benefit(1), loc([%root]) -> ^end

  ^end:
    pdl_interp.finalize
  }

  module @rewriters {
    func @success(%root : !pdl.operation) {
      %op = pdl_interp.create_operation "test.success"() -> ()
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
  func @matcher(%root : !pdl.operation) {
    pdl_interp.switch_result_count of %root to dense<[0, 1]> : vector<2xi32>(^end, ^pat) -> ^end

  ^pat:
    pdl_interp.switch_result_count of %root to dense<[0, 2]> : vector<2xi32>(^end, ^end) -> ^pat2

  ^pat2:
    pdl_interp.record_match @rewriters::@success(%root : !pdl.operation) : benefit(1), loc([%root]) -> ^end

  ^end:
    pdl_interp.finalize
  }

  module @rewriters {
    func @success(%root : !pdl.operation) {
      %op = pdl_interp.create_operation "test.success"() -> ()
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
  func @matcher(%root : !pdl.operation) {
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
    func @success(%root : !pdl.operation) {
      %op = pdl_interp.create_operation "test.success"() -> ()
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
