// RUN: mlir-opt -split-input-file %s | mlir-opt
// RUN: mlir-opt -split-input-file -mlir-print-op-generic -mlir-print-local-scope %s | FileCheck %s --check-prefix=CHECK-GENERIC

// -----

pdl.pattern @operations : benefit(1) {
  // Operation with attributes and results.
  %attribute = attribute
  %type = type
  %op0 = operation {"attr" = %attribute} -> (%type : !pdl.type)
  %op0_result = pdl.result 0 of %op0

  // Operation with input.
  %input = operand
  %root = operation(%op0_result, %input : !pdl.value, !pdl.value)
  rewrite %root with "rewriter"
}

// -----

pdl.pattern @rewrite_with_args : benefit(1) {
  %input = operand
  %root = operation(%input : !pdl.value)
  rewrite %root with "rewriter"(%input : !pdl.value)
}

// -----

pdl.pattern @rewrite_multi_root_optimal : benefit(2) {
  %input1 = operand
  %input2 = operand
  %type = type
  %op1 = operation(%input1 : !pdl.value) -> (%type : !pdl.type)
  %val1 = result 0 of %op1
  %root1 = operation(%val1 : !pdl.value)
  %op2 = operation(%input2 : !pdl.value) -> (%type : !pdl.type)
  %val2 = result 0 of %op2
  %root2 = operation(%val1, %val2 : !pdl.value, !pdl.value)
  rewrite with "rewriter"(%root1, %root2 : !pdl.operation, !pdl.operation)
}

// -----

pdl.pattern @rewrite_multi_root_forced : benefit(2) {
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

// -----

// Check that the result type of an operation within a rewrite can be inferred
// from a pdl.replace.
pdl.pattern @infer_type_from_operation_replace : benefit(1) {
  %type1 = type : i32
  %type2 = type
  %root = operation -> (%type1, %type2 : !pdl.type, !pdl.type)
  rewrite %root {
    %type3 = type
    %newOp = operation "foo.op" -> (%type1, %type3 : !pdl.type, !pdl.type)
    replace %root with %newOp
  }
}

// -----

// Check that the result type of an operation within a rewrite can be inferred
// from the result types of an operation within the match block.
pdl.pattern @infer_type_from_type_used_in_match : benefit(1) {
  %type1 = type : i32
  %type2 = type
  %root = operation -> (%type1, %type2 : !pdl.type, !pdl.type)
  rewrite %root {
    %newOp = operation "foo.op" -> (%type1, %type2 : !pdl.type, !pdl.type)
  }
}

// -----

// Check that the result type of an operation within a rewrite can be inferred
// from the result types of an operation within the match block.
pdl.pattern @infer_type_from_type_used_in_match : benefit(1) {
  %types = types
  %root = operation -> (%types : !pdl.range<type>)
  rewrite %root {
    %otherTypes = types : [i32, i64]
    %newOp = operation "foo.op" -> (%types, %otherTypes : !pdl.range<type>, !pdl.range<type>)
  }
}

// -----

// Check that the result type of an operation within a rewrite can be inferred
// from the type of an operand within the match block.
pdl.pattern @infer_type_from_type_used_in_match : benefit(1) {
  %type1 = type
  %type2 = type
  %operand1 = operand : %type1
  %operand2 = operand : %type2
  %root = operation (%operand1, %operand2 : !pdl.value, !pdl.value)
  rewrite %root {
    %newOp = operation "foo.op" -> (%type1, %type2 : !pdl.type, !pdl.type)
  }
}

// -----

// Check that the result type of an operation within a rewrite can be inferred
// from the types of operands within the match block.
pdl.pattern @infer_type_from_type_used_in_match : benefit(1) {
  %types = types
  %operands = operands : %types
  %root = operation (%operands : !pdl.range<value>)
  rewrite %root {
    %newOp = operation "foo.op" -> (%types : !pdl.range<type>)
  }
}

// -----

pdl.pattern @apply_rewrite_with_no_results : benefit(1) {
  %root = operation
  rewrite %root {
    apply_native_rewrite "NativeRewrite"(%root : !pdl.operation)
  }
}

// -----

pdl.pattern @attribute_with_dict : benefit(1) {
  %root = operation
  rewrite %root {
    %attr = attribute = {some_unit_attr} attributes {pdl.special_attribute}
    apply_native_rewrite "NativeRewrite"(%attr : !pdl.attribute)
  }
}

// -----

// Check that we don't treat the trailing location of a pdl.attribute as the
// attribute value.

pdl.pattern @attribute_with_loc : benefit(1) {
  // CHECK-GENERIC: "pdl.attribute"
  // CHECK-GENERIC-NOT: value = loc
  %attr = attribute loc("bar")

  %root = operation {"attribute" = %attr}
  rewrite %root with "rewriter"
}
