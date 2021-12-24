// RUN: mlir-opt -split-input-file %s | mlir-opt
// Verify the generic form can be parsed.
// RUN: mlir-opt -split-input-file -mlir-print-op-generic %s | mlir-opt

// -----

pdl.pattern @operations : benefit(1) {
  // Operation with attributes and results.
  %attribute = pdl.attribute
  %type = pdl.type
  %op0 = pdl.operation {"attr" = %attribute} -> (%type : !pdl.type)
  %op0_result = pdl.result 0 of %op0

  // Operation with input.
  %input = pdl.operand
  %root = pdl.operation(%op0_result, %input : !pdl.value, !pdl.value)
  pdl.rewrite %root with "rewriter"
}

// -----

pdl.pattern @rewrite_with_args : benefit(1) {
  %input = pdl.operand
  %root = pdl.operation(%input : !pdl.value)
  pdl.rewrite %root with "rewriter"(%input : !pdl.value)
}

// -----

pdl.pattern @rewrite_with_params : benefit(1) {
  %root = pdl.operation
  pdl.rewrite %root with "rewriter"["I am param"]
}

// -----

pdl.pattern @rewrite_with_args_and_params : benefit(1) {
  %input = pdl.operand
  %root = pdl.operation(%input : !pdl.value)
  pdl.rewrite %root with "rewriter"["I am param"](%input : !pdl.value)
}

// -----

pdl.pattern @rewrite_multi_root_optimal : benefit(2) {
  %input1 = pdl.operand
  %input2 = pdl.operand
  %type = pdl.type
  %op1 = pdl.operation(%input1 : !pdl.value) -> (%type : !pdl.type)
  %val1 = pdl.result 0 of %op1
  %root1 = pdl.operation(%val1 : !pdl.value)
  %op2 = pdl.operation(%input2 : !pdl.value) -> (%type : !pdl.type)
  %val2 = pdl.result 0 of %op2
  %root2 = pdl.operation(%val1, %val2 : !pdl.value, !pdl.value)
  pdl.rewrite with "rewriter"["I am param"](%root1, %root2 : !pdl.operation, !pdl.operation)
}

// -----

pdl.pattern @rewrite_multi_root_forced : benefit(2) {
  %input1 = pdl.operand
  %input2 = pdl.operand
  %type = pdl.type
  %op1 = pdl.operation(%input1 : !pdl.value) -> (%type : !pdl.type)
  %val1 = pdl.result 0 of %op1
  %root1 = pdl.operation(%val1 : !pdl.value)
  %op2 = pdl.operation(%input2 : !pdl.value) -> (%type : !pdl.type)
  %val2 = pdl.result 0 of %op2
  %root2 = pdl.operation(%val1, %val2 : !pdl.value, !pdl.value)
  pdl.rewrite %root1 with "rewriter"["I am param"](%root2 : !pdl.operation)
}

// -----

// Check that the result type of an operation within a rewrite can be inferred
// from a pdl.replace.
pdl.pattern @infer_type_from_operation_replace : benefit(1) {
  %type1 = pdl.type : i32
  %type2 = pdl.type
  %root = pdl.operation -> (%type1, %type2 : !pdl.type, !pdl.type)
  pdl.rewrite %root {
    %type3 = pdl.type
    %newOp = pdl.operation "foo.op" -> (%type1, %type3 : !pdl.type, !pdl.type)
    pdl.replace %root with %newOp
  }
}

// -----

// Check that the result type of an operation within a rewrite can be inferred
// from types used within the match block.
pdl.pattern @infer_type_from_type_used_in_match : benefit(1) {
  %type1 = pdl.type : i32
  %type2 = pdl.type
  %root = pdl.operation -> (%type1, %type2 : !pdl.type, !pdl.type)
  pdl.rewrite %root {
    %newOp = pdl.operation "foo.op" -> (%type1, %type2 : !pdl.type, !pdl.type)
  }
}

// -----

// Check that the result type of an operation within a rewrite can be inferred
// from types used within the match block.
pdl.pattern @infer_type_from_type_used_in_match : benefit(1) {
  %types = pdl.types
  %root = pdl.operation -> (%types : !pdl.range<type>)
  pdl.rewrite %root {
    %otherTypes = pdl.types : [i32, i64]
    %newOp = pdl.operation "foo.op" -> (%types, %otherTypes : !pdl.range<type>, !pdl.range<type>)
  }
}

// -----

pdl.pattern @apply_rewrite_with_no_results : benefit(1) {
  %root = pdl.operation
  pdl.rewrite %root {
    pdl.apply_native_rewrite "NativeRewrite"(%root : !pdl.operation)
  }
}

// -----

pdl.pattern @attribute_with_dict : benefit(1) {
  %root = pdl.operation
  pdl.rewrite %root {
    %attr = pdl.attribute {some_unit_attr} attributes {pdl.special_attribute}
    pdl.apply_native_rewrite "NativeRewrite"(%attr : !pdl.attribute)
  }
}
