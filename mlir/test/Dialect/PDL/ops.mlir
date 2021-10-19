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
