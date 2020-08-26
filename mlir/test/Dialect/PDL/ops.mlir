// RUN: mlir-opt -split-input-file %s | mlir-opt
// Verify the generic form can be parsed.
// RUN: mlir-opt -split-input-file -mlir-print-op-generic %s | mlir-opt

// -----

pdl.pattern @operations : benefit(1) {
  // Operation with attributes and results.
  %attribute = pdl.attribute
  %type = pdl.type
  %op0, %op0_result = pdl.operation {"attr" = %attribute} -> %type

  // Operation with input.
  %input = pdl.input
  %root = pdl.operation(%op0_result, %input)
  pdl.rewrite %root with "rewriter"
}

// -----

pdl.pattern @rewrite_with_args : benefit(1) {
  %input = pdl.input
  %root = pdl.operation(%input)
  pdl.rewrite %root with "rewriter"(%input : !pdl.value)
}

// -----

pdl.pattern @rewrite_with_params : benefit(1) {
  %root = pdl.operation
  pdl.rewrite %root with "rewriter"["I am param"]
}

// -----

pdl.pattern @rewrite_with_args_and_params : benefit(1) {
  %input = pdl.input
  %root = pdl.operation(%input)
  pdl.rewrite %root with "rewriter"["I am param"](%input : !pdl.value)
}

// -----

// Check that the result type of an operation within a rewrite can be inferred
// from a pdl.replace.
pdl.pattern @infer_type_from_operation_replace : benefit(1) {
  %type1 = pdl.type : i32
  %type2 = pdl.type
  %root, %results:2 = pdl.operation -> %type1, %type2
  pdl.rewrite %root {
    %type3 = pdl.type
    %newOp, %newResults:2 = pdl.operation "foo.op" -> %type1, %type3
    pdl.replace %root with %newOp
  }
}

// -----

// Check that the result type of an operation within a rewrite can be inferred
// from a pdl.replace.
pdl.pattern @infer_type_from_result_replace : benefit(1) {
  %type1 = pdl.type : i32
  %type2 = pdl.type
  %root, %results:2 = pdl.operation -> %type1, %type2
  pdl.rewrite %root {
    %type3 = pdl.type
    %newOp, %newResults:2 = pdl.operation "foo.op" -> %type1, %type3
    pdl.replace %root with (%newResults#0, %newResults#1)
  }
}

// -----

// Check that the result type of an operation within a rewrite can be inferred
// from a pdl.replace.
pdl.pattern @infer_type_from_type_used_in_match : benefit(1) {
  %type1 = pdl.type : i32
  %type2 = pdl.type
  %root, %results:2 = pdl.operation -> %type1, %type2
  pdl.rewrite %root {
    %newOp, %newResults:2 = pdl.operation "foo.op" -> %type1, %type2
  }
}
