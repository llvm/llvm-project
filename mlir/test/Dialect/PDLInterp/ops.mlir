// RUN: mlir-opt -split-input-file %s | mlir-opt
// Verify the printed output can be parsed.
// RUN: mlir-opt %s | mlir-opt
// Verify the generic form can be parsed.
// RUN: mlir-opt -mlir-print-op-generic %s | mlir-opt

// -----

func @operations(%attribute: !pdl.attribute,
                 %input: !pdl.value,
                 %type: !pdl.type) {
  // attributes, operands, and results
  %op0 = pdl_interp.create_operation "foo.op"(%input) {"attr" = %attribute} -> %type

  // attributes, and results
  %op1 = pdl_interp.create_operation "foo.op"() {"attr" = %attribute} -> %type

  // attributes
  %op2 = pdl_interp.create_operation "foo.op"() {"attr" = %attribute, "attr1" = %attribute} -> ()

  // operands, and results
  %op3 = pdl_interp.create_operation "foo.op"(%input) -> %type

  pdl_interp.finalize
}
