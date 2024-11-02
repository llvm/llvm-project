// RUN: mlir-opt %s -split-input-file -verify-diagnostics

//===----------------------------------------------------------------------===//
// pdl::CreateOperationOp
//===----------------------------------------------------------------------===//

pdl_interp.func @rewriter() {
  // expected-error@+1 {{op has inferred results, but the created operation 'foo.op' does not support result type inference}}
  %op = pdl_interp.create_operation "foo.op" -> <inferred>
  pdl_interp.finalize
}

// -----

pdl_interp.func @rewriter() {
  %type = pdl_interp.create_type i32
  // expected-error@+1 {{op with inferred results cannot also have explicit result types}}
  %op = "pdl_interp.create_operation"(%type) {
    inferredResultTypes,
    inputAttributeNames = [],
    name = "foo.op",
    operand_segment_sizes = dense<[0, 0, 1]> : vector<3xi32>
  } : (!pdl.type) -> (!pdl.operation)
  pdl_interp.finalize
}

