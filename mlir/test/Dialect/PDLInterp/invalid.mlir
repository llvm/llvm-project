// RUN: mlir-opt %s -split-input-file -verify-diagnostics

//===----------------------------------------------------------------------===//
// pdl_interp::CreateOperationOp
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
    operand_segment_sizes = array<i32: 0, 0, 1>
  } : (!pdl.type) -> (!pdl.operation)
  pdl_interp.finalize
}

// -----

//===----------------------------------------------------------------------===//
// pdl_interp::CreateRangeOp
//===----------------------------------------------------------------------===//

pdl_interp.func @rewriter(%value: !pdl.value, %type: !pdl.type) {
  // expected-error @below {{expected operand to have element type '!pdl.value', but got '!pdl.type'}}
  %range = pdl_interp.create_range %value, %type : !pdl.value, !pdl.type
  pdl_interp.finalize
}
