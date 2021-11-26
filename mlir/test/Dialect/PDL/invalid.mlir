// RUN: mlir-opt %s -split-input-file -verify-diagnostics

//===----------------------------------------------------------------------===//
// pdl::ApplyNativeConstraintOp
//===----------------------------------------------------------------------===//

pdl.pattern : benefit(1) {
  %op = pdl.operation "foo.op"

  // expected-error@below {{expected at least one argument}}
  "pdl.apply_native_constraint"() {name = "foo", params = []} : () -> ()
  pdl.rewrite %op with "rewriter"
}

// -----

//===----------------------------------------------------------------------===//
// pdl::ApplyNativeRewriteOp
//===----------------------------------------------------------------------===//

pdl.pattern : benefit(1) {
  %op = pdl.operation "foo.op"
  pdl.rewrite %op {
    // expected-error@below {{expected at least one argument}}
    "pdl.apply_native_rewrite"() {name = "foo", params = []} : () -> ()
  }
}

// -----

//===----------------------------------------------------------------------===//
// pdl::AttributeOp
//===----------------------------------------------------------------------===//

pdl.pattern : benefit(1) {
  %type = pdl.type

  // expected-error@below {{expected only one of [`type`, `value`] to be set}}
  %attr = pdl.attribute : %type 10

  %op = pdl.operation "foo.op" {"attr" = %attr} -> (%type : !pdl.type)
  pdl.rewrite %op with "rewriter"
}

// -----

pdl.pattern : benefit(1) {
  %op = pdl.operation "foo.op"
  pdl.rewrite %op {
    %type = pdl.type

    // expected-error@below {{expected constant value when specified within a `pdl.rewrite`}}
    %attr = pdl.attribute : %type
  }
}

// -----

pdl.pattern : benefit(1) {
  %op = pdl.operation "foo.op"
  pdl.rewrite %op {
    // expected-error@below {{expected constant value when specified within a `pdl.rewrite`}}
    %attr = pdl.attribute
  }
}

// -----

pdl.pattern : benefit(1) {
  // expected-error@below {{expected a bindable user when defined in the matcher body of a `pdl.pattern`}}
  %unused = pdl.attribute

  %op = pdl.operation "foo.op"
  pdl.rewrite %op with "rewriter"
}

// -----

//===----------------------------------------------------------------------===//
// pdl::OperandOp
//===----------------------------------------------------------------------===//

pdl.pattern : benefit(1) {
  // expected-error@below {{expected a bindable user when defined in the matcher body of a `pdl.pattern`}}
  %unused = pdl.operand

  %op = pdl.operation "foo.op"
  pdl.rewrite %op with "rewriter"
}

// -----

//===----------------------------------------------------------------------===//
// pdl::OperandsOp
//===----------------------------------------------------------------------===//

pdl.pattern : benefit(1) {
  // expected-error@below {{expected a bindable user when defined in the matcher body of a `pdl.pattern`}}
  %unused = pdl.operands

  %op = pdl.operation "foo.op"
  pdl.rewrite %op with "rewriter"
}

// -----

//===----------------------------------------------------------------------===//
// pdl::OperationOp
//===----------------------------------------------------------------------===//

pdl.pattern : benefit(1) {
  %op = pdl.operation "foo.op"
  pdl.rewrite %op {
    // expected-error@below {{must have an operation name when nested within a `pdl.rewrite`}}
    %newOp = pdl.operation
  }
}

// -----

pdl.pattern : benefit(1) {
  // expected-error@below {{expected the same number of attribute values and attribute names, got 1 names and 0 values}}
  %op = "pdl.operation"() {
    attributeNames = ["attr"],
    operand_segment_sizes = dense<0> : vector<3xi32>
  } : () -> (!pdl.operation)
  pdl.rewrite %op with "rewriter"
}

// -----

pdl.pattern : benefit(1) {
  %op = pdl.operation "foo.op"
  pdl.rewrite %op {
    %type = pdl.type

    // expected-error@below {{op must have inferable or constrained result types when nested within `pdl.rewrite`}}
    // expected-note@below {{result type #0 was not constrained}}
    %newOp = pdl.operation "foo.op" -> (%type : !pdl.type)
  }
}

// -----

pdl.pattern : benefit(1) {
  // expected-error@below {{expected a bindable user when defined in the matcher body of a `pdl.pattern`}}
  %unused = pdl.operation "foo.op"

  %op = pdl.operation "foo.op"
  pdl.rewrite %op with "rewriter"
}

// -----

//===----------------------------------------------------------------------===//
// pdl::PatternOp
//===----------------------------------------------------------------------===//

// expected-error@below {{expected body to terminate with `pdl.rewrite`}}
pdl.pattern : benefit(1) {
  // expected-note@below {{see terminator defined here}}
  return
}

// -----

// expected-error@below {{the pattern must contain at least one `pdl.operation`}}
pdl.pattern : benefit(1) {
  pdl.rewrite with "foo"
}

// -----
// expected-error@below {{expected only `pdl` operations within the pattern body}}
pdl.pattern : benefit(1) {
  // expected-note@below {{see non-`pdl` operation defined here}}
  "test.foo.other_op"() : () -> ()

  %root = pdl.operation "foo.op"
  pdl.rewrite %root with "foo"
}

// -----
// expected-error@below {{the operations must form a connected component}}
pdl.pattern : benefit(1) {
  %op1 = pdl.operation "foo.op"
  %op2 = pdl.operation "bar.op"
  // expected-note@below {{see a disconnected value / operation here}}
  %val = pdl.result 0 of %op2
  pdl.rewrite %op1 with "foo"(%val : !pdl.value)
}

// -----
// expected-error@below {{the operations must form a connected component}}
pdl.pattern : benefit(1) {
  %type = pdl.type
  %op1 = pdl.operation "foo.op" -> (%type : !pdl.type)
  %val = pdl.result 0 of %op1
  %op2 = pdl.operation "bar.op"(%val : !pdl.value)
  // expected-note@below {{see a disconnected value / operation here}}
  %op3 = pdl.operation "baz.op"
  pdl.rewrite {
    pdl.erase %op1
    pdl.erase %op2
    pdl.erase %op3
  }
}

// -----

pdl.pattern : benefit(1) {
  %type = pdl.type : i32
  %root = pdl.operation "foo.op" -> (%type : !pdl.type)
  pdl.rewrite %root {
    %newOp = pdl.operation "foo.op" -> (%type : !pdl.type)
    %newResult = pdl.result 0 of %newOp

    // expected-error@below {{expected no replacement values to be provided when the replacement operation is present}}
    "pdl.replace"(%root, %newOp, %newResult) {
      operand_segment_sizes = dense<1> : vector<3xi32>
    } : (!pdl.operation, !pdl.operation, !pdl.value) -> ()
  }
}

// -----

//===----------------------------------------------------------------------===//
// pdl::ResultsOp
//===----------------------------------------------------------------------===//

pdl.pattern : benefit(1) {
  %root = pdl.operation "foo.op"
  // expected-error@below {{expected `pdl.range<value>` result type when no index is specified, but got: '!pdl.value'}}
  %results = "pdl.results"(%root) : (!pdl.operation) -> !pdl.value
  pdl.rewrite %root with "rewriter"
}

// -----

//===----------------------------------------------------------------------===//
// pdl::RewriteOp
//===----------------------------------------------------------------------===//

pdl.pattern : benefit(1) {
  %op = pdl.operation "foo.op"

  // expected-error@below {{expected rewrite region to be non-empty if external name is not specified}}
  "pdl.rewrite"(%op) ({}) {
    operand_segment_sizes = dense<[1,0]> : vector<2xi32>
  } : (!pdl.operation) -> ()
}

// -----

pdl.pattern : benefit(1) {
  %op = pdl.operation "foo.op"

  // expected-error@below {{expected no external arguments when the rewrite is specified inline}}
  "pdl.rewrite"(%op, %op) ({
    ^bb1:
  }) {
    operand_segment_sizes = dense<1> : vector<2xi32>
  }: (!pdl.operation, !pdl.operation) -> ()
}

// -----

pdl.pattern : benefit(1) {
  %op = pdl.operation "foo.op"

  // expected-error@below {{expected no external constant parameters when the rewrite is specified inline}}
  "pdl.rewrite"(%op) ({
    ^bb1:
  }) {
    operand_segment_sizes = dense<[1,0]> : vector<2xi32>,
    externalConstParams = []} : (!pdl.operation) -> ()
}

// -----

pdl.pattern : benefit(1) {
  %op = pdl.operation "foo.op"

  // expected-error@below {{expected rewrite region to be empty when rewrite is external}}
  "pdl.rewrite"(%op) ({
    ^bb1:
  }) {
    name = "foo",
    operand_segment_sizes = dense<[1,0]> : vector<2xi32>
  } : (!pdl.operation) -> ()
}

// -----

//===----------------------------------------------------------------------===//
// pdl::TypeOp
//===----------------------------------------------------------------------===//

pdl.pattern : benefit(1) {
  // expected-error@below {{expected a bindable user when defined in the matcher body of a `pdl.pattern`}}
  %unused = pdl.type

  %op = pdl.operation "foo.op"
  pdl.rewrite %op with "rewriter"
}

// -----

//===----------------------------------------------------------------------===//
// pdl::TypesOp
//===----------------------------------------------------------------------===//

pdl.pattern : benefit(1) {
  // expected-error@below {{expected a bindable user when defined in the matcher body of a `pdl.pattern`}}
  %unused = pdl.types

  %op = pdl.operation "foo.op"
  pdl.rewrite %op with "rewriter"
}
