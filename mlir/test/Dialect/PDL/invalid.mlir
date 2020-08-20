// RUN: mlir-opt %s -split-input-file -verify-diagnostics

//===----------------------------------------------------------------------===//
// pdl::ApplyConstraintOp
//===----------------------------------------------------------------------===//

pdl.pattern : benefit(1) {
  %op = pdl.operation "foo.op"

  // expected-error@below {{expected at least one argument}}
  "pdl.apply_constraint"() {name = "foo", params = []} : () -> ()
  pdl.rewrite "rewriter"(%op)
}

// -----

//===----------------------------------------------------------------------===//
// pdl::AttributeOp
//===----------------------------------------------------------------------===//

pdl.pattern : benefit(1) {
  %type = pdl.type

  // expected-error@below {{expected only one of [`type`, `value`] to be set}}
  %attr = pdl.attribute : %type 10

  %op, %result = pdl.operation "foo.op" {"attr" = %attr} -> %type
  pdl.rewrite "rewriter"(%op)
}

// -----

pdl.pattern : benefit(1) {
  %op = pdl.operation "foo.op"
  pdl.rewrite(%op) {
    %type = pdl.type

    // expected-error@below {{expected constant value when specified within a `pdl.rewrite`}}
    %attr = pdl.attribute : %type
  }
}

// -----

pdl.pattern : benefit(1) {
  %op = pdl.operation "foo.op"
  pdl.rewrite(%op) {
    // expected-error@below {{expected constant value when specified within a `pdl.rewrite`}}
    %attr = pdl.attribute
  }
}

// -----

pdl.pattern : benefit(1) {
  // expected-error@below {{expected a bindable (i.e. `pdl.operation`) user when defined in the matcher body of a `pdl.pattern`}}
  %unused = pdl.attribute

  %op = pdl.operation "foo.op"
  pdl.rewrite "rewriter"(%op)
}

// -----

//===----------------------------------------------------------------------===//
// pdl::InputOp
//===----------------------------------------------------------------------===//

pdl.pattern : benefit(1) {
  // expected-error@below {{expected a bindable (i.e. `pdl.operation`) user when defined in the matcher body of a `pdl.pattern`}}
  %unused = pdl.input

  %op = pdl.operation "foo.op"
  pdl.rewrite "rewriter"(%op)
}

// -----

//===----------------------------------------------------------------------===//
// pdl::OperationOp
//===----------------------------------------------------------------------===//

pdl.pattern : benefit(1) {
  %op = pdl.operation "foo.op"
  pdl.rewrite(%op) {
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
  pdl.rewrite "rewriter"(%op)
}

// -----

pdl.pattern : benefit(1) {
  %op = pdl.operation "foo.op"()
  pdl.rewrite (%op) {
    %type = pdl.type

    // expected-error@below {{op must have inferable or constrained result types when nested within `pdl.rewrite`}}
    // expected-note@below {{result type #0 was not constrained}}
    %newOp, %result = pdl.operation "foo.op" -> %type
  }
}

// -----

pdl.pattern : benefit(1) {
  // expected-error@below {{expected a bindable (i.e. `pdl.operation` or `pdl.rewrite`) user when defined in the matcher body of a `pdl.pattern`}}
  %unused = pdl.operation "foo.op"

  %op = pdl.operation "foo.op"
  pdl.rewrite "rewriter"(%op)
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

// expected-error@below {{expected only `pdl` operations within the pattern body}}
pdl.pattern : benefit(1) {
  // expected-note@below {{see non-`pdl` operation defined here}}
  "foo.other_op"() : () -> ()

  %root = pdl.operation "foo.op"
  pdl.rewrite "foo"(%root)
}

// -----

//===----------------------------------------------------------------------===//
// pdl::ReplaceOp
//===----------------------------------------------------------------------===//

pdl.pattern : benefit(1) {
  %root = pdl.operation "foo.op"
  pdl.rewrite (%root) {
    %type = pdl.type : i32
    %newOp, %newResult = pdl.operation "foo.op" -> %type

    // expected-error@below {{to have the same number of results as the replacement operation}}
    pdl.replace %root with %newOp
  }
}

// -----

pdl.pattern : benefit(1) {
  %type = pdl.type : i32
  %root, %oldResult = pdl.operation "foo.op" -> %type
  pdl.rewrite (%root) {
    %newOp, %newResult = pdl.operation "foo.op" -> %type

    // expected-error@below {{expected no replacement values to be provided when the replacement operation is present}}
    "pdl.replace"(%root, %newOp, %newResult) {
      operand_segment_sizes = dense<1> : vector<3xi32>
    } : (!pdl.operation, !pdl.operation, !pdl.value) -> ()
  }
}

// -----

pdl.pattern : benefit(1) {
  %root = pdl.operation "foo.op"
  pdl.rewrite (%root) {
    %type = pdl.type : i32
    %newOp, %newResult = pdl.operation "foo.op" -> %type

    // expected-error@below {{to have the same number of results as the provided replacement values}}
    pdl.replace %root with (%newResult)
  }
}

// -----

//===----------------------------------------------------------------------===//
// pdl::TypeOp
//===----------------------------------------------------------------------===//

pdl.pattern : benefit(1) {
  // expected-error@below {{expected a bindable (i.e. `pdl.attribute`, `pdl.input`, or `pdl.operation`) user when defined in the matcher body of a `pdl.pattern`}}
  %unused = pdl.type

  %op = pdl.operation "foo.op"
  pdl.rewrite "rewriter"(%op)
}
