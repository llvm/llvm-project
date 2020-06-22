// RUN: mlir-tblgen -gen-op-decls -asmformat-error-is-fatal=false -I %S/../../include %s -o=%t 2>&1 | FileCheck %s

// This file contains tests for the specification of the declarative op format.

include "mlir/IR/OpBase.td"

def TestDialect : Dialect {
  let name = "test";
}
class TestFormat_Op<string name, string fmt, list<OpTrait> traits = []>
    : Op<TestDialect, name, traits> {
  let assemblyFormat = fmt;
}

//===----------------------------------------------------------------------===//
// Directives
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// attr-dict

// CHECK: error: 'attr-dict' directive not found
def DirectiveAttrDictInvalidA : TestFormat_Op<"attrdict_invalid_a", [{
}]>;
// CHECK: error: 'attr-dict' directive has already been seen
def DirectiveAttrDictInvalidB : TestFormat_Op<"attrdict_invalid_b", [{
  attr-dict attr-dict
}]>;
// CHECK: error: 'attr-dict' directive has already been seen
def DirectiveAttrDictInvalidC : TestFormat_Op<"attrdict_invalid_c", [{
  attr-dict attr-dict-with-keyword
}]>;
// CHECK: error: 'attr-dict' directive can only be used as a top-level directive
def DirectiveAttrDictInvalidD : TestFormat_Op<"attrdict_invalid_d", [{
  type(attr-dict)
}]>;
// CHECK-NOT: error
def DirectiveAttrDictValidA : TestFormat_Op<"attrdict_valid_a", [{
  attr-dict
}]>;
def DirectiveAttrDictValidB : TestFormat_Op<"attrdict_valid_b", [{
  attr-dict-with-keyword
}]>;

//===----------------------------------------------------------------------===//
// functional-type

// CHECK: error: 'functional-type' is only valid as a top-level directive
def DirectiveFunctionalTypeInvalidA : TestFormat_Op<"functype_invalid_a", [{
  functional-type(functional-type)
}]>;
// CHECK: error: expected '(' before argument list
def DirectiveFunctionalTypeInvalidB : TestFormat_Op<"functype_invalid_b", [{
  functional-type
}]>;
// CHECK: error: expected directive, literal, variable, or optional group
def DirectiveFunctionalTypeInvalidC : TestFormat_Op<"functype_invalid_c", [{
  functional-type(
}]>;
// CHECK: error: expected ',' after inputs argument
def DirectiveFunctionalTypeInvalidD : TestFormat_Op<"functype_invalid_d", [{
  functional-type(operands
}]>;
// CHECK: error: expected directive, literal, variable, or optional group
def DirectiveFunctionalTypeInvalidE : TestFormat_Op<"functype_invalid_e", [{
  functional-type(operands,
}]>;
// CHECK: error: expected ')' after argument list
def DirectiveFunctionalTypeInvalidF : TestFormat_Op<"functype_invalid_f", [{
  functional-type(operands, results
}]>;
// CHECK-NOT: error
def DirectiveFunctionalTypeValid : TestFormat_Op<"functype_invalid_a", [{
  functional-type(operands, results) attr-dict
}]>;

//===----------------------------------------------------------------------===//
// operands

// CHECK: error: 'operands' directive creates overlap in format
def DirectiveOperandsInvalidA : TestFormat_Op<"operands_invalid_a", [{
  operands operands
}]>;
// CHECK: error: 'operands' directive creates overlap in format
def DirectiveOperandsInvalidB : TestFormat_Op<"operands_invalid_b", [{
  $operand operands
}]>, Arguments<(ins I64:$operand)>;
// CHECK-NOT: error:
def DirectiveOperandsValid : TestFormat_Op<"operands_valid", [{
  operands attr-dict
}]>;

//===----------------------------------------------------------------------===//
// results

// CHECK: error: 'results' directive can not be used as a top-level directive
def DirectiveResultsInvalidA : TestFormat_Op<"results_invalid_a", [{
  results
}]>;

//===----------------------------------------------------------------------===//
// successors

// CHECK: error: 'successors' is only valid as a top-level directive
def DirectiveSuccessorsInvalidA : TestFormat_Op<"successors_invalid_a", [{
  type(successors)
}]>;

//===----------------------------------------------------------------------===//
// type

// CHECK: error: expected '(' before argument list
def DirectiveTypeInvalidA : TestFormat_Op<"type_invalid_a", [{
  type
}]>;
// CHECK: error: expected directive, literal, variable, or optional group
def DirectiveTypeInvalidB : TestFormat_Op<"type_invalid_b", [{
  type(
}]>;
// CHECK: error: expected ')' after argument list
def DirectiveTypeInvalidC : TestFormat_Op<"type_invalid_c", [{
  type(operands
}]>;
// CHECK-NOT: error:
def DirectiveTypeValid : TestFormat_Op<"type_valid", [{
  type(operands) attr-dict
}]>;

//===----------------------------------------------------------------------===//
// functional-type/type operands

// CHECK: error: 'type' directive operand expects variable or directive operand
def DirectiveTypeZOperandInvalidA : TestFormat_Op<"type_operand_invalid_a", [{
  type(`literal`)
}]>;
// CHECK: error: 'operands' 'type' is already bound
def DirectiveTypeZOperandInvalidB : TestFormat_Op<"type_operand_invalid_b", [{
  type(operands) type(operands)
}]>;
// CHECK: error: 'operands' 'type' is already bound
def DirectiveTypeZOperandInvalidC : TestFormat_Op<"type_operand_invalid_c", [{
  type($operand) type(operands)
}]>, Arguments<(ins I64:$operand)>;
// CHECK: error: 'type' of 'operand' is already bound
def DirectiveTypeZOperandInvalidD : TestFormat_Op<"type_operand_invalid_d", [{
  type(operands) type($operand)
}]>, Arguments<(ins I64:$operand)>;
// CHECK: error: 'type' of 'operand' is already bound
def DirectiveTypeZOperandInvalidE : TestFormat_Op<"type_operand_invalid_e", [{
  type($operand) type($operand)
}]>, Arguments<(ins I64:$operand)>;
// CHECK: error: 'results' 'type' is already bound
def DirectiveTypeZOperandInvalidF : TestFormat_Op<"type_operand_invalid_f", [{
  type(results) type(results)
}]>;
// CHECK: error: 'results' 'type' is already bound
def DirectiveTypeZOperandInvalidG : TestFormat_Op<"type_operand_invalid_g", [{
  type($result) type(results)
}]>, Results<(outs I64:$result)>;
// CHECK: error: 'type' of 'result' is already bound
def DirectiveTypeZOperandInvalidH : TestFormat_Op<"type_operand_invalid_h", [{
  type(results) type($result)
}]>, Results<(outs I64:$result)>;
// CHECK: error: 'type' of 'result' is already bound
def DirectiveTypeZOperandInvalidI : TestFormat_Op<"type_operand_invalid_i", [{
  type($result) type($result)
}]>, Results<(outs I64:$result)>;
// CHECK-NOT: error:
def DirectiveTypeZOperandValid : TestFormat_Op<"type_operand_valid", [{
  type(operands) type(results) attr-dict
}]>;

//===----------------------------------------------------------------------===//
// Literals
//===----------------------------------------------------------------------===//

// Test all of the valid literals.
// CHECK: error: expected valid literal
def LiteralInvalidA : TestFormat_Op<"literal_invalid_a", [{
  `1`
}]>;
// CHECK: error: unexpected end of file in literal
// CHECK: error: expected directive, literal, variable, or optional group
def LiteralInvalidB : TestFormat_Op<"literal_invalid_b", [{
  `
}]>;
// CHECK-NOT: error
def LiteralValid : TestFormat_Op<"literal_valid", [{
  `_` `:` `,` `=` `<` `>` `(` `)` `[` `]` `->` `abc$._`
  attr-dict
}]>;

//===----------------------------------------------------------------------===//
// Optional Groups
//===----------------------------------------------------------------------===//

// CHECK: error: optional groups can only be used as top-level elements
def OptionalInvalidA : TestFormat_Op<"optional_invalid_a", [{
  type(($attr^)?) attr-dict
}]>, Arguments<(ins OptionalAttr<I64Attr>:$attr)>;
// CHECK: error: expected directive, literal, variable, or optional group
def OptionalInvalidB : TestFormat_Op<"optional_invalid_b", [{
  () attr-dict
}]>, Arguments<(ins OptionalAttr<I64Attr>:$attr)>;
// CHECK: error: optional group specified no anchor element
def OptionalInvalidC : TestFormat_Op<"optional_invalid_c", [{
  ($attr)? attr-dict
}]>, Arguments<(ins OptionalAttr<I64Attr>:$attr)>;
// CHECK: error: first element of an operand group must be a literal or operand
def OptionalInvalidD : TestFormat_Op<"optional_invalid_d", [{
  ($attr^)? attr-dict
}]>, Arguments<(ins OptionalAttr<I64Attr>:$attr)>;
// CHECK: error: type directive can only refer to variables within the optional group
def OptionalInvalidE : TestFormat_Op<"optional_invalid_e", [{
  (`,` $attr^ type(operands))? attr-dict
}]>, Arguments<(ins OptionalAttr<I64Attr>:$attr)>;
// CHECK: error: only one element can be marked as the anchor of an optional group
def OptionalInvalidF : TestFormat_Op<"optional_invalid_f", [{
  ($attr^ $attr2^) attr-dict
}]>, Arguments<(ins OptionalAttr<I64Attr>:$attr, OptionalAttr<I64Attr>:$attr2)>;
// CHECK: error: only optional attributes can be used to anchor an optional group
def OptionalInvalidG : TestFormat_Op<"optional_invalid_g", [{
  ($attr^) attr-dict
}]>, Arguments<(ins I64Attr:$attr)>;
// CHECK: error: only variable length operands can be used within an optional group
def OptionalInvalidH : TestFormat_Op<"optional_invalid_h", [{
  ($arg^) attr-dict
}]>, Arguments<(ins I64:$arg)>;
// CHECK: error: only variables can be used to anchor an optional group
def OptionalInvalidI : TestFormat_Op<"optional_invalid_i", [{
  ($arg type($arg)^) attr-dict
}]>, Arguments<(ins Variadic<I64>:$arg)>;
// CHECK: error: only literals, types, and variables can be used within an optional group
def OptionalInvalidJ : TestFormat_Op<"optional_invalid_j", [{
  (attr-dict)
}]>;
// CHECK: error: expected '?' after optional group
def OptionalInvalidK : TestFormat_Op<"optional_invalid_k", [{
  ($arg^)
}]>, Arguments<(ins Variadic<I64>:$arg)>;

//===----------------------------------------------------------------------===//
// Variables
//===----------------------------------------------------------------------===//

// CHECK: error: expected variable to refer to an argument, result, or successor
def VariableInvalidA : TestFormat_Op<"variable_invalid_a", [{
  $unknown_arg attr-dict
}]>;
// CHECK: error: attribute 'attr' is already bound
def VariableInvalidB : TestFormat_Op<"variable_invalid_b", [{
  $attr $attr attr-dict
}]>, Arguments<(ins I64Attr:$attr)>;
// CHECK: error: operand 'operand' is already bound
def VariableInvalidC : TestFormat_Op<"variable_invalid_c", [{
  $operand $operand attr-dict
}]>, Arguments<(ins I64:$operand)>;
// CHECK: error: operand 'operand' is already bound
def VariableInvalidD : TestFormat_Op<"variable_invalid_d", [{
  operands $operand attr-dict
}]>, Arguments<(ins I64:$operand)>;
// CHECK: error: results can not be used at the top level
def VariableInvalidE : TestFormat_Op<"variable_invalid_e", [{
  $result attr-dict
}]>, Results<(outs I64:$result)>;
// CHECK: error: successor 'successor' is already bound
def VariableInvalidF : TestFormat_Op<"variable_invalid_f", [{
  $successor $successor attr-dict
}]> {
  let successors = (successor AnySuccessor:$successor);
}
// CHECK: error: successor 'successor' is already bound
def VariableInvalidG : TestFormat_Op<"variable_invalid_g", [{
  successors $successor attr-dict
}]> {
  let successors = (successor AnySuccessor:$successor);
}
// CHECK: error: format ambiguity caused by `:` literal found after attribute `attr` which does not have a buildable type
def VariableInvalidH : TestFormat_Op<"variable_invalid_h", [{
  $attr `:` attr-dict
}]>, Arguments<(ins ElementsAttr:$attr)>;
// CHECK: error: format ambiguity caused by `:` literal found after attribute `attr` which does not have a buildable type
def VariableInvalidI : TestFormat_Op<"variable_invalid_i", [{
  (`foo` $attr^)? `:` attr-dict
}]>, Arguments<(ins OptionalAttr<ElementsAttr>:$attr)>;
// CHECK-NOT: error:
def VariableInvalidJ : TestFormat_Op<"variable_invalid_j", [{
  $attr `:` attr-dict
}]>, Arguments<(ins OptionalAttr<I1Attr>:$attr)>;
def VariableInvalidK : TestFormat_Op<"variable_invalid_k", [{
  (`foo` $attr^)? `:` attr-dict
}]>, Arguments<(ins OptionalAttr<I1Attr>:$attr)>;

//===----------------------------------------------------------------------===//
// Coverage Checks
//===----------------------------------------------------------------------===//

// CHECK: error: type of result #0, named 'result', is not buildable and a buildable type cannot be inferred
// CHECK: note: suggest adding a type constraint to the operation or adding a 'type($result)' directive to the custom assembly format
def ZCoverageInvalidA : TestFormat_Op<"variable_invalid_a", [{
  attr-dict
}]>, Arguments<(ins AnyMemRef:$operand)>, Results<(outs AnyMemRef:$result)>;
// CHECK: error: operand #0, named 'operand', not found
// CHECK: note: suggest adding a '$operand' directive to the custom assembly format
def ZCoverageInvalidB : TestFormat_Op<"variable_invalid_b", [{
  type($result) attr-dict
}]>, Arguments<(ins AnyMemRef:$operand)>, Results<(outs AnyMemRef:$result)>;
// CHECK: error: type of operand #0, named 'operand', is not buildable and a buildable type cannot be inferred
// CHECK: note: suggest adding a type constraint to the operation or adding a 'type($operand)' directive to the custom assembly format
def ZCoverageInvalidC : TestFormat_Op<"variable_invalid_c", [{
  $operand type($result) attr-dict
}]>, Arguments<(ins AnyMemRef:$operand)>, Results<(outs AnyMemRef:$result)>;
// CHECK: error: type of operand #0, named 'operand', is not buildable and a buildable type cannot be inferred
// CHECK: note: suggest adding a type constraint to the operation or adding a 'type($operand)' directive to the custom assembly format
def ZCoverageInvalidD : TestFormat_Op<"variable_invalid_d", [{
  operands attr-dict
}]>, Arguments<(ins Variadic<I64>:$operand)>;
// CHECK: error: type of result #0, named 'result', is not buildable and a buildable type cannot be inferred
// CHECK: note: suggest adding a type constraint to the operation or adding a 'type($result)' directive to the custom assembly format
def ZCoverageInvalidE : TestFormat_Op<"variable_invalid_e", [{
  attr-dict
}]>, Results<(outs Variadic<I64>:$result)>;
// CHECK: error: successor #0, named 'successor', not found
// CHECK: note: suggest adding a '$successor' directive to the custom assembly format
def ZCoverageInvalidF : TestFormat_Op<"variable_invalid_f", [{
	 attr-dict
}]> {
  let successors = (successor AnySuccessor:$successor);
}
// CHECK: error: type of operand #0, named 'operand', is not buildable and a buildable type cannot be inferred
// CHECK: note: suggest adding a type constraint to the operation or adding a 'type($operand)' directive to the custom assembly format
def ZCoverageInvalidG : TestFormat_Op<"variable_invalid_g", [{
  operands attr-dict
}]>, Arguments<(ins Optional<I64>:$operand)>;
// CHECK: error: type of result #0, named 'result', is not buildable and a buildable type cannot be inferred
// CHECK: note: suggest adding a type constraint to the operation or adding a 'type($result)' directive to the custom assembly format
def ZCoverageInvalidH : TestFormat_Op<"variable_invalid_h", [{
  attr-dict
}]>, Results<(outs Optional<I64>:$result)>;

// CHECK-NOT: error
def ZCoverageValidA : TestFormat_Op<"variable_valid_a", [{
  $operand type($operand) type($result) attr-dict
}]>, Arguments<(ins AnyMemRef:$operand)>, Results<(outs AnyMemRef:$result)>;
def ZCoverageValidB : TestFormat_Op<"variable_valid_b", [{
  $operand type(operands) type(results) attr-dict
}]>, Arguments<(ins AnyMemRef:$operand)>, Results<(outs AnyMemRef:$result)>;
def ZCoverageValidC : TestFormat_Op<"variable_valid_c", [{
  operands functional-type(operands, results) attr-dict
}]>, Arguments<(ins AnyMemRef:$operand)>, Results<(outs AnyMemRef:$result)>;

// Check that we can infer type equalities from certain traits.
def ZCoverageValidD : TestFormat_Op<"variable_valid_d", [{
  operands type($result) attr-dict
}], [SameOperandsAndResultType]>, Arguments<(ins AnyMemRef:$operand)>,
     Results<(outs AnyMemRef:$result)>;
def ZCoverageValidE : TestFormat_Op<"variable_valid_e", [{
  $operand type($operand) attr-dict
}], [SameOperandsAndResultType]>, Arguments<(ins AnyMemRef:$operand)>,
     Results<(outs AnyMemRef:$result)>;
def ZCoverageValidF : TestFormat_Op<"variable_valid_f", [{
  operands type($other) attr-dict
}], [SameTypeOperands]>, Arguments<(ins AnyMemRef:$operand, AnyMemRef:$other)>;
def ZCoverageValidG : TestFormat_Op<"variable_valid_g", [{
  operands type($other) attr-dict
}], [AllTypesMatch<["operand", "other"]>]>,
     Arguments<(ins AnyMemRef:$operand, AnyMemRef:$other)>;
def ZCoverageValidH : TestFormat_Op<"variable_valid_h", [{
  operands type($result) attr-dict
}], [AllTypesMatch<["operand", "result"]>]>,
     Arguments<(ins AnyMemRef:$operand)>, Results<(outs AnyMemRef:$result)>;

