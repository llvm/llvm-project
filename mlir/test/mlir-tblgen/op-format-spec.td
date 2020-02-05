// RUN: mlir-tblgen -gen-op-decls -asmformat-error-is-fatal=false -I %S/../../include %s 2>&1 | FileCheck %s --dump-input-on-failure

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

// CHECK: error: format missing 'attr-dict' directive
def DirectiveAttrDictInvalidA : TestFormat_Op<"attrdict_invalid_a", [{
}]>;
// CHECK: error: 'attr-dict' directive has already been seen
def DirectiveAttrDictInvalidB : TestFormat_Op<"attrdict_invalid_b", [{
  attr-dict attr-dict
}]>;
// CHECK: error: 'attr-dict' directive can only be used as a top-level directive
def DirectiveAttrDictInvalidC : TestFormat_Op<"attrdict_invalid_c", [{
  type(attr-dict)
}]>;
// CHECK-NOT: error
def DirectiveAttrDictValid : TestFormat_Op<"attrdict_valid", [{
  attr-dict
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
// CHECK: error: expected directive, literal, or variable
def DirectiveFunctionalTypeInvalidC : TestFormat_Op<"functype_invalid_c", [{
  functional-type(
}]>;
// CHECK: error: expected ',' after inputs argument
def DirectiveFunctionalTypeInvalidD : TestFormat_Op<"functype_invalid_d", [{
  functional-type(operands
}]>;
// CHECK: error: expected directive, literal, or variable
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
def DirectiveResultsInvalidA : TestFormat_Op<"operands_invalid_a", [{
  results
}]>;

//===----------------------------------------------------------------------===//
// type

// CHECK: error: expected '(' before argument list
def DirectiveTypeInvalidA : TestFormat_Op<"type_invalid_a", [{
  type
}]>;
// CHECK: error: expected directive, literal, or variable
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
// CHECK: error: expected directive, literal, or variable
def LiteralInvalidB : TestFormat_Op<"literal_invalid_b", [{
  `
}]>;
// CHECK-NOT: error
def LiteralValid : TestFormat_Op<"literal_valid", [{
  `_` `:` `,` `=` `<` `>` `(` `)` `[` `]` `->` `abc$._`
  attr-dict
}]>;

//===----------------------------------------------------------------------===//
// Variables
//===----------------------------------------------------------------------===//

// CHECK: error: expected variable to refer to a argument or result
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

//===----------------------------------------------------------------------===//
// Coverage Checks
//===----------------------------------------------------------------------===//

// CHECK: error: format missing instance of result #0('result') type
def ZCoverageInvalidA : TestFormat_Op<"variable_invalid_a", [{
  attr-dict
}]>, Arguments<(ins AnyMemRef:$operand)>, Results<(outs AnyMemRef:$result)>;
// CHECK: error: format missing instance of operand #0('operand')
def ZCoverageInvalidB : TestFormat_Op<"variable_invalid_b", [{
  type($result) attr-dict
}]>, Arguments<(ins AnyMemRef:$operand)>, Results<(outs AnyMemRef:$result)>;
// CHECK: error: format missing instance of operand #0('operand') type
def ZCoverageInvalidC : TestFormat_Op<"variable_invalid_c", [{
  $operand type($result) attr-dict
}]>, Arguments<(ins AnyMemRef:$operand)>, Results<(outs AnyMemRef:$result)>;
// CHECK: error: format missing instance of operand #0('operand') type
def ZCoverageInvalidD : TestFormat_Op<"variable_invalid_d", [{
  operands attr-dict
}]>, Arguments<(ins Variadic<I64>:$operand)>;
// CHECK: error: format missing instance of result #0('result') type
def ZCoverageInvalidE : TestFormat_Op<"variable_invalid_e", [{
  attr-dict
}]>, Results<(outs Variadic<I64>:$result)>;
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

