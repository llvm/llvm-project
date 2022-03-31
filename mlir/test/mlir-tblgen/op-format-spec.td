// RUN: mlir-tblgen -gen-op-decls -asmformat-error-is-fatal=false -I %S/../../include %s -o=%t 2>&1 | FileCheck %s

// This file contains tests for the specification of the declarative op format.

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

def TestDialect : Dialect {
  let name = "test";
}
class TestFormat_Op<string fmt, list<Trait> traits = []>
    : Op<TestDialect, "format_op", traits> {
  let assemblyFormat = fmt;
}

//===----------------------------------------------------------------------===//
// Directives
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// attr-dict

// CHECK: error: 'attr-dict' directive not found
def DirectiveAttrDictInvalidA : TestFormat_Op<[{
}]>;
// CHECK: error: 'attr-dict' directive has already been seen
def DirectiveAttrDictInvalidB : TestFormat_Op<[{
  attr-dict attr-dict
}]>;
// CHECK: error: 'attr-dict' directive has already been seen
def DirectiveAttrDictInvalidC : TestFormat_Op<[{
  attr-dict attr-dict-with-keyword
}]>;
// CHECK: error: 'attr-dict' directive can only be used as a top-level directive
def DirectiveAttrDictInvalidD : TestFormat_Op<[{
  type(attr-dict)
}]>;
// CHECK-NOT: error
def DirectiveAttrDictValidA : TestFormat_Op<[{
  attr-dict
}]>;
def DirectiveAttrDictValidB : TestFormat_Op<[{
  attr-dict-with-keyword
}]>;

//===----------------------------------------------------------------------===//
// custom

// CHECK: error: expected '<' before custom directive name
def DirectiveCustomInvalidA : TestFormat_Op<[{
  custom(
}]>;
// CHECK: error: expected custom directive name identifier
def DirectiveCustomInvalidB : TestFormat_Op<[{
  custom<>
}]>;
// CHECK: error: expected '>' after custom directive name
def DirectiveCustomInvalidC : TestFormat_Op<[{
  custom<MyDirective(
}]>;
// CHECK: error: expected '(' before custom directive parameters
def DirectiveCustomInvalidD : TestFormat_Op<[{
  custom<MyDirective>)
}]>;
// CHECK: error: only variables and types may be used as parameters to a custom directive
def DirectiveCustomInvalidE : TestFormat_Op<[{
  custom<MyDirective>(operands)
}]>;
// CHECK: error: expected ')' after custom directive parameters
def DirectiveCustomInvalidF : TestFormat_Op<[{
  custom<MyDirective>($operand<
}]>, Arguments<(ins I64:$operand)>;
// CHECK: error: type directives within a custom directive may only refer to variables
def DirectiveCustomInvalidH : TestFormat_Op<[{
  custom<MyDirective>(type(operands))
}]>;

// CHECK-NOT: error
def DirectiveCustomValidA : TestFormat_Op<[{
  custom<MyDirective>($operand) attr-dict
}]>, Arguments<(ins Optional<I64>:$operand)>;
def DirectiveCustomValidB : TestFormat_Op<[{
  custom<MyDirective>($operand, type($operand), type($result)) attr-dict
}]>, Arguments<(ins I64:$operand)>, Results<(outs I64:$result)>;
def DirectiveCustomValidC : TestFormat_Op<[{
  custom<MyDirective>($attr) attr-dict
}]>, Arguments<(ins I64Attr:$attr)>;

//===----------------------------------------------------------------------===//
// functional-type

// CHECK: error: 'functional-type' is only valid as a top-level directive
def DirectiveFunctionalTypeInvalidA : TestFormat_Op<[{
  functional-type(functional-type)
}]>;
// CHECK: error: expected '(' before argument list
def DirectiveFunctionalTypeInvalidB : TestFormat_Op<[{
  functional-type
}]>;
// CHECK: error: expected literal, variable, directive, or optional group
def DirectiveFunctionalTypeInvalidC : TestFormat_Op<[{
  functional-type(
}]>;
// CHECK: error: expected ',' after inputs argument
def DirectiveFunctionalTypeInvalidD : TestFormat_Op<[{
  functional-type(operands
}]>;
// CHECK: error: expected literal, variable, directive, or optional group
def DirectiveFunctionalTypeInvalidE : TestFormat_Op<[{
  functional-type(operands,
}]>;
// CHECK: error: expected ')' after argument list
def DirectiveFunctionalTypeInvalidF : TestFormat_Op<[{
  functional-type(operands, results
}]>;
// CHECK-NOT: error
def DirectiveFunctionalTypeValid : TestFormat_Op<[{
  functional-type(operands, results) attr-dict
}]>;

//===----------------------------------------------------------------------===//
// operands

// CHECK: error: 'operands' directive creates overlap in format
def DirectiveOperandsInvalidA : TestFormat_Op<[{
  operands operands
}]>;
// CHECK: error: 'operands' directive creates overlap in format
def DirectiveOperandsInvalidB : TestFormat_Op<[{
  $operand operands
}]>, Arguments<(ins I64:$operand)>;
// CHECK-NOT: error:
def DirectiveOperandsValid : TestFormat_Op<[{
  operands attr-dict
}]>;

//===----------------------------------------------------------------------===//
// ref

// CHECK: error: 'ref' is only valid within a `custom` directive
def DirectiveRefInvalidA : TestFormat_Op<[{
  ref(type($operand))
}]>, Arguments<(ins I64:$operand)>;

// CHECK: error: 'ref' of 'type($operand)' is not bound by a prior 'type' directive
def DirectiveRefInvalidB : TestFormat_Op<[{
  custom<Foo>(ref(type($operand)))
}]>, Arguments<(ins I64:$operand)>;

// CHECK: error: 'ref' of 'type(operands)' is not bound by a prior 'type' directive
def DirectiveRefInvalidC : TestFormat_Op<[{
  custom<Foo>(ref(type(operands)))
}]>;

// CHECK: error: 'ref' of 'type($result)' is not bound by a prior 'type' directive
def DirectiveRefInvalidD : TestFormat_Op<[{
  custom<Foo>(ref(type($result)))
}]>, Results<(outs I64:$result)>;

// CHECK: error: 'ref' of 'type(results)' is not bound by a prior 'type' directive
def DirectiveRefInvalidE : TestFormat_Op<[{
  custom<Foo>(ref(type(results)))
}]>;

// CHECK: error: 'ref' of 'successors' is not bound by a prior 'successors' directive
def DirectiveRefInvalidF : TestFormat_Op<[{
  custom<Foo>(ref(successors))
}]>;

// CHECK: error: 'ref' of 'regions' is not bound by a prior 'regions' directive
def DirectiveRefInvalidG : TestFormat_Op<[{
  custom<Foo>(ref(regions))
}]>;

// CHECK: error: expected '(' before argument list
def DirectiveRefInvalidH : TestFormat_Op<[{
  custom<Foo>(ref)
}]>;

// CHECK: error: expected ')' after argument list
def DirectiveRefInvalidI : TestFormat_Op<[{
  operands custom<Foo>(ref(operands(
}]>;

// CHECK: error: 'ref' of 'operands' is not bound by a prior 'operands' directive
def DirectiveRefInvalidJ : TestFormat_Op<[{
  custom<Foo>(ref(operands))
}]>;

// CHECK: error: 'ref' of 'attr-dict' is not bound by a prior 'attr-dict' directive
def DirectiveRefInvalidK : TestFormat_Op<[{
  custom<Foo>(ref(attr-dict))
}]>;

// CHECK: error: successor 'successor' must be bound before it is referenced
def DirectiveRefInvalidL : TestFormat_Op<[{
  custom<Foo>(ref($successor))
}]> {
  let successors = (successor AnySuccessor:$successor);
}

// CHECK: error: region 'region' must be bound before it is referenced
def DirectiveRefInvalidM : TestFormat_Op<[{
  custom<Foo>(ref($region))
}]> {
  let regions = (region AnyRegion:$region);
}

// CHECK: error: attribute 'attr' must be bound before it is referenced
def DirectiveRefInvalidN : TestFormat_Op<[{
  custom<Foo>(ref($attr))
}]>, Arguments<(ins I64Attr:$attr)>;


// CHECK: error: operand 'operand' must be bound before it is referenced
def DirectiveRefInvalidO : TestFormat_Op<[{
  custom<Foo>(ref($operand))
}]>, Arguments<(ins I64:$operand)>;

//===----------------------------------------------------------------------===//
// regions

// CHECK: error: 'regions' directive creates overlap in format
def DirectiveRegionsInvalidA : TestFormat_Op<[{
  regions regions attr-dict
}]>;
// CHECK: error: 'regions' directive creates overlap in format
def DirectiveRegionsInvalidB : TestFormat_Op<[{
  $region regions attr-dict
}]> {
  let regions = (region AnyRegion:$region);
}
// CHECK: error: 'regions' is only valid as a top-level directive
def DirectiveRegionsInvalidC : TestFormat_Op<[{
  type(regions)
}]>;
// CHECK-NOT: error:
def DirectiveRegionsValid : TestFormat_Op<[{
  regions attr-dict
}]>;

//===----------------------------------------------------------------------===//
// results

// CHECK: error: 'results' directive can can only be used as a child to a 'type' directive
def DirectiveResultsInvalidA : TestFormat_Op<[{
  results
}]>;

//===----------------------------------------------------------------------===//
// successors

// CHECK: error: 'successors' is only valid as a top-level directive
def DirectiveSuccessorsInvalidA : TestFormat_Op<[{
  type(successors)
}]>;

//===----------------------------------------------------------------------===//
// type

// CHECK: error: expected '(' before argument list
def DirectiveTypeInvalidA : TestFormat_Op<[{
  type
}]>;
// CHECK: error: expected literal, variable, directive, or optional group
def DirectiveTypeInvalidB : TestFormat_Op<[{
  type(
}]>;
// CHECK: error: expected ')' after argument list
def DirectiveTypeInvalidC : TestFormat_Op<[{
  type(operands
}]>;
// CHECK-NOT: error:
def DirectiveTypeValid : TestFormat_Op<[{
  type(operands) attr-dict
}]>;

//===----------------------------------------------------------------------===//
// functional-type/type operands

// CHECK: error: literals may only be used in the top-level section of the format
def DirectiveTypeZOperandInvalidA : TestFormat_Op<[{
  type(`literal`)
}]>;
// CHECK: error: 'operands' 'type' is already bound
def DirectiveTypeZOperandInvalidB : TestFormat_Op<[{
  type(operands) type(operands)
}]>;
// CHECK: error: 'operands' 'type' is already bound
def DirectiveTypeZOperandInvalidC : TestFormat_Op<[{
  type($operand) type(operands)
}]>, Arguments<(ins I64:$operand)>;
// CHECK: error: 'type' of 'operand' is already bound
def DirectiveTypeZOperandInvalidD : TestFormat_Op<[{
  type(operands) type($operand)
}]>, Arguments<(ins I64:$operand)>;
// CHECK: error: 'type' of 'operand' is already bound
def DirectiveTypeZOperandInvalidE : TestFormat_Op<[{
  type($operand) type($operand)
}]>, Arguments<(ins I64:$operand)>;
// CHECK: error: 'results' 'type' is already bound
def DirectiveTypeZOperandInvalidF : TestFormat_Op<[{
  type(results) type(results)
}]>;
// CHECK: error: 'results' 'type' is already bound
def DirectiveTypeZOperandInvalidG : TestFormat_Op<[{
  type($result) type(results)
}]>, Results<(outs I64:$result)>;
// CHECK: error: 'type' of 'result' is already bound
def DirectiveTypeZOperandInvalidH : TestFormat_Op<[{
  type(results) type($result)
}]>, Results<(outs I64:$result)>;
// CHECK: error: 'type' of 'result' is already bound
def DirectiveTypeZOperandInvalidI : TestFormat_Op<[{
  type($result) type($result)
}]>, Results<(outs I64:$result)>;

//===----------------------------------------------------------------------===//
// Literals
//===----------------------------------------------------------------------===//

// Test all of the valid literals.
// CHECK: error: expected valid literal but got 'a:': keywords should contain only alphanum, '_', '$', or '.' characters
def LiteralInvalidA : TestFormat_Op<[{
  `a:`
}]>;
// CHECK: error: expected valid literal but got '1': single character literal must be a letter or one of '_:,=<>()[]{}?+*'
def LiteralInvalidB : TestFormat_Op<[{
  `1`
}]>;
// CHECK: error: expected valid literal but got ':abc': valid keyword starts with a letter or '_'
def LiteralInvalidC : TestFormat_Op<[{
  `:abc`
}]>;

// CHECK: error: unexpected end of file in literal
// CHECK: error: expected literal, variable, directive, or optional group
def LiteralInvalidD : TestFormat_Op<[{
  `
}]>;
// CHECK-NOT: error
def LiteralValid : TestFormat_Op<[{
  `_` `:` `,` `=` `<` `>` `(` `)` `[` `]` `?` `+` `*` ` ` `` `->` `\n` `abc$._`
  attr-dict
}]>;

//===----------------------------------------------------------------------===//
// OIList Element
//===----------------------------------------------------------------------===//

// CHECK: error: format ambiguity because bar is used in two adjacent oilist elements.
def OIListAdjacentOIList : TestFormat_Op<[{
  oilist ( `foo` | `bar` ) oilist ( `bar` | `buzz` ) attr-dict
}]>;
// CHECK: error: expected literal, but got ')'
def OIListErrorExpectedLiteral : TestFormat_Op<[{
  oilist( `keyword` | ) attr-dict
}]>;
// CHECK: error: expected literal, but got ')'
def OIListErrorExpectedEmpty : TestFormat_Op<[{
  oilist() attr-dict
}]>;
// CHECK: error: expected literal, but got '$arg0'
def OIListErrorNoLiteral : TestFormat_Op<[{
  oilist( $arg0 `:` type($arg0) | $arg1 `:` type($arg1) ) attr-dict
}], [AttrSizedOperandSegments]>, Arguments<(ins Optional<AnyType>:$arg0, Optional<AnyType>:$arg1)>;
// CHECK: error: format ambiguity because foo is used both in oilist element and the adjacent literal.
def OIListLiteralAmbiguity : TestFormat_Op<[{
  oilist( `foo` | `bar` ) `foo` attr-dict
}]>;
// CHECK: error: expected '(' before oilist argument list
def OIListStartingToken : TestFormat_Op<[{
  oilist `wrong` attr-dict
}]>;

// CHECK-NOT: error
def OIListTrivial : TestFormat_Op<[{
  oilist(`keyword` `(` `)` | `otherkeyword` `(` `)`) attr-dict
}]>;
def OIListSimple : TestFormat_Op<[{
  oilist( `keyword` $arg0 `:` type($arg0)
        | `otherkeyword` $arg1 `:` type($arg1)
        | `thirdkeyword` $arg2 `:` type($arg2) )
  attr-dict
}], [AttrSizedOperandSegments]>, Arguments<(ins Optional<AnyType>:$arg0, Optional<AnyType>:$arg1, Optional<AnyType>:$arg2)>;
def OIListVariadic : TestFormat_Op<[{
  oilist( `keyword` `(` $args0 `:` type($args0) `)`
        | `otherkeyword` `(` $args1 `:` type($args1) `)`
        | `thirdkeyword` `(` $args2 `:` type($args2) `)`)
  attr-dict
}], [AttrSizedOperandSegments]>, Arguments<(ins Variadic<AnyType>:$args0, Variadic<AnyType>:$args1, Variadic<AnyType>:$args2)>;
def OIListCustom : TestFormat_Op<[{
  oilist( `private` `(` $arg0 `:` type($arg0) `)`
        | `nowait`
        | `reduction` custom<ReductionClause>($arg1, type($arg1))) attr-dict
}], [AttrSizedOperandSegments]>, Arguments<(ins Optional<AnyType>:$arg0, Optional<AnyType>:$arg1)>;

//===----------------------------------------------------------------------===//
// Optional Groups
//===----------------------------------------------------------------------===//

// CHECK: error: optional groups can only be used as top-level elements
def OptionalInvalidA : TestFormat_Op<[{
  type(($attr^)?) attr-dict
}]>, Arguments<(ins OptionalAttr<I64Attr>:$attr)>;
// CHECK: error: expected literal, variable, directive, or optional group
def OptionalInvalidB : TestFormat_Op<[{
  () attr-dict
}]>, Arguments<(ins OptionalAttr<I64Attr>:$attr)>;
// CHECK: error: optional group has no anchor element
def OptionalInvalidC : TestFormat_Op<[{
  ($attr)? attr-dict
}]>, Arguments<(ins OptionalAttr<I64Attr>:$attr)>;
// CHECK: error: first parsable element of an optional group must be a literal or variable
def OptionalInvalidD : TestFormat_Op<[{
  (type($operand) $operand^)? attr-dict
}]>, Arguments<(ins Optional<I64>:$operand)>;
// CHECK: error: only literals, types, and variables can be used within an optional group
def OptionalInvalidE : TestFormat_Op<[{
  (`,` $attr^ type(operands))? attr-dict
}]>, Arguments<(ins OptionalAttr<I64Attr>:$attr)>;
// CHECK: error: only one element can be marked as the anchor of an optional group
def OptionalInvalidF : TestFormat_Op<[{
  ($attr^ $attr2^)? attr-dict
}]>, Arguments<(ins OptionalAttr<I64Attr>:$attr, OptionalAttr<I64Attr>:$attr2)>;
// CHECK: error: only optional attributes can be used to anchor an optional group
def OptionalInvalidG : TestFormat_Op<[{
  ($attr^)? attr-dict
}]>, Arguments<(ins I64Attr:$attr)>;
// CHECK: error: only variable length operands can be used within an optional group
def OptionalInvalidH : TestFormat_Op<[{
  ($arg^)? attr-dict
}]>, Arguments<(ins I64:$arg)>;
// CHECK: error: only literals, types, and variables can be used within an optional group
def OptionalInvalidI : TestFormat_Op<[{
  (functional-type($arg, results)^)? attr-dict
}]>, Arguments<(ins Variadic<I64>:$arg)>;
// CHECK: error: only literals, types, and variables can be used within an optional group
def OptionalInvalidJ : TestFormat_Op<[{
  (attr-dict^)?
}]>;
// CHECK: error: expected '?' after optional group
def OptionalInvalidK : TestFormat_Op<[{
  ($arg^)
}]>, Arguments<(ins Variadic<I64>:$arg)>;
// CHECK: error: only variables and types can be used to anchor an optional group
def OptionalInvalidL : TestFormat_Op<[{
  (custom<MyDirective>($arg)^)?
}]>, Arguments<(ins I64:$arg)>;
// CHECK: error: only variables and types can be used to anchor an optional group
def OptionalInvalidM : TestFormat_Op<[{
  (` `^)?
}]>, Arguments<(ins)>;
// CHECK: error: expected '(' to start else branch of optional group
def OptionalInvalidN : TestFormat_Op<[{
  ($arg^):
}]>, Arguments<(ins Variadic<I64>:$arg)>;
// CHECK: error: expected literal, variable, directive, or optional group
def OptionalInvalidO : TestFormat_Op<[{
  ($arg^):(`test`
}]>, Arguments<(ins Variadic<I64>:$arg)>;
// CHECK: error: expected '?' after optional group
def OptionalInvalidP : TestFormat_Op<[{
  ($arg^):(`test`)
}]>, Arguments<(ins Variadic<I64>:$arg)>;

// CHECK-NOT: error
def OptionalValidA : TestFormat_Op<[{
  (` ` `` $arg^)?
}]>;

//===----------------------------------------------------------------------===//
// Variables
//===----------------------------------------------------------------------===//

// CHECK: error: expected variable to refer to an argument, region, result, or successor
def VariableInvalidA : TestFormat_Op<[{
  $unknown_arg attr-dict
}]>;
// CHECK: error: attribute 'attr' is already bound
def VariableInvalidB : TestFormat_Op<[{
  $attr $attr attr-dict
}]>, Arguments<(ins I64Attr:$attr)>;
// CHECK: error: operand 'operand' is already bound
def VariableInvalidC : TestFormat_Op<[{
  $operand $operand attr-dict
}]>, Arguments<(ins I64:$operand)>;
// CHECK: error: operand 'operand' is already bound
def VariableInvalidD : TestFormat_Op<[{
  operands $operand attr-dict
}]>, Arguments<(ins I64:$operand)>;
// CHECK: error: result variables can can only be used as a child to a 'type' directive
def VariableInvalidE : TestFormat_Op<[{
  $result attr-dict
}]>, Results<(outs I64:$result)>;
// CHECK: error: successor 'successor' is already bound
def VariableInvalidF : TestFormat_Op<[{
  $successor $successor attr-dict
}]> {
  let successors = (successor AnySuccessor:$successor);
}
// CHECK: error: successor 'successor' is already bound
def VariableInvalidG : TestFormat_Op<[{
  successors $successor attr-dict
}]> {
  let successors = (successor AnySuccessor:$successor);
}
// CHECK: error: format ambiguity caused by `:` literal found after attribute `attr` which does not have a buildable type
def VariableInvalidH : TestFormat_Op<[{
  $attr `:` attr-dict
}]>, Arguments<(ins ElementsAttr:$attr)>;
// CHECK: error: format ambiguity caused by `:` literal found after attribute `attr` which does not have a buildable type
def VariableInvalidI : TestFormat_Op<[{
  (`foo` $attr^)? `:` attr-dict
}]>, Arguments<(ins OptionalAttr<ElementsAttr>:$attr)>;
// CHECK: error: format ambiguity caused by `:` literal found after attribute `attr` which does not have a buildable type
def VariableInvalidJ : TestFormat_Op<[{
  $attr ` ` `:` attr-dict
}]>, Arguments<(ins ElementsAttr:$attr)>;
// CHECK: error: region 'region' is already bound
def VariableInvalidK : TestFormat_Op<[{
  $region $region attr-dict
}]> {
  let regions = (region AnyRegion:$region);
}
// CHECK: error: region 'region' is already bound
def VariableInvalidL : TestFormat_Op<[{
  regions $region attr-dict
}]> {
  let regions = (region AnyRegion:$region);
}
// CHECK: error: regions can only be used at the top level
def VariableInvalidM : TestFormat_Op<[{
  type($region)
}]> {
  let regions = (region AnyRegion:$region);
}
// CHECK: error: region #0, named 'region', not found
def VariableInvalidN : TestFormat_Op<[{
  attr-dict
}]> {
  let regions = (region AnyRegion:$region);
}
// CHECK-NOT: error:
def VariableValidA : TestFormat_Op<[{
  $attr `:` attr-dict
}]>, Arguments<(ins OptionalAttr<I1Attr>:$attr)>;
def VariableValidB : TestFormat_Op<[{
  (`foo` $attr^)? `:` attr-dict
}]>, Arguments<(ins OptionalAttr<I1Attr>:$attr)>;

//===----------------------------------------------------------------------===//
// Coverage Checks
//===----------------------------------------------------------------------===//

// CHECK: error: type of result #0, named 'result', is not buildable and a buildable type cannot be inferred
// CHECK: note: suggest adding a type constraint to the operation or adding a 'type($result)' directive to the custom assembly format
def ZCoverageInvalidA : TestFormat_Op<[{
  attr-dict
}]>, Arguments<(ins AnyMemRef:$operand)>, Results<(outs AnyMemRef:$result)>;
// CHECK: error: operand #0, named 'operand', not found
// CHECK: note: suggest adding a '$operand' directive to the custom assembly format
def ZCoverageInvalidB : TestFormat_Op<[{
  type($result) attr-dict
}]>, Arguments<(ins AnyMemRef:$operand)>, Results<(outs AnyMemRef:$result)>;
// CHECK: error: type of operand #0, named 'operand', is not buildable and a buildable type cannot be inferred
// CHECK: note: suggest adding a type constraint to the operation or adding a 'type($operand)' directive to the custom assembly format
def ZCoverageInvalidC : TestFormat_Op<[{
  $operand type($result) attr-dict
}]>, Arguments<(ins AnyMemRef:$operand)>, Results<(outs AnyMemRef:$result)>;
// CHECK: error: type of operand #0, named 'operand', is not buildable and a buildable type cannot be inferred
// CHECK: note: suggest adding a type constraint to the operation or adding a 'type($operand)' directive to the custom assembly format
def ZCoverageInvalidD : TestFormat_Op<[{
  operands attr-dict
}]>, Arguments<(ins Variadic<I64>:$operand)>;
// CHECK: error: type of result #0, named 'result', is not buildable and a buildable type cannot be inferred
// CHECK: note: suggest adding a type constraint to the operation or adding a 'type($result)' directive to the custom assembly format
def ZCoverageInvalidE : TestFormat_Op<[{
  attr-dict
}]>, Results<(outs Variadic<I64>:$result)>;
// CHECK: error: successor #0, named 'successor', not found
// CHECK: note: suggest adding a '$successor' directive to the custom assembly format
def ZCoverageInvalidF : TestFormat_Op<[{
	 attr-dict
}]> {
  let successors = (successor AnySuccessor:$successor);
}
// CHECK: error: type of operand #0, named 'operand', is not buildable and a buildable type cannot be inferred
// CHECK: note: suggest adding a type constraint to the operation or adding a 'type($operand)' directive to the custom assembly format
def ZCoverageInvalidG : TestFormat_Op<[{
  operands attr-dict
}]>, Arguments<(ins Optional<I64>:$operand)>;
// CHECK: error: type of result #0, named 'result', is not buildable and a buildable type cannot be inferred
// CHECK: note: suggest adding a type constraint to the operation or adding a 'type($result)' directive to the custom assembly format
def ZCoverageInvalidH : TestFormat_Op<[{
  attr-dict
}]>, Results<(outs Optional<I64>:$result)>;

// CHECK-NOT: error
def ZCoverageValidA : TestFormat_Op<[{
  $operand type($operand) type($result) attr-dict
}]>, Arguments<(ins AnyMemRef:$operand)>, Results<(outs AnyMemRef:$result)>;
def ZCoverageValidB : TestFormat_Op<[{
  $operand type(operands) type(results) attr-dict
}]>, Arguments<(ins AnyMemRef:$operand)>, Results<(outs AnyMemRef:$result)>;
def ZCoverageValidC : TestFormat_Op<[{
  operands functional-type(operands, results) attr-dict
}]>, Arguments<(ins AnyMemRef:$operand)>, Results<(outs AnyMemRef:$result)>;

// Check that we can infer type equalities from certain traits.
def ZCoverageValidD : TestFormat_Op<[{
  operands type($result) attr-dict
}], [SameOperandsAndResultType]>, Arguments<(ins AnyMemRef:$operand)>,
     Results<(outs AnyMemRef:$result)>;
def ZCoverageValidE : TestFormat_Op<[{
  $operand type($operand) attr-dict
}], [SameOperandsAndResultType]>, Arguments<(ins AnyMemRef:$operand)>,
     Results<(outs AnyMemRef:$result)>;
def ZCoverageValidF : TestFormat_Op<[{
  operands type($other) attr-dict
}], [SameTypeOperands]>, Arguments<(ins AnyMemRef:$operand, AnyMemRef:$other)>;
def ZCoverageValidG : TestFormat_Op<[{
  operands type($other) attr-dict
}], [AllTypesMatch<["operand", "other"]>]>,
     Arguments<(ins AnyMemRef:$operand, AnyMemRef:$other)>;
def ZCoverageValidH : TestFormat_Op<[{
  operands type($result) attr-dict
}], [AllTypesMatch<["operand", "result"]>]>,
     Arguments<(ins AnyMemRef:$operand)>, Results<(outs AnyMemRef:$result)>;
def ZCoverageValidI : TestFormat_Op<[{
  operands type(operands) attr-dict
}], [InferTypeOpInterface]>, Arguments<(ins Variadic<I64>:$inputs)>, Results<(outs I64:$result)>;
