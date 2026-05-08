// RUN: mlir-tblgen -gen-op-decls -I %S/../../include %s -o=%t

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
//===----------------------------------------------------------------------===//

// CHECK-NOT: error
def DirectiveAttrDictValidA : TestFormat_Op<[{
  attr-dict
}]>;
def DirectiveAttrDictValidB : TestFormat_Op<[{
  attr-dict-with-keyword
}]>;

//===----------------------------------------------------------------------===//
// custom
//===----------------------------------------------------------------------===//

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
def DirectiveCustomValidD : TestFormat_Op<[{
  (`(` custom<MyDirective>($operand)^ `)`)? attr-dict
}]>, Arguments<(ins Optional<I64>:$operand)>;
def DirectiveCustomValidE : TestFormat_Op<[{
  custom<MyDirective>(prop-dict) attr-dict
}]>, Arguments<(ins UnitAttr:$flag)>;
def DirectiveCustomValidF : TestFormat_Op<[{
  $operand custom<MyDirective>(ref($operand)) attr-dict
}]>, Arguments<(ins Optional<I64>:$operand)>;
def DirectiveCustomValidG : TestFormat_Op<[{
  $body custom<MyDirective>(ref($body)) attr-dict
}]> {
  let regions = (region AnyRegion:$body);
}
def DirectiveCustomValidH : TestFormat_Op<[{
  $successor custom<MyDirective>(ref($successor)) attr-dict
}]> {
  let successors = (successor AnySuccessor:$successor);
}

//===----------------------------------------------------------------------===//
// functional-type
//===----------------------------------------------------------------------===//

// CHECK-NOT: error
def DirectiveFunctionalTypeValid : TestFormat_Op<[{
  functional-type(operands, results) attr-dict
}]>;

//===----------------------------------------------------------------------===//
// operands
//===----------------------------------------------------------------------===//

// CHECK-NOT: error:
def DirectiveOperandsValid : TestFormat_Op<[{
  operands attr-dict
}]>;

//===----------------------------------------------------------------------===//
// regions
//===----------------------------------------------------------------------===//

// CHECK-NOT: error:
def DirectiveRegionsValid : TestFormat_Op<[{
  regions attr-dict
}]>;

//===----------------------------------------------------------------------===//
// results
//===----------------------------------------------------------------------===//

// CHECK-NOT: error:
def DirectiveResultsValid : TestFormat_Op<[{
  type(results) attr-dict
}]>;

//===----------------------------------------------------------------------===//
// successors
//===----------------------------------------------------------------------===//

// CHECK-NOT: error:
def DirectiveSuccessorsValid : TestFormat_Op<[{
  successors attr-dict
}]>;

//===----------------------------------------------------------------------===//
// type
//===----------------------------------------------------------------------===//

// CHECK-NOT: error:
def DirectiveTypeValid : TestFormat_Op<[{
  type(operands) attr-dict
}]>;

//===----------------------------------------------------------------------===//
// Literals
//===----------------------------------------------------------------------===//

// CHECK-NOT: error
def LiteralValid : TestFormat_Op<[{
  `_` `:` `,` `=` `<` `>` `(` `)` `[` `]` `?` `+` `-` `*` ` ` `` `->` `\n` `abc$._`
  attr-dict
}]>;

//===----------------------------------------------------------------------===//
// OIList Element
//===----------------------------------------------------------------------===//

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

// CHECK-NOT: error
def OptionalValidA : TestFormat_Op<[{
  (` ` `` $arg^)? attr-dict
}]>, Arguments<(ins Optional<I32>:$arg)>;

//===----------------------------------------------------------------------===//
// Strings
//===----------------------------------------------------------------------===//

// CHECK-NOT: error
def StringValid : TestFormat_Op<[{ custom<Foo>("foo") attr-dict }]>;

//===----------------------------------------------------------------------===//
// Variables
//===----------------------------------------------------------------------===//

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
}], [SameOperandsAndResultType]>, Arguments<(ins AnyMemRef)>,
     Results<(outs AnyMemRef:$result)>;
def ZCoverageValidE : TestFormat_Op<[{
  $operand type($operand) attr-dict
}], [SameOperandsAndResultType]>, Arguments<(ins AnyMemRef:$operand)>,
     Results<(outs AnyMemRef)>;
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
