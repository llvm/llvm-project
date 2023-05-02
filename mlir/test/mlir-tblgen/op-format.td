// RUN: mlir-tblgen -gen-op-defs -I %S/../../include %s | FileCheck %s

include "mlir/IR/OpBase.td"

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
// custom

// CHECK-LABEL: CustomStringLiteralA::parse
// CHECK: parseFoo({{.*}}, parser.getBuilder().getI1Type())
// CHECK-LABEL: CustomStringLiteralA::print
// CHECK: printFoo({{.*}}, ::mlir::Builder(getContext()).getI1Type())
def CustomStringLiteralA : TestFormat_Op<[{
  custom<Foo>("$_builder.getI1Type()") attr-dict
}]>;

// CHECK-LABEL: CustomStringLiteralB::parse
// CHECK: parseFoo({{.*}}, IndexType::get(parser.getContext()))
// CHECK-LABEL: CustomStringLiteralB::print
// CHECK: printFoo({{.*}}, IndexType::get(getContext()))
def CustomStringLiteralB : TestFormat_Op<[{
  custom<Foo>("IndexType::get($_ctxt)") attr-dict
}]>;

// CHECK-LABEL: CustomStringLiteralC::parse
// CHECK: parseFoo({{.*}}, parser.getBuilder().getStringAttr("foo"))
// CHECK-LABEL: CustomStringLiteralC::print
// CHECK: printFoo({{.*}}, ::mlir::Builder(getContext()).getStringAttr("foo"))
def CustomStringLiteralC : TestFormat_Op<[{
  custom<Foo>("$_builder.getStringAttr(\"foo\")") attr-dict
}]>;

//===----------------------------------------------------------------------===//
// Optional Groups
//===----------------------------------------------------------------------===//

// CHECK-LABEL: OptionalGroupA::parse
// CHECK: if (::mlir::succeeded(parser.parseOptionalQuestion())
// CHECK-NEXT: else
// CHECK: parser.parseOptionalOperand
// CHECK-LABEL: OptionalGroupA::print
// CHECK: if (!getA())
// CHECK-NEXT: odsPrinter << ' ' << "?";
// CHECK-NEXT: else
// CHECK: odsPrinter << value;
def OptionalGroupA : TestFormat_Op<[{
  (`?`) : ($a^)? attr-dict
}]>, Arguments<(ins Optional<I1>:$a)>;

// CHECK-LABEL: OptionalGroupB::parse
// CHECK: if (::mlir::succeeded(parser.parseOptionalKeyword("foo")))
// CHECK-NEXT: else
// CHECK-NEXT: result.addAttribute("a", parser.getBuilder().getUnitAttr())
// CHECK: parser.parseKeyword("bar")
// CHECK-LABEL: OptionalGroupB::print
// CHECK: if (!(*this)->getAttr("a"))
// CHECK-NEXT: odsPrinter << ' ' << "foo"
// CHECK-NEXT: else
// CHECK-NEXT: odsPrinter << ' ' << "bar"
def OptionalGroupB : TestFormat_Op<[{
  (`foo`) : (`bar` $a^)? attr-dict
}]>, Arguments<(ins UnitAttr:$a)>;

// Optional group anchored on a default-valued attribute:
// CHECK-LABEL: OptionalGroupC::parse
//       CHECK: if ((*this)->getAttr("a") != ::mlir::OpBuilder((*this)->getContext()).getStringAttr("default")) {
//  CHECK-NEXT:   odsPrinter << ' ';
//  CHECK-NEXT:   odsPrinter.printAttributeWithoutType(getAAttr());
//  CHECK-NEXT: }
def OptionalGroupC : TestFormat_Op<[{
  ($a^)? attr-dict
}]>, Arguments<(ins DefaultValuedStrAttr<StrAttr, "default">:$a)>;
