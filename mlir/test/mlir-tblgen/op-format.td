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

// CHECK-LABEL: CustomStringLiteralD::parse
// CHECK: parseFoo({{.*}}, result)
// CHECK-LABEL: CustomStringLiteralD::print
// CHECK: printFoo({{.*}}, getProperties())
def CustomStringLiteralD : TestFormat_Op<[{
  custom<Foo>(prop-dict) attr-dict
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
// CHECK-NEXT: result.getOrAddProperties<OptionalGroupB::Properties>().a = parser.getBuilder().getUnitAttr()
// CHECK: parser.parseKeyword("bar")
// CHECK-LABEL: OptionalGroupB::print
// CHECK: if (!(getAAttr() && getAAttr() != ((false) ? ::mlir::OpBuilder((*this)->getContext()).getUnitAttr() : nullptr)))
// CHECK-NEXT: odsPrinter << ' ' << "foo"
// CHECK-NEXT: else
// CHECK-NEXT: odsPrinter << ' ' << "bar"
def OptionalGroupB : TestFormat_Op<[{
  (`foo`) : (`bar` $a^)? attr-dict
}]>, Arguments<(ins UnitAttr:$a)>;

// Optional group anchored on a default-valued attribute:
// CHECK-LABEL: OptionalGroupC::parse

//       CHECK: if (getAAttr() != ::mlir::OpBuilder((*this)->getContext()).getStringAttr("default")) {
//  CHECK-NEXT:   odsPrinter << ' ';
//  CHECK-NEXT:   odsPrinter.printAttributeWithoutType(getAAttr());
//  CHECK-NEXT: }
def OptionalGroupC : TestFormat_Op<[{
  ($a^)? attr-dict
}]>, Arguments<(ins DefaultValuedStrAttr<StrAttr, "default">:$a)>;

// CHECK-LABEL: OptionalGroupD::parse
// CHECK: if (auto optResult = [&]() -> ::mlir::OptionalParseResult {
// CHECK:   auto odsResult = parseCustom(parser, aOperand, bOperand);
// CHECK:   if (!odsResult.has_value()) return {};
// CHECK:   if (::mlir::failed(*odsResult)) return ::mlir::failure();
// CHECK:   return ::mlir::success();
// CHECK: }(); optResult.has_value() && ::mlir::failed(*optResult)) {
// CHECK:   return ::mlir::failure();
// CHECK: } else if (optResult.has_value()) {

// CHECK-LABEL: OptionalGroupD::print
// CHECK-NEXT: if (((getA()) || (getB()))) {
// CHECK-NEXT:   odsPrinter << ' ';
// CHECK-NEXT:   printCustom
def OptionalGroupD : TestFormat_Op<[{
  (custom<Custom>($a, $b)^)? attr-dict
}], [AttrSizedOperandSegments]>, Arguments<(ins Optional<I64>:$a, Optional<I64>:$b)>;
