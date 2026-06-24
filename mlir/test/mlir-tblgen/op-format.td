// RUN: mlir-tblgen -gen-op-defs -I %S/../../include %s | FileCheck %s

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"

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
//===----------------------------------------------------------------------===//

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
// EnumAttr formatting
//===----------------------------------------------------------------------===//

// Test that EnumAttr (backed by EnumInfo, not EnumAttrInfo) is recognized as
// an enum attribute and uses the enum-keyword format path.

def TestEnumCase0 : I32EnumCase<"Case0", 0>;
def TestEnumCase1 : I32EnumCase<"Case1", 1>;

def TestEnum : I32Enum<"TestEnum", "a test enum", [TestEnumCase0, TestEnumCase1]> {
  let cppNamespace = "::test";
}

def TestEnumAttr : EnumAttr<TestDialect, TestEnum, "enum">;

// CHECK-LABEL: EnumAttrOp::parse
// CHECK: symbolizeTestEnum
// CHECK-LABEL: EnumAttrOp::print
// CHECK: stringifyTestEnum
def EnumAttrOp : TestFormat_Op<"$attr attr-dict">,
    Arguments<(ins TestEnumAttr:$attr)>;

// Test that legacy EnumAttrInfo-based attributes (I32EnumAttr) also use the
// enum-keyword format path.

def LegacyEnumCase0 : I32EnumAttrCase<"LCase0", 0>;
def LegacyEnumCase1 : I32EnumAttrCase<"LCase1", 1>;

def LegacyTestEnum : I32EnumAttr<"LegacyTestEnum", "a legacy test enum",
                                 [LegacyEnumCase0, LegacyEnumCase1]> {
  let cppNamespace = "::test";
}

// CHECK-LABEL: LegacyEnumAttrOp::parse
// CHECK: symbolizeLegacyTestEnum
// CHECK-LABEL: LegacyEnumAttrOp::print
// CHECK: stringifyLegacyTestEnum
def LegacyEnumAttrOp : TestFormat_Op<"$attr attr-dict">,
    Arguments<(ins LegacyTestEnum:$attr)>;

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

// CHECK-LABEL: RegionRef::parse
// CHECK:   auto odsResult = parseCustom(parser, *bodyRegion);
// CHECK-LABEL: RegionRef::print
// CHECK:   printCustom(_odsPrinter, *this, getBody());
def RegionRef : TestFormat_Op<[{
  $body custom<Custom>(ref($body)) attr-dict
}]> {
  let regions = (region AnyRegion:$body);
}

// CHECK-LABEL: SuccessorRef::parse
// CHECK:   auto odsResult = parseCustom(parser, successorSuccessor);
// CHECK-LABEL: SuccessorRef::print
// CHECK:   printCustom(_odsPrinter, *this, getSuccessor());
def SuccessorRef : TestFormat_Op<[{
  $successor custom<Custom>(ref($successor)) attr-dict
}]> {
  let successors = (successor AnySuccessor:$successor);
}
