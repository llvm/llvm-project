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
// CHECK: printFoo({{.*}}, parser.getBuilder().getI1Type())
def CustomStringLiteralA : TestFormat_Op<[{
  custom<Foo>("$_builder.getI1Type()") attr-dict
}]>;

// CHECK-LABEL: CustomStringLiteralB::parse
// CHECK: parseFoo({{.*}}, IndexType::get(parser.getContext()))
// CHECK-LABEL: CustomStringLiteralB::print
// CHECK: printFoo({{.*}}, IndexType::get(parser.getContext()))
def CustomStringLiteralB : TestFormat_Op<[{
  custom<Foo>("IndexType::get($_ctxt)") attr-dict
}]>;

// CHECK-LABEL: CustomStringLiteralC::parse
// CHECK: parseFoo({{.*}}, parser.getBuilder().getStringAttr("foo"))
// CHECK-LABEL: CustomStringLiteralC::print
// CHECK: printFoo({{.*}}, parser.getBuilder().getStringAttr("foo"))
def CustomStringLiteralC : TestFormat_Op<[{
  custom<Foo>("$_builder.getStringAttr(\"foo\")") attr-dict
}]>;
