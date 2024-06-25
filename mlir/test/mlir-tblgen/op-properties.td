// RUN: mlir-tblgen -gen-op-decls -I %S/../../include %s | FileCheck %s

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

def Test_Dialect : Dialect {
  let name = "test";
  let cppNamespace = "foobar";
}
class NS_Op<string mnemonic, list<Trait> traits = []> :
    Op<Test_Dialect, mnemonic, traits>;

def OpWithAttr : NS_Op<"op_with_attr">{
  let arguments = (ins AnyAttr:$attr, OptionalAttr<AnyAttr>:$optional);
}

// CHECK: void setAttrAttr(::mlir::Attribute attr)
// CHECK-NEXT: getProperties().attr = attr
// CHECK: void setOptionalAttr(::mlir::Attribute attr)
// CHECK-NEXT: getProperties().optional = attr
