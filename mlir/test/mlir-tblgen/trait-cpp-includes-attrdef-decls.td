// RUN: mlir-tblgen -gen-attrdef-decls -I %S/../../include %s | FileCheck %s

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/Interfaces.td"

def Test_Dialect : Dialect {
  let name = "test";
  let cppNamespace = "::test";
}

def TestAttrInterface : AttrInterface<"TestAttrInterface"> {
  let cppNamespace = "::test";
  let cppIncludes = ["test/TestAttrInterface.h"];
}
def TestAttrTrait : NativeAttrTrait<"TestAttrTrait"> {
  let cppIncludes = ["test/TestAttrTrait.h"];
}

// CHECK-LABEL: #ifdef GET_ATTRDEF_CLASSES
// CHECK-NEXT: #undef GET_ATTRDEF_CLASSES
// CHECK-EMPTY:
// CHECK-NEXT: #include "mlir/IR/Attributes.h"
// CHECK-NEXT: #include "mlir/IR/BuiltinAttributeInterfaces.h"
// CHECK-NEXT: #include "test/TestAttrInterface.h"
// CHECK-NEXT: #include "test/TestAttrTrait.h"
// CHECK-EMPTY:
// CHECK-NOT: #include

def TestAttr : AttrDef<Test_Dialect, "Test", [
    TestAttrTrait, TestAttrInterface]> {
  let mnemonic = "test";
}

// A generated mnemonic alias implements OpAsmAttrInterface.
def AliasAttr : AttrDef<Test_Dialect, "Alias"> {
  let mnemonic = "alias";
  let genMnemonicAlias = 1;
}
