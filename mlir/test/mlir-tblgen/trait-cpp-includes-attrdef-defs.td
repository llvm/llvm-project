// RUN: mlir-tblgen -gen-attrdef-defs -I %S/../../include %s | FileCheck %s

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

// Attribute and type definitions include the headers they need beyond the
// declarations, but not the headers of the traits, which are included by the
// declarations.

// CHECK-LABEL: #ifdef GET_ATTRDEF_CLASSES
// CHECK-NEXT: #undef GET_ATTRDEF_CLASSES
// CHECK-EMPTY:
// CHECK-NEXT: #include "llvm/ADT/TypeSwitch.h"
// CHECK-NEXT: #include "mlir/IR/Builders.h"
// CHECK-NEXT: #include "mlir/IR/DialectImplementation.h"
// CHECK-NEXT: #include "mlir/IR/OpImplementation.h"
// CHECK-EMPTY:
// CHECK-NOT: #include

def TestAttr : AttrDef<Test_Dialect, "Test", [
    TestAttrTrait, TestAttrInterface]> {
  let mnemonic = "test";
}
