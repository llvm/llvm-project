// RUN: mlir-tblgen -gen-typedef-decls -I %S/../../include %s | FileCheck %s

include "mlir/IR/AttrTypeBase.td"

def Test_Dialect : Dialect {
  let name = "test";
  let cppNamespace = "::test";
}

def TestTypeTrait : NativeTypeTrait<"TestTypeTrait"> {
  let cppIncludes = ["test/TestTypeTrait.h"];
}

// A generated mnemonic alias implements OpAsmTypeInterface.

// CHECK-LABEL: #ifdef GET_TYPEDEF_CLASSES
// CHECK-NEXT: #undef GET_TYPEDEF_CLASSES
// CHECK-EMPTY:
// CHECK-NEXT: #include "mlir/IR/BuiltinTypeInterfaces.h"
// CHECK-NEXT: #include "mlir/IR/Types.h"
// CHECK-NEXT: #include "test/TestTypeTrait.h"
// CHECK-EMPTY:
// CHECK-NOT: #include

def TestType : TypeDef<Test_Dialect, "Test", [TestTypeTrait]> {
  let mnemonic = "test";
  let genMnemonicAlias = 1;
}
