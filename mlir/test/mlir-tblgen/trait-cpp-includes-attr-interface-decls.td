// RUN: mlir-tblgen -gen-attr-interface-decls -I %S/../../include %s | FileCheck %s

include "mlir/IR/Interfaces.td"

// An interface declaration doesn't include its own header.

// CHECK: #include "mlir/IR/Attributes.h"
// CHECK-EMPTY:
// CHECK-NOT: #include

def TestAttrInterface : AttrInterface<"TestAttrInterface"> {
  let cppNamespace = "::test";
  let cppIncludes = ["test/TestAttrInterface.h"];
}
