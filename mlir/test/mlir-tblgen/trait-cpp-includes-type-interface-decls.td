// RUN: mlir-tblgen -gen-type-interface-decls -I %S/../../include %s | FileCheck %s

include "mlir/IR/Interfaces.td"

// An interface declaration doesn't include its own header.

// CHECK: #include "mlir/IR/Types.h"
// CHECK-EMPTY:
// CHECK-NOT: #include

def TestTypeInterface : TypeInterface<"TestTypeInterface"> {
  let cppNamespace = "::test";
  let cppIncludes = ["test/TestTypeInterface.h"];
}
