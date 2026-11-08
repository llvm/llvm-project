// RUN: mlir-tblgen -gen-op-interface-decls -I %S/../../include %s | FileCheck %s

include "mlir/IR/OpBase.td"

// Interface declarations include the headers of their base interfaces, but not
// their own.

// CHECK: #include "mlir/IR/OpDefinition.h"
// CHECK-NEXT: #include "test/BaseOpInterface.h"
// CHECK-EMPTY:
// CHECK-NOT: #include

def BaseOpInterface : OpInterface<"BaseOpInterface"> {
  let cppNamespace = "::test";
  let cppIncludes = ["test/BaseOpInterface.h"];
}
def DerivedOpInterface : OpInterface<"DerivedOpInterface", [BaseOpInterface]> {
  let cppNamespace = "::test";
  let cppIncludes = ["test/DerivedOpInterface.h"];
}
