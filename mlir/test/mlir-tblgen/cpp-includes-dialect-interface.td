// RUN: mlir-tblgen -gen-dialect-interface-decls -I %S/../../include %s | FileCheck %s

include "mlir/IR/Interfaces.td"

// Dialect interface declarations include the header declaring the dialect
// interface base class.

// CHECK: #include "mlir/IR/DialectInterface.h"
// CHECK-EMPTY:
// CHECK-NOT: #include

def TestDialectInterface : DialectInterface<"TestDialectInterface"> {
  let cppNamespace = "::test";
}
