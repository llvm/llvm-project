// RUN: mlir-tblgen -gen-bytecode -bytecode-dialect=Test -I %S/../../include %s | FileCheck %s

include "mlir/IR/BytecodeBase.td"

// The bytecode readers and writers include the headers declaring the bytecode
// reader and writer and `TypeSwitch`, which is used by the write dispatch.

// CHECK: #include "llvm/ADT/TypeSwitch.h"
// CHECK-NEXT: #include "mlir/Bytecode/BytecodeImplementation.h"
// CHECK-EMPTY:
// CHECK-NOT: #include

def TestDialectTypes : DialectTypes<"Test"> {
  let elems = [ReservedOrDead];
}
