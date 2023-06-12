// RUN: mlir-tblgen -gen-bytecode -bytecode-dialect=Test -I %S/../../include %s 2>&1 | FileCheck %s

include "mlir/IR/BuiltinDialectBytecode.td"

def TestDialectTypes : DialectTypes<"Test"> {
// CHECK: static Type readType
  let elems = [
// CHECK:      case 0:
// CHECK-NEXT:   return readIntegerType(context, reader);
      IntegerType,
// No case 1 generated as only reserved.
      ReservedOrDead,
// CHECK-NEXT: case 2:
// CHECK-NEXT:   return readIndexType(context, reader);
      IndexType,
// CHECK-NEXT: case 3:
// CHECK-NEXT:   return readBFloat16Type(context, reader);
      BFloat16Type,
// No case 4 generated as only reserved.
      ReservedOrDead,
// CHECK-NEXT: case 5:
// CHECK-NEXT:   return readFloat16Type(context, reader);
      Float16Type
  ];
}

