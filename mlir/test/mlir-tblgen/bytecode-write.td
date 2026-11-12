// RUN: mlir-tblgen -gen-bytecode -bytecode-dialect=Test -I %S/../../include %s 2>&1 | FileCheck %s

// Test to show fall through behavior.

include "mlir/IR/BytecodeBase.td"

def Float64Type : DialectType<(type)> {
  let printerPredicate = "false";
}

def TestDialectTypes : DialectTypes<"Test"> {
  let elems = [
      ReservedOrDead,
      Float64Type
  ];
}

// CHECK-LABEL: static ::llvm::LogicalResult write(Float64Type type, ::mlir::DialectBytecodeWriter &writer) {
// CHECK:         if (false) {
// CHECK:           writer.writeVarInt(/* Float64Type */ 1);
// CHECK:           return ::mlir::success();
// CHECK:         }
// CHECK:         return ::mlir::failure();
// CHECK-LABEL: static ::llvm::LogicalResult writeType(::mlir::Type type,
// CHECK:         return ::llvm::TypeSwitch<::mlir::Type, ::llvm::LogicalResult>(type)
// CHECK-NEXT:       .Case([&](Float64Type t) {
// CHECK-NEXT:         return write(t, writer);

