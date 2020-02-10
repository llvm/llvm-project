// RUN: mlir-opt -convert-std-to-llvm %s | FileCheck %s

// CHECK-LABEL: func @check_attributes
// When expanding the memref to multiple arguments, argument attributes are replicated.
// CHECK-COUNT-7: {dialect.a = true, dialect.b = 4 : i64}
func @check_attributes(%static: memref<10x20xf32> {dialect.a = true, dialect.b = 4 : i64 }) {
  %c0 = constant 0 : index
  %0 = load %static[%c0, %c0]: memref<10x20xf32>
  return
}

