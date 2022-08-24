// RUN: mlir-opt -emit-bytecode %s | mlir-opt | FileCheck %s

// Bytecode currently does not support big-endian platforms
// UNSUPPORTED: s390x-

// CHECK-LABEL: @TestArray
module @TestArray attributes {
  // CHECK: bytecode.array = [unit]
  bytecode.array = [unit]
} {}

// CHECK-LABEL: @TestString
module @TestString attributes {
  // CHECK: bytecode.string = "hello"
  bytecode.string = "hello"
} {}
