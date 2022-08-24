// RUN: mlir-opt -emit-bytecode %s | mlir-opt | FileCheck %s

// Bytecode currently does not support big-endian platforms
// UNSUPPORTED: s390x-

// CHECK-LABEL: @TestInteger
module @TestInteger attributes {
  // CHECK: bytecode.int = i1024,
  // CHECK: bytecode.int1 = si32,
  // CHECK: bytecode.int2 = ui512
  bytecode.int = i1024,
  bytecode.int1 = si32,
  bytecode.int2 = ui512
} {}

// CHECK-LABEL: @TestIndex
module @TestIndex attributes {
  // CHECK: bytecode.index = index
  bytecode.index = index
} {}

// CHECK-LABEL: @TestFunc
module @TestFunc attributes {
  // CHECK: bytecode.func = () -> (),
  // CHECK: bytecode.func1 = (i1) -> i32
  bytecode.func = () -> (),
  bytecode.func1 = (i1) -> (i32)
} {}
