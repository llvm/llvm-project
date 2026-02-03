// RUN: mlir-opt -convert-memref-to-emitc="lower-to-cpp=true" %s -split-input-file | FileCheck %s --check-prefixes=CPP,CHECK
// RUN: mlir-opt -convert-memref-to-emitc="lower-to-cpp=false" %s -split-input-file | FileCheck %s --check-prefixes=NOCPP,CHECK

func.func @copying(%arg0 : memref<9x4x5x7xf32>, %arg1 : memref<9x4x5x7xf32>) {
  memref.copy %arg0, %arg1 : memref<9x4x5x7xf32> to memref<9x4x5x7xf32>
  return
}

// CHECK: module {
// NOCPP:  emitc.include <"string.h">
// CPP:  emitc.include <"cstring">

// CHECK-LABEL:  copying
// CHECK-SAME: %[[arg0:.*]]: memref<9x4x5x7xf32>, %[[arg1:.*]]: memref<9x4x5x7xf32>
// CHECK-NEXT: %0 = builtin.unrealized_conversion_cast %arg1 : memref<9x4x5x7xf32> to !emitc.ptr<!emitc.array<9x4x5x7xf32>>
// CHECK-NEXT: %1 = builtin.unrealized_conversion_cast %arg0 : memref<9x4x5x7xf32> to !emitc.ptr<!emitc.array<9x4x5x7xf32>>
// CHECK-NEXT: %2 = emitc.call_opaque "sizeof"() {args = [f32]} : () -> !emitc.size_t
// CHECK-NEXT: %3 = "emitc.constant"() <{value = 1260 : index}> : () -> index
// CHECK-NEXT: %4 = emitc.mul %2, %3 : (!emitc.size_t, index) -> !emitc.size_t
// CHECK-NEXT: emitc.call_opaque "memcpy"(%0, %1, %4) : (!emitc.ptr<!emitc.array<9x4x5x7xf32>>, !emitc.ptr<!emitc.array<9x4x5x7xf32>>, !emitc.size_t) -> ()
// CHECK-NEXT:    return
// CHECK-NEXT:  }
// CHECK-NEXT:}

