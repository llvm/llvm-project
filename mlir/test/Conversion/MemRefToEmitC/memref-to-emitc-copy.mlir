// RUN: mlir-opt -convert-memref-to-emitc="lower-to-cpp=true" %s -split-input-file | FileCheck %s --check-prefix=CPP
// RUN: mlir-opt -convert-memref-to-emitc="lower-to-cpp=false" %s -split-input-file | FileCheck %s --check-prefix=NOCPP

func.func @copying(%arg0 : memref<9x4x5x7xf32>, %arg1 : memref<9x4x5x7xf32>) {
  memref.copy %arg0, %arg1 : memref<9x4x5x7xf32> to memref<9x4x5x7xf32>
  return
}

// CHECK: module {
// NOCPP:  emitc.include <"string.h">
// CPP:  emitc.include <"cstring">

// CHECK-LABEL:  copying
// CHECK-SAME: %[[arg0:.*]]: memref<9x4x5x7xf32>, %[[arg1:.*]]: memref<9x4x5x7xf32>
// CHECK-NEXT: %0 = builtin.unrealized_conversion_cast %arg1 : memref<9x4x5x7xf32> to !emitc.array<9x4x5x7xf32>
// CHECK-NEXT: %1 = builtin.unrealized_conversion_cast %arg0 : memref<9x4x5x7xf32> to !emitc.array<9x4x5x7xf32>
// CHECK-NEXT: %2 = "emitc.constant"() <{value = 0 : index}> : () -> index
// CHECK-NEXT: %3 = emitc.subscript %1[%2, %2, %2, %2] : (!emitc.array<9x4x5x7xf32>, index, index, index, index) -> !emitc.lvalue<f32>
// CHECK-NEXT: %4 = emitc.apply "&"(%3) : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
// CHECK-NEXT: %5 = emitc.subscript %0[%2, %2, %2, %2] : (!emitc.array<9x4x5x7xf32>, index, index, index, index) -> !emitc.lvalue<f32>
// CHECK-NEXT: %6 = emitc.apply "&"(%5) : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
// CHECK-NEXT: %7 = emitc.call_opaque "sizeof"() {args = [f32]} : () -> !emitc.size_t
// CHECK-NEXT: %8 = "emitc.constant"() <{value = 1260 : index}> : () -> index
// CHECK-NEXT: %9 = emitc.mul %7, %8 : (!emitc.size_t, index) -> !emitc.size_t
// CHECK-NEXT: emitc.call_opaque "memcpy"(%6, %4, %9) : (!emitc.ptr<f32>, !emitc.ptr<f32>, !emitc.size_t) -> ()
// CHECK-NEXT:    return
// CHECK-NEXT:  }
// CHECK-NEXT:}

