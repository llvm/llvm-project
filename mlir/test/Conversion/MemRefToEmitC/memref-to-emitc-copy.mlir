// RUN: mlir-opt -convert-memref-to-emitc %s  | FileCheck %s

func.func @copying(%arg0 : memref<2x4xf32>) {
  memref.copy %arg0, %arg0 : memref<2x4xf32> to memref<2x4xf32>
  return
}

// CHECK: module {
// CHECK-NEXT:  emitc.include <"string.h">
// CHECK-LABEL:  copying
// CHECK-NEXT: %0 = builtin.unrealized_conversion_cast %arg0 : memref<2x4xf32> to !emitc.array<2x4xf32>
// CHECK-NEXT: %1 = "emitc.constant"() <{value = 0 : index}> : () -> index
// CHECK-NEXT: %2 = emitc.subscript %0[%1, %1] : (!emitc.array<2x4xf32>, index, index) -> !emitc.lvalue<f32>
// CHECK-NEXT: %3 = emitc.apply "&"(%2) : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
// CHECK-NEXT: %4 = emitc.subscript %0[%1, %1] : (!emitc.array<2x4xf32>, index, index) -> !emitc.lvalue<f32>
// CHECK-NEXT: %5 = emitc.apply "&"(%4) : (!emitc.lvalue<f32>) -> !emitc.ptr<f32>
// CHECK-NEXT: %6 = emitc.call_opaque "sizeof"() {args = [f32]} : () -> !emitc.size_t
// CHECK-NEXT: %7 = "emitc.constant"() <{value = 8 : index}> : () -> index
// CHECK-NEXT: %8 = emitc.mul %6, %7 : (!emitc.size_t, index) -> !emitc.size_t
// CHECK-NEXT: emitc.call_opaque "memcpy"(%5, %3, %8) : (!emitc.ptr<f32>, !emitc.ptr<f32>, !emitc.size_t) -> ()
// CHECK-NEXT:    return
// CHECK-NEXT:  }
// CHECK-NEXT:}

