// RUN: mlir-opt -convert-memref-to-emitc %s -split-input-file | FileCheck %s

func.func @casting(%arg0: memref<999xi32>) {
  %reinterpret_cast_5 = memref.reinterpret_cast %arg0 to offset: [0], sizes: [1, 1, 999], strides: [999, 999, 1] : memref<999xi32> to memref<1x1x999xi32>
  return
}

//CHECK: module {
//CHECK-NEXT:   func.func @casting(%arg0: memref<999xi32>) {
//CHECK-NEXT:     %0 = builtin.unrealized_conversion_cast %arg0 : memref<999xi32> to !emitc.array<999xi32>
//CHECK-NEXT:     %1 = "emitc.constant"() <{value = 0 : index}> : () -> index
//CHECK-NEXT:     %2 = emitc.subscript %0[%1] : (!emitc.array<999xi32>, index) -> !emitc.lvalue<i32>
//CHECK-NEXT:     %3 = emitc.apply "&"(%2) : (!emitc.lvalue<i32>) -> !emitc.ptr<i32>
//CHECK-NEXT:     %4 = emitc.call_opaque "reinterpret_cast"(%3) {args = [0 : index], template_args = [!emitc.ptr<!emitc.array<1x1x999xi32>>]} : (!emitc.ptr<i32>) -> !emitc.ptr<!emitc.array<1x1x999xi32>>
//CHECK-NEXT:     return

