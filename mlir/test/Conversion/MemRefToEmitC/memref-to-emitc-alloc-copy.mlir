// RUN: mlir-opt -convert-memref-to-emitc="lower-to-cpp=true" %s -split-input-file | FileCheck %s --check-prefix=CPP
// RUN: mlir-opt -convert-memref-to-emitc="lower-to-cpp=false" %s -split-input-file | FileCheck %s --check-prefix=NOCPP


func.func @alloc_copy(%arg0: memref<999xi32>) {
  %alloc = memref.alloc() : memref<999xi32>
  memref.copy %arg0, %alloc : memref<999xi32> to memref<999xi32>
  return
} 

// NOCPP: module {
// NOCPP-NEXT:  emitc.include <"string.h">
// NOCPP-NEXT:  emitc.include <"stdlib.h">

// CPP: module {
// CPP-NEXT:  emitc.include <"cstring">
// CHECK-NEXT: emitc.include <"cstdlib">
// CHECK-LABEL: alloc_copy
// CHECK-SAME: %[[arg0:.*]]: memref<999xi32>
// CHECK-NEXT:  %0 = builtin.unrealized_conversion_cast %arg0 : memref<999xi32> to !emitc.array<999xi32> 
// CHECK-NEXT:  emitc.call_opaque "sizeof"() {args = [i32]} : () -> !emitc.size_t 
// CHECK-NEXT:  "emitc.constant"() <{value = 999 : index}> : () -> index 
// CHECK-NEXT:  emitc.mul %1, %2 : (!emitc.size_t, index) -> !emitc.size_t 
// CHECK-NEXT:  emitc.call_opaque "malloc"(%3) : (!emitc.size_t) -> !emitc.ptr<!emitc.opaque<"void">> 
// CHECK-NEXT:  emitc.cast %4 : !emitc.ptr<!emitc.opaque<"void">> to !emitc.ptr<i32> 
// CHECK-NEXT:  builtin.unrealized_conversion_cast %5 : !emitc.ptr<i32> to !emitc.array<999xi32> 
// CHECK-NEXT:  "emitc.constant"() <{value = 0 : index}> : () -> index 
// CHECK-NEXT:  emitc.subscript %0[%7] : (!emitc.array<999xi32>, index) -> !emitc.lvalue<i32> 
// CHECK-NEXT:  emitc.apply "&"(%8) : (!emitc.lvalue<i32>) -> !emitc.ptr<i32> 

