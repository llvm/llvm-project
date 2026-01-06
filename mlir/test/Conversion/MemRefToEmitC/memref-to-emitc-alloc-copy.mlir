// RUN: mlir-opt -convert-memref-to-emitc="lower-to-cpp=true" %s -split-input-file | FileCheck %s --check-prefixes=CPP,CHECK
// RUN: mlir-opt -convert-memref-to-emitc="lower-to-cpp=false" %s -split-input-file | FileCheck %s --check-prefixes=NOCPP,CHECK

func.func @alloc_copy(%arg0: memref<999xi32>) {
  %alloc = memref.alloc() : memref<999xi32>
  memref.copy %arg0, %alloc : memref<999xi32> to memref<999xi32>
  %alloc_1 = memref.alloc() : memref<999xi32>
  memref.copy %arg0, %alloc_1 : memref<999xi32> to memref<999xi32>
  return
} 

// NOCPP:  emitc.include <"stdlib.h">
// NOCPP-NEXT:  emitc.include <"string.h">

// CPP:  emitc.include <"cstdlib">
// CPP-NEXT:  emitc.include <"cstring">

// CHECK-LABEL: alloc_copy
// CHECK-SAME: %[[arg0:.*]]: memref<999xi32>
// CHECK-NEXT:  builtin.unrealized_conversion_cast %arg0 : memref<999xi32> to !emitc.array<999xi32> 
// CHECK-NEXT:  emitc.call_opaque "sizeof"() {args = [i32]} : () -> !emitc.size_t 
// CHECK-NEXT:  "emitc.constant"() <{value = 999 : index}> : () -> index 
// CHECK-NEXT:  emitc.mul %1, %2 : (!emitc.size_t, index) -> !emitc.size_t 
// CHECK-NEXT:  emitc.call_opaque "malloc"(%3) : (!emitc.size_t) -> !emitc.ptr<!emitc.opaque<"void">> 
// CHECK-NEXT:  emitc.cast %4 : !emitc.ptr<!emitc.opaque<"void">> to !emitc.ptr<!emitc.array<999xi32>>  
// CHECK-NEXT:  emitc.call_opaque "sizeof"() {args = [i32]} : () -> !emitc.size_t
// CHECK-NEXT:  "emitc.constant"() <{value = 999 : index}> : () -> index
// CHECK-NEXT:  emitc.mul %6, %7 : (!emitc.size_t, index) -> !emitc.size_t
// CHECK-NEXT:  emitc.call_opaque "memcpy"(%5, %0, %8) : (!emitc.ptr<i32>, !emitc.ptr<i32>, !emitc.size_t) -> ()
// CHECK-NEXT:  emitc.call_opaque "sizeof"() {args = [i32]} : () -> !emitc.size_t
// CHECK-NEXT:  "emitc.constant"() <{value = 999 : index}> : () -> index
// CHECK-NEXT:  emitc.mul %9, %10 : (!emitc.size_t, index) -> !emitc.size_t
// CHECK-NEXT:  emitc.call_opaque "malloc"(%11) : (!emitc.size_t) -> !emitc.ptr<!emitc.opaque<"void">>
// CHECK-NEXT:  emitc.cast %12 : !emitc.ptr<!emitc.opaque<"void">> to !emitc.ptr<!emitc.array<999xi32>>
// CHECK-NEXT:  emitc.call_opaque "sizeof"() {args = [i32]} : () -> !emitc.size_t
// CHECK-NEXT:  "emitc.constant"() <{value = 999 : index}> : () -> index
// CHECK-NEXT:  emitc.mul %14, %15 : (!emitc.size_t, index) -> !emitc.size_t
// CHECK-NEXT:  emitc.call_opaque "memcpy"(%13, %0, %16) : (!emitc.ptr<i32>, !emitc.ptr<i32>, !emitc.size_t) -> ()
// CHECK-NEXT:    return
