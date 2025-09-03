// RUN: mlir-opt -convert-memref-to-emitc="lower-to-cpp=true" %s -split-input-file | FileCheck %s --check-prefix=CPP
// RUN: mlir-opt -convert-memref-to-emitc="lower-to-cpp=false" %s -split-input-file | FileCheck %s --check-prefix=NOCPP

func.func @alloc_copy(%arg0: memref<999xi32>) {
  %alloc = memref.alloc() : memref<999xi32>
  memref.copy %arg0, %alloc : memref<999xi32> to memref<999xi32>
  %alloc_1 = memref.alloc() : memref<999xi32>
  memref.copy %arg0, %alloc_1 : memref<999xi32> to memref<999xi32>
  return
} 

// CHECK: module {
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
// CHECK-NEXT:  emitc.cast %4 : !emitc.ptr<!emitc.opaque<"void">> to !emitc.ptr<i32> 
// CHECK-NEXT:  builtin.unrealized_conversion_cast %5 : !emitc.ptr<i32> to !emitc.array<999xi32> 
// CHECK-NEXT:  "emitc.constant"() <{value = 0 : index}> : () -> index 
// CHECK-NEXT:  emitc.subscript %0[%7] : (!emitc.array<999xi32>, index) -> !emitc.lvalue<i32> 
// CHECK-NEXT:  emitc.apply "&"(%8) : (!emitc.lvalue<i32>) -> !emitc.ptr<i32> 
// CHECK-NEXT:  emitc.call_opaque "sizeof"() {args = [i32]} : () -> !emitc.size_t
// CHECK-NEXT:  "emitc.constant"() <{value = 999 : index}> : () -> index
// CHECK-NEXT:  emitc.mul %12, %13 : (!emitc.size_t, index) -> !emitc.size_t
// CHECK-NEXT:  emitc.call_opaque "memcpy"(%11, %9, %14) : (!emitc.ptr<i32>, !emitc.ptr<i32>, !emitc.size_t) -> ()
// CHECK-NEXT:  emitc.call_opaque "sizeof"() {args = [i32]} : () -> !emitc.size_t
// CHECK-NEXT:  "emitc.constant"() <{value = 999 : index}> : () -> index
// CHECK-NEXT:  emitc.mul %15, %16 : (!emitc.size_t, index) -> !emitc.size_t
// CHECK-NEXT:  emitc.call_opaque "malloc"(%17) : (!emitc.size_t) -> !emitc.ptr<!emitc.opaque<"void">>
// CHECK-NEXT:  emitc.cast %18 : !emitc.ptr<!emitc.opaque<"void">> to !emitc.ptr<i32>
// CHECK-NEXT:  builtin.unrealized_conversion_cast %19 : !emitc.ptr<i32> to !emitc.array<999xi32>
// CHECK-NEXT:  "emitc.constant"() <{value = 0 : index}> : () -> index
// CHECK-NEXT:  emitc.subscript %0[%21] : (!emitc.array<999xi32>, index) -> !emitc.lvalue<i32>
// CHECK-NEXT:  emitc.apply "&"(%22) : (!emitc.lvalue<i32>) -> !emitc.ptr<i32>
// CHECK-NEXT:  emitc.subscript %20[%21] : (!emitc.array<999xi32>, index) -> !emitc.lvalue<i32>
// CHECK-NEXT:  emitc.apply "&"(%24) : (!emitc.lvalue<i32>) -> !emitc.ptr<i32>
// CHECK-NEXT:  emitc.call_opaque "sizeof"() {args = [i32]} : () -> !emitc.size_t
// CHECK-NEXT:  "emitc.constant"() <{value = 999 : index}> : () -> index
// CHECK-NEXT:  emitc.mul %26, %27 : (!emitc.size_t, index) -> !emitc.size_t
// CHECK-NEXT:  emitc.call_opaque "memcpy"(%25, %23, %28) : (!emitc.ptr<i32>, !emitc.ptr<i32>, !emitc.size_t) -> ()
// CHECK-NEXT:    return
