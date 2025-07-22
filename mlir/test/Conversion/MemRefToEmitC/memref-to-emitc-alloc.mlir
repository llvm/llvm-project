// RUN: mlir-opt -convert-memref-to-emitc="lower-to-cpp=true" %s -split-input-file | FileCheck %s

func.func @alloc() {
  %alloc = memref.alloc() : memref<999xi32>
  return
}

// CHECK:module {
// CHECK-NEXT: emitc.include <"cstdlib">
// CHECK-LABEL: alloc()
// CHECK-NEXT: %[[ALLOC:.*]] = emitc.call_opaque "sizeof"() {args = [i32]} : () -> !emitc.size_t 
// CHECK-NEXT: %[[ALLOC:.*]] = "emitc.constant"() <{value = 32 : index}> : () -> index
// CHECK-NEXT: %[[ALLOC:.*]] = emitc.mul %0, %1 : (!emitc.size_t, index) -> !emitc.size_t
// CHECK-NEXT: %[[ALLOC:.*]] = "emitc.constant"() <{value = 32 : index}> : () -> !emitc.size_t 
// CHECK-NEXT: %[[ALLOC:.*]] = emitc.call_opaque "aligned_alloc"(%3, %2) : (!emitc.size_t, !emitc.size_t) -> !emitc.ptr<!emitc.opaque<"void">>
// CHECK-NEXT: %[[ALLOC:.*]] = emitc.cast %4 : !emitc.ptr<!emitc.opaque<"void">> to !emitc.ptr<i32>
// CHECK-NEXT: return 