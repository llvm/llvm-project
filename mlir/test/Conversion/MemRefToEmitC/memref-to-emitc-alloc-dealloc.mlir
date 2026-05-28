// RUN: mlir-opt -convert-memref-to-emitc="lower-to-cpp=true" %s -split-input-file | FileCheck %s --check-prefix=CPP
// RUN: mlir-opt -convert-memref-to-emitc="lower-to-cpp=false" %s -split-input-file | FileCheck %s --check-prefix=NOCPP

func.func @alloc() {
  %alloc = memref.alloc() : memref<999xi32>
  return
}

// CPP:      module {
// CPP-NEXT:   emitc.include <"cstdlib">
// CPP-LABEL:  alloc()
// CPP-NEXT:   %[[ALLOC:.*]] = emitc.call_opaque "sizeof"() {args = [i32]} : () -> !emitc.size_t 
// CPP-NEXT:   %[[ALLOC_SIZE:.*]] = "emitc.constant"() <{value = 999 : index}> : () -> index
// CPP-NEXT:   %[[ALLOC_TOTAL_SIZE:.*]] = emitc.mul %[[ALLOC]], %[[ALLOC_SIZE]] : (!emitc.size_t, index) -> !emitc.size_t
// CPP-NEXT:   %[[ALLOC_PTR:.*]] = emitc.call_opaque "malloc"(%[[ALLOC_TOTAL_SIZE]]) : (!emitc.size_t) -> !emitc.ptr<!emitc.opaque<"void">>
// CPP-NEXT:   %[[ALLOC_CAST:.*]] = emitc.cast %[[ALLOC_PTR]] : !emitc.ptr<!emitc.opaque<"void">> to !emitc.ptr<i32>
// CPP-NEXT:   return

// NOCPP:      module {
// NOCPP-NEXT:   emitc.include <"stdlib.h">
// NOCPP-LABEL: alloc()
// NOCPP-NEXT:   %[[ALLOC:.*]] = emitc.call_opaque "sizeof"() {args = [i32]} : () -> !emitc.size_t 
// NOCPP-NEXT:   %[[ALLOC_SIZE:.*]] = "emitc.constant"() <{value = 999 : index}> : () -> index
// NOCPP-NEXT:   %[[ALLOC_TOTAL_SIZE:.*]] = emitc.mul %[[ALLOC]], %[[ALLOC_SIZE]] : (!emitc.size_t, index) -> !emitc.size_t
// NOCPP-NEXT:   %[[ALLOC_PTR:.*]] = emitc.call_opaque "malloc"(%[[ALLOC_TOTAL_SIZE]]) : (!emitc.size_t) -> !emitc.ptr<!emitc.opaque<"void">>
// NOCPP-NEXT:   %[[ALLOC_CAST:.*]] = emitc.cast %[[ALLOC_PTR]] : !emitc.ptr<!emitc.opaque<"void">> to !emitc.ptr<i32>
// NOCPP-NEXT:   return

func.func @alloc_aligned() {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<999xf32>
  return
}

// CPP-LABEL: alloc_aligned
// CPP-NEXT: %[[ALLOC:.*]] = emitc.call_opaque "sizeof"() {args = [f32]} : () -> !emitc.size_t 
// CPP-NEXT: %[[ALLOC_SIZE:.*]] = "emitc.constant"() <{value = 999 : index}> : () -> index
// CPP-NEXT: %[[ALLOC_TOTAL_SIZE:.*]] = emitc.mul %[[ALLOC]], %[[ALLOC_SIZE]] : (!emitc.size_t, index) -> !emitc.size_t
// CPP-NEXT: %[[ALIGNMENT:.*]] = "emitc.constant"() <{value = 64 : index}> : () -> !emitc.size_t 
// CPP-NEXT: %[[ALLOC_PTR:.*]] = emitc.call_opaque "aligned_alloc"(%[[ALIGNMENT]], %[[ALLOC_TOTAL_SIZE]]) : (!emitc.size_t, !emitc.size_t) -> !emitc.ptr<!emitc.opaque<"void">>
// CPP-NEXT: %[[ALLOC_CAST:.*]] = emitc.cast %[[ALLOC_PTR]] : !emitc.ptr<!emitc.opaque<"void">> to !emitc.ptr<f32>
// CPP-NEXT: return

// NOCPP-LABEL: alloc_aligned
// NOCPP-NEXT: %[[ALLOC:.*]] = emitc.call_opaque "sizeof"() {args = [f32]} : () -> !emitc.size_t 
// NOCPP-NEXT: %[[ALLOC_SIZE:.*]] = "emitc.constant"() <{value = 999 : index}> : () -> index
// NOCPP-NEXT: %[[ALLOC_TOTAL_SIZE:.*]] = emitc.mul %[[ALLOC]], %[[ALLOC_SIZE]] : (!emitc.size_t, index) -> !emitc.size_t
// NOCPP-NEXT: %[[ALIGNMENT:.*]] = "emitc.constant"() <{value = 64 : index}> : () -> !emitc.size_t 
// NOCPP-NEXT: %[[ALLOC_PTR:.*]] = emitc.call_opaque "aligned_alloc"(%[[ALIGNMENT]], %[[ALLOC_TOTAL_SIZE]]) : (!emitc.size_t, !emitc.size_t) -> !emitc.ptr<!emitc.opaque<"void">>
// NOCPP-NEXT: %[[ALLOC_CAST:.*]] = emitc.cast %[[ALLOC_PTR]] : !emitc.ptr<!emitc.opaque<"void">> to !emitc.ptr<f32>
// NOCPP-NEXT: return

func.func @allocating_multi() {
  %alloc_5 = memref.alloc() : memref<7x999xi32>
  return
}

// CPP-LABEL: allocating_multi
// CPP-NEXT: %[[ALLOC:.*]] = emitc.call_opaque "sizeof"() {args = [i32]} : () -> !emitc.size_t 
// CPP-NEXT: %[[ALLOC_SIZE:.*]] = "emitc.constant"() <{value = 6993 : index}> : () -> index
// CPP-NEXT: %[[ALLOC_TOTAL_SIZE:.*]] = emitc.mul %[[ALLOC]], %[[ALLOC_SIZE]] : (!emitc.size_t, index) -> !emitc.size_t
// CPP-NEXT: %[[ALLOC_PTR:.*]] = emitc.call_opaque "malloc"(%[[ALLOC_TOTAL_SIZE]]) : (!emitc.size_t) -> !emitc.ptr<!emitc.opaque<"void">
// CPP-NEXT: %[[ALLOC_CAST:.*]] = emitc.cast %[[ALLOC_PTR]] : !emitc.ptr<!emitc.opaque<"void">> to !emitc.ptr<i32>
// CPP-NEXT: return 

// NOCPP-LABEL: allocating_multi
// NOCPP-NEXT: %[[ALLOC:.*]] = emitc.call_opaque "sizeof"() {args = [i32]} : () -> !emitc.size_t 
// NOCPP-NEXT: %[[ALLOC_SIZE:.*]] = "emitc.constant"() <{value = 6993 : index}> : () -> index
// NOCPP-NEXT: %[[ALLOC_TOTAL_SIZE:.*]] = emitc.mul %[[ALLOC]], %[[ALLOC_SIZE]] : (!emitc.size_t, index) -> !emitc.size_t
// NOCPP-NEXT: %[[ALLOC_PTR:.*]] = emitc.call_opaque "malloc"(%[[ALLOC_TOTAL_SIZE]]) : (!emitc.size_t) -> !emitc.ptr<!emitc.opaque<"void">>
// NOCPP-NEXT: %[[ALLOC_CAST:.*]] = emitc.cast %[[ALLOC_PTR]] : !emitc.ptr<!emitc.opaque<"void">> to !emitc.ptr<i32>
// NOCPP-NEXT: return 

