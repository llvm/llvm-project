// RUN: mlir-opt -convert-memref-to-emitc="lower-to-cpp=true" %s | FileCheck %s --check-prefix=CPP
// RUN: mlir-opt -convert-memref-to-emitc="lower-to-cpp=false" %s | FileCheck %s --check-prefix=NOCPP

/// Tests for converting `memref.alloc` and `memref.dealloc`.
/// At the moment, `memref.dealloc` lowering only accepts the pointer-backed form
/// produced by the current `memref.alloc` lowering, so alloc and dealloc tests
/// are kept together.

func.func @alloc_and_dealloc() {
  %alloc = memref.alloc() : memref<999xi32>
  memref.dealloc %alloc : memref<999xi32>
  return
}

// CPP:        emitc.include <"cstdlib">
// CPP-LABEL:  alloc_and_dealloc
// CPP:          %[[ALLOC:.*]] = emitc.call_opaque "sizeof"() {args = [i32]} : () -> !emitc.size_t
// CPP:          %[[ALLOC_SIZE:.*]] = "emitc.constant"() <{value = 999 : index}> : () -> index
// CPP:          %[[ALLOC_TOTAL_SIZE:.*]] = emitc.mul %[[ALLOC]], %[[ALLOC_SIZE]]
// CPP-SAME:       : (!emitc.size_t, index) -> !emitc.size_t
// CPP:          %[[ALLOC_PTR:.*]] = emitc.call_opaque "malloc"(%[[ALLOC_TOTAL_SIZE]])
// CPP-SAME:       : (!emitc.size_t) -> !emitc.ptr<!emitc.opaque<"void">>
// CPP:          %[[ALLOC_CAST:.*]] = emitc.cast %[[ALLOC_PTR]]
// CPP-SAME:       : !emitc.ptr<!emitc.opaque<"void">> to !emitc.ptr<i32>
// CPP:          %[[FREE_PTR:.*]] = emitc.cast %[[ALLOC_CAST]]
// CPP-SAME:       : !emitc.ptr<i32> to !emitc.ptr<!emitc.opaque<"void">>
// CPP:          emitc.call_opaque "free"(%[[FREE_PTR]]) : (!emitc.ptr<!emitc.opaque<"void">>) -> ()
// CPP:          return

// NOCPP:        emitc.include <"stdlib.h">
// NOCPP-LABEL:  alloc_and_dealloc
// NOCPP:          %[[ALLOC:.*]] = emitc.call_opaque "sizeof"() {args = [i32]} : () -> !emitc.size_t
// NOCPP:          %[[ALLOC_SIZE:.*]] = "emitc.constant"() <{value = 999 : index}> : () -> index
// NOCPP:          %[[ALLOC_TOTAL_SIZE:.*]] = emitc.mul %[[ALLOC]], %[[ALLOC_SIZE]]
// NOCPP-SAME:       : (!emitc.size_t, index) -> !emitc.size_t
// NOCPP:          %[[ALLOC_PTR:.*]] = emitc.call_opaque "malloc"(%[[ALLOC_TOTAL_SIZE]])
// NOCPP-SAME:       : (!emitc.size_t) -> !emitc.ptr<!emitc.opaque<"void">>
// NOCPP:          %[[ALLOC_CAST:.*]] = emitc.cast %[[ALLOC_PTR]]
// NOCPP-SAME:       : !emitc.ptr<!emitc.opaque<"void">> to !emitc.ptr<i32>
// NOCPP:          %[[FREE_PTR:.*]] = emitc.cast %[[ALLOC_CAST]]
// NOCPP-SAME:       : !emitc.ptr<i32> to !emitc.ptr<!emitc.opaque<"void">>
// NOCPP:          emitc.call_opaque "free"(%[[FREE_PTR]]) : (!emitc.ptr<!emitc.opaque<"void">>) -> ()
// NOCPP:          return

func.func @alloc_and_dealloc_aligned() {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<999xf32>
  memref.dealloc %alloc : memref<999xf32>
  return
}

// CPP-LABEL: alloc_and_dealloc_aligned
// CPP:         %[[ALLOC:.*]] = emitc.call_opaque "sizeof"() {args = [f32]} : () -> !emitc.size_t 
// CPP:         %[[ALLOC_SIZE:.*]] = "emitc.constant"() <{value = 999 : index}> : () -> index
// CPP:         %[[ALLOC_TOTAL_SIZE:.*]] = emitc.mul %[[ALLOC]], %[[ALLOC_SIZE]]
// CPP-SAME:       : (!emitc.size_t, index) -> !emitc.size_t
// CPP:         %[[ALIGNMENT:.*]] = "emitc.constant"() <{value = 64 : index}> : () -> !emitc.size_t 
// CPP:         %[[ALLOC_PTR:.*]] = emitc.call_opaque "aligned_alloc"(%[[ALIGNMENT]], %[[ALLOC_TOTAL_SIZE]])
// CPP-SAME:       : (!emitc.size_t, !emitc.size_t) -> !emitc.ptr<!emitc.opaque<"void">>
// CPP:         %[[ALLOC_CAST:.*]] = emitc.cast %[[ALLOC_PTR]]
// CPP-SAME:       : !emitc.ptr<!emitc.opaque<"void">> to !emitc.ptr<f32>
// CPP:         %[[FREE_PTR:.*]] = emitc.cast %[[ALLOC_CAST]]
// CPP-SAME:       : !emitc.ptr<f32> to !emitc.ptr<!emitc.opaque<"void">>
// CPP:         emitc.call_opaque "free"(%[[FREE_PTR]]) : (!emitc.ptr<!emitc.opaque<"void">>) -> ()
// CPP:         return

// NOCPP-LABEL: alloc_and_dealloc_aligned
// NOCPP:         %[[ALLOC:.*]] = emitc.call_opaque "sizeof"() {args = [f32]} : () -> !emitc.size_t 
// NOCPP:         %[[ALLOC_SIZE:.*]] = "emitc.constant"() <{value = 999 : index}> : () -> index
// NOCPP:         %[[ALLOC_TOTAL_SIZE:.*]] = emitc.mul %[[ALLOC]], %[[ALLOC_SIZE]]
// NOCPP-SAME:       : (!emitc.size_t, index) -> !emitc.size_t
// NOCPP:         %[[ALIGNMENT:.*]] = "emitc.constant"() <{value = 64 : index}> : () -> !emitc.size_t 
// NOCPP:         %[[ALLOC_PTR:.*]] = emitc.call_opaque "aligned_alloc"(%[[ALIGNMENT]], %[[ALLOC_TOTAL_SIZE]])
// NOCPP-SAME:      : (!emitc.size_t, !emitc.size_t) -> !emitc.ptr<!emitc.opaque<"void">>
// NOCPP:         %[[ALLOC_CAST:.*]] = emitc.cast %[[ALLOC_PTR]]
// NOCPP-SAME:      : !emitc.ptr<!emitc.opaque<"void">> to !emitc.ptr<f32>
// NOCPP:         %[[FREE_PTR:.*]] = emitc.cast %[[ALLOC_CAST]]
// NOCPP-SAME:      : !emitc.ptr<f32> to !emitc.ptr<!emitc.opaque<"void">>
// NOCPP:         emitc.call_opaque "free"(%[[FREE_PTR]]) : (!emitc.ptr<!emitc.opaque<"void">>) -> ()
// NOCPP:         return

func.func @allocating_and_deallocating_multi() {
  %alloc = memref.alloc() : memref<7x999xi32>
  memref.dealloc %alloc : memref<7x999xi32>
  return
}

// CPP-LABEL: allocating_and_deallocating_multi
// CPP:         %[[ALLOC:.*]] = emitc.call_opaque "sizeof"() {args = [i32]} : () -> !emitc.size_t 
// CPP:         %[[ALLOC_SIZE:.*]] = "emitc.constant"() <{value = 6993 : index}> : () -> index
// CPP:         %[[ALLOC_TOTAL_SIZE:.*]] = emitc.mul %[[ALLOC]], %[[ALLOC_SIZE]]
// CPP-SAME:       : (!emitc.size_t, index) -> !emitc.size_t
// CPP:         %[[ALLOC_PTR:.*]] = emitc.call_opaque "malloc"(%[[ALLOC_TOTAL_SIZE]])
// CPP-SAME:       : (!emitc.size_t) -> !emitc.ptr<!emitc.opaque<"void">
// CPP:         %[[ALLOC_CAST:.*]] = emitc.cast %[[ALLOC_PTR]]
// CPP-SAME:       : !emitc.ptr<!emitc.opaque<"void">> to !emitc.ptr<i32>
// CPP:         %[[FREE_PTR:.*]] = emitc.cast %[[ALLOC_CAST]]
// CPP-SAME:       : !emitc.ptr<i32> to !emitc.ptr<!emitc.opaque<"void">>
// CPP:         emitc.call_opaque "free"(%[[FREE_PTR]]) : (!emitc.ptr<!emitc.opaque<"void">>) -> ()
// CPP:         return

// NOCPP-LABEL: allocating_and_deallocating_multi
// NOCPP:         %[[ALLOC:.*]] = emitc.call_opaque "sizeof"() {args = [i32]} : () -> !emitc.size_t 
// NOCPP:         %[[ALLOC_SIZE:.*]] = "emitc.constant"() <{value = 6993 : index}> : () -> index
// NOCPP:         %[[ALLOC_TOTAL_SIZE:.*]] = emitc.mul %[[ALLOC]], %[[ALLOC_SIZE]]
// NOCPP-SAME:      : (!emitc.size_t, index) -> !emitc.size_t
// NOCPP:         %[[ALLOC_PTR:.*]] = emitc.call_opaque "malloc"(%[[ALLOC_TOTAL_SIZE]])
// NOCPP-SAME:      : (!emitc.size_t) -> !emitc.ptr<!emitc.opaque<"void">>
// NOCPP:         %[[ALLOC_CAST:.*]] = emitc.cast %[[ALLOC_PTR]]
// NOCPP-SAME:      : !emitc.ptr<!emitc.opaque<"void">> to !emitc.ptr<i32>
// NOCPP:         %[[FREE_PTR:.*]] = emitc.cast %[[ALLOC_CAST]]
// NOCPP-SAME:      : !emitc.ptr<i32> to !emitc.ptr<!emitc.opaque<"void">>
// NOCPP:         emitc.call_opaque "free"(%[[FREE_PTR]]) : (!emitc.ptr<!emitc.opaque<"void">>) -> ()
// NOCPP:         return
