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

// CPP:      module {
// CPP-NEXT:   emitc.include <"cstdlib">
// CPP-LABEL:  alloc_and_dealloc()
// CPP-NEXT:   %[[ALLOC:.*]] = emitc.call_opaque "sizeof"() <{args = [i32]}> : () -> !emitc.size_t
// CPP-NEXT:   %[[ALLOC_SIZE:.*]] = "emitc.constant"() <{value = 999 : index}> : () -> index
// CPP-NEXT:   %[[ALLOC_TOTAL_SIZE:.*]] = emitc.mul %[[ALLOC]], %[[ALLOC_SIZE]] : (!emitc.size_t, index) -> !emitc.size_t
// CPP-NEXT:   %[[ALLOC_PTR:.*]] = emitc.call_opaque "malloc"(%[[ALLOC_TOTAL_SIZE]]) : (!emitc.size_t) -> !emitc.ptr<!emitc.opaque<"void">>
// CPP-NEXT:   %[[ALLOC_CAST:.*]] = emitc.cast %[[ALLOC_PTR]] : !emitc.ptr<!emitc.opaque<"void">> to !emitc.ptr<i32>
// CPP-NEXT:   %[[FREE_PTR:.*]] = emitc.cast %[[ALLOC_CAST]] : !emitc.ptr<i32> to !emitc.ptr<!emitc.opaque<"void">>
// CPP-NEXT:   emitc.call_opaque "free"(%[[FREE_PTR]]) : (!emitc.ptr<!emitc.opaque<"void">>) -> ()
// CPP-NEXT:   return

// NOCPP:      module {
// NOCPP-NEXT:   emitc.include <"stdlib.h">
// NOCPP-LABEL:  alloc_and_dealloc()
// NOCPP-NEXT:   %[[ALLOC:.*]] = emitc.call_opaque "sizeof"() <{args = [i32]}> : () -> !emitc.size_t
// NOCPP-NEXT:   %[[ALLOC_SIZE:.*]] = "emitc.constant"() <{value = 999 : index}> : () -> index
// NOCPP-NEXT:   %[[ALLOC_TOTAL_SIZE:.*]] = emitc.mul %[[ALLOC]], %[[ALLOC_SIZE]] : (!emitc.size_t, index) -> !emitc.size_t
// NOCPP-NEXT:   %[[ALLOC_PTR:.*]] = emitc.call_opaque "malloc"(%[[ALLOC_TOTAL_SIZE]]) : (!emitc.size_t) -> !emitc.ptr<!emitc.opaque<"void">>
// NOCPP-NEXT:   %[[ALLOC_CAST:.*]] = emitc.cast %[[ALLOC_PTR]] : !emitc.ptr<!emitc.opaque<"void">> to !emitc.ptr<i32>
// NOCPP-NEXT:   %[[FREE_PTR:.*]] = emitc.cast %[[ALLOC_CAST]] : !emitc.ptr<i32> to !emitc.ptr<!emitc.opaque<"void">>
// NOCPP-NEXT:   emitc.call_opaque "free"(%[[FREE_PTR]]) : (!emitc.ptr<!emitc.opaque<"void">>) -> ()
// NOCPP-NEXT:   return

func.func @alloc_and_dealloc_aligned() {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<999xf32>
  memref.dealloc %alloc : memref<999xf32>
  return
}

// CPP-LABEL: alloc_and_dealloc_aligned
// CPP-NEXT: %[[ALLOC:.*]] = emitc.call_opaque "sizeof"() <{args = [f32]}> : () -> !emitc.size_t
// CPP-NEXT: %[[ALLOC_SIZE:.*]] = "emitc.constant"() <{value = 999 : index}> : () -> index
// CPP-NEXT: %[[ALLOC_TOTAL_SIZE:.*]] = emitc.mul %[[ALLOC]], %[[ALLOC_SIZE]] : (!emitc.size_t, index) -> !emitc.size_t
// CPP-NEXT: %[[ALIGNMENT:.*]] = "emitc.constant"() <{value = 64 : index}> : () -> !emitc.size_t 
// CPP-NEXT: %[[ALLOC_PTR:.*]] = emitc.call_opaque "aligned_alloc"(%[[ALIGNMENT]], %[[ALLOC_TOTAL_SIZE]]) : (!emitc.size_t, !emitc.size_t) -> !emitc.ptr<!emitc.opaque<"void">>
// CPP-NEXT: %[[ALLOC_CAST:.*]] = emitc.cast %[[ALLOC_PTR]] : !emitc.ptr<!emitc.opaque<"void">> to !emitc.ptr<f32>
// CPP-NEXT: %[[FREE_PTR:.*]] = emitc.cast %[[ALLOC_CAST]] : !emitc.ptr<f32> to !emitc.ptr<!emitc.opaque<"void">>
// CPP-NEXT: emitc.call_opaque "free"(%[[FREE_PTR]]) : (!emitc.ptr<!emitc.opaque<"void">>) -> ()
// CPP-NEXT: return

// NOCPP-LABEL: alloc_and_dealloc_aligned
// NOCPP-NEXT: %[[ALLOC:.*]] = emitc.call_opaque "sizeof"() <{args = [f32]}> : () -> !emitc.size_t
// NOCPP-NEXT: %[[ALLOC_SIZE:.*]] = "emitc.constant"() <{value = 999 : index}> : () -> index
// NOCPP-NEXT: %[[ALLOC_TOTAL_SIZE:.*]] = emitc.mul %[[ALLOC]], %[[ALLOC_SIZE]] : (!emitc.size_t, index) -> !emitc.size_t
// NOCPP-NEXT: %[[ALIGNMENT:.*]] = "emitc.constant"() <{value = 64 : index}> : () -> !emitc.size_t 
// NOCPP-NEXT: %[[ALLOC_PTR:.*]] = emitc.call_opaque "aligned_alloc"(%[[ALIGNMENT]], %[[ALLOC_TOTAL_SIZE]]) : (!emitc.size_t, !emitc.size_t) -> !emitc.ptr<!emitc.opaque<"void">>
// NOCPP-NEXT: %[[ALLOC_CAST:.*]] = emitc.cast %[[ALLOC_PTR]] : !emitc.ptr<!emitc.opaque<"void">> to !emitc.ptr<f32>
// NOCPP-NEXT: %[[FREE_PTR:.*]] = emitc.cast %[[ALLOC_CAST]] : !emitc.ptr<f32> to !emitc.ptr<!emitc.opaque<"void">>
// NOCPP-NEXT: emitc.call_opaque "free"(%[[FREE_PTR]]) : (!emitc.ptr<!emitc.opaque<"void">>) -> ()
// NOCPP-NEXT: return

func.func @allocating_and_deallocating_multi() {
  %alloc = memref.alloc() : memref<7x999xi32>
  memref.dealloc %alloc : memref<7x999xi32>
  return
}

// CPP-LABEL: allocating_and_deallocating_multi
// CPP-NEXT: %[[ALLOC:.*]] = emitc.call_opaque "sizeof"() <{args = [i32]}> : () -> !emitc.size_t
// CPP-NEXT: %[[ALLOC_SIZE:.*]] = "emitc.constant"() <{value = 6993 : index}> : () -> index
// CPP-NEXT: %[[ALLOC_TOTAL_SIZE:.*]] = emitc.mul %[[ALLOC]], %[[ALLOC_SIZE]] : (!emitc.size_t, index) -> !emitc.size_t
// CPP-NEXT: %[[ALLOC_PTR:.*]] = emitc.call_opaque "malloc"(%[[ALLOC_TOTAL_SIZE]]) : (!emitc.size_t) -> !emitc.ptr<!emitc.opaque<"void">
// CPP-NEXT: %[[ALLOC_CAST:.*]] = emitc.cast %[[ALLOC_PTR]] : !emitc.ptr<!emitc.opaque<"void">> to !emitc.ptr<i32>
// CPP-NEXT: %[[FREE_PTR:.*]] = emitc.cast %[[ALLOC_CAST]] : !emitc.ptr<i32> to !emitc.ptr<!emitc.opaque<"void">>
// CPP-NEXT: emitc.call_opaque "free"(%[[FREE_PTR]]) : (!emitc.ptr<!emitc.opaque<"void">>) -> ()
// CPP-NEXT: return

// NOCPP-LABEL: allocating_and_deallocating_multi
// NOCPP-NEXT: %[[ALLOC:.*]] = emitc.call_opaque "sizeof"() <{args = [i32]}> : () -> !emitc.size_t
// NOCPP-NEXT: %[[ALLOC_SIZE:.*]] = "emitc.constant"() <{value = 6993 : index}> : () -> index
// NOCPP-NEXT: %[[ALLOC_TOTAL_SIZE:.*]] = emitc.mul %[[ALLOC]], %[[ALLOC_SIZE]] : (!emitc.size_t, index) -> !emitc.size_t
// NOCPP-NEXT: %[[ALLOC_PTR:.*]] = emitc.call_opaque "malloc"(%[[ALLOC_TOTAL_SIZE]]) : (!emitc.size_t) -> !emitc.ptr<!emitc.opaque<"void">>
// NOCPP-NEXT: %[[ALLOC_CAST:.*]] = emitc.cast %[[ALLOC_PTR]] : !emitc.ptr<!emitc.opaque<"void">> to !emitc.ptr<i32>
// NOCPP-NEXT: %[[FREE_PTR:.*]] = emitc.cast %[[ALLOC_CAST]] : !emitc.ptr<i32> to !emitc.ptr<!emitc.opaque<"void">>
// NOCPP-NEXT: emitc.call_opaque "free"(%[[FREE_PTR]]) : (!emitc.ptr<!emitc.opaque<"void">>) -> ()
// NOCPP-NEXT: return
