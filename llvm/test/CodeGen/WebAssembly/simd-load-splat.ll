; RUN: llc < %s -asm-verbose=false -verify-machineinstrs -disable-wasm-fallthrough-return-opt -wasm-keep-registers -wasm-disable-explicit-locals -mattr=+simd128 | FileCheck %s

; Regression test for an ISel failure when a splatted load had more
; than one use. The main tests for load_splat are in simd-offset.ll.

target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: load_splat:
; CHECK-NEXT: .functype load_splat (i32, i32) -> (i32)
; CHECK-NEXT: i32.load8_u $push[[E:[0-9]+]]=, 0($0){{$}}
; CHECK-NEXT: local.tee $push[[T:[0-9]+]]=, $[[R:[0-9]+]]=, $pop[[E]]{{$}}
; CHECK-NEXT: i8x16.splat $push[[V:[0-9]+]]=, $pop[[T]]{{$}}
; CHECK-NEXT: v128.store 0($1), $pop[[V]]{{$}}
; CHECK-NEXT: return $[[R]]{{$}}
define i8 @load_splat(ptr %p, ptr %out) {
  %e = load i8, ptr %p
  %v1 = insertelement <16 x i8> undef, i8 %e, i32 0
  %v2 = shufflevector <16 x i8> %v1, <16 x i8> undef, <16 x i32> zeroinitializer
  store <16 x i8> %v2, ptr %out
  ret i8 %e
}
