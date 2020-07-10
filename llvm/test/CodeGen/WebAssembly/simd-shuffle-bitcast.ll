; RUN: llc < %s -asm-verbose=false -verify-machineinstrs -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers -mattr=+simd128 | FileCheck %s

; Test that a splat shuffle of an fp-to-int bitcasted vector correctly
; optimizes and lowers to a single splat instruction. Without a custom
; DAG combine, this ends up doing both a splat and a shuffle.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: f32x4_splat:
; CHECK-NEXT: .functype f32x4_splat (f32) -> (v128){{$}}
; CHECK-NEXT: f32x4.splat $push[[R:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @f32x4_splat(float %x) {
  %vecinit = insertelement <4 x float> undef, float %x, i32 0
  %a = bitcast <4 x float> %vecinit to <4 x i32>
  %b = shufflevector <4 x i32> %a, <4 x i32> undef, <4 x i32> zeroinitializer
  ret <4 x i32> %b
}

; CHECK-LABEL: not_a_vec:
; CHECK-NEXT: .functype not_a_vec (i64, i64) -> (v128){{$}}
; CHECK-NEXT: i64x2.splat $push[[L1:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: v8x16.shuffle $push[[R:[0-9]+]]=, $pop[[L1]], $2, 0, 1, 2, 3
; CHECK-NEXT: return $pop[[R]]
define <4 x i32> @not_a_vec(i128 %x) {
  %a = bitcast i128 %x to <4 x i32>
  %b = shufflevector <4 x i32> %a, <4 x i32> undef, <4 x i32> zeroinitializer
  ret <4 x i32> %b
}
