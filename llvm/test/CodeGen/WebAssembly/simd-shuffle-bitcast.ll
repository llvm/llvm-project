; RUN: llc < %s -asm-verbose=false -verify-machineinstrs -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers -mattr=+simd128 | FileCheck %s

; Test that a splat shuffle of an fp-to-int bitcasted vector correctly
; optimizes and lowers to a single splat instruction. Without a custom
; DAG combine, this ends up doing both a splat and a shuffle.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unkown"

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
