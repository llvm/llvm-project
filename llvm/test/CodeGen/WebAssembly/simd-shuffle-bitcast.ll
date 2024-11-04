; RUN: llc < %s -asm-verbose=false -verify-machineinstrs -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers -mattr=+simd128 | FileCheck %s

; Test that a splat shuffle of an fp-to-int bitcasted vector correctly
; optimizes and lowers to a single splat instruction.

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

; CHECK-LABEL: i2x2_splat:
; CHECK-NEXT: .functype i2x2_splat (i32) -> (v128){{$}}
define <2 x i2> @i2x2_splat(i1 %x) {
  %vecinit = insertelement <4 x i1> undef, i1 %x, i32 0
  %a = bitcast <4 x i1> %vecinit to <2 x i2>
  %b = shufflevector <2 x i2> %a, <2 x i2> undef, <2 x i32> zeroinitializer
  ret <2 x i2> %b
}

; CHECK-LABEL: not_a_vec:
; CHECK-NEXT: .functype not_a_vec (i64, i64) -> (v128){{$}}
; CHECK-NEXT: i32.wrap_i64    $push[[L:[0-9]+]]=, $0
; CHECK-NEXT: i32x4.splat     $push[[R:[0-9]+]]=, $pop[[L]]
; CHECK-NEXT: return $pop[[R]]
define <4 x i32> @not_a_vec(i128 %x) {
  %a = bitcast i128 %x to <4 x i32>
  %b = shufflevector <4 x i32> %a, <4 x i32> undef, <4 x i32> zeroinitializer
  ret <4 x i32> %b
}
