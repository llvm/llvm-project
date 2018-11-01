; RUN: llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers -wasm-enable-unimplemented-simd -mattr=+simd128,+sign-ext | FileCheck %s

; Test that lanewise vector selects lower correctly to bitselects

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: vselect_v16i8:
; CHECK-NEXT: .param v128, v128, v128{{$}}
; CHECK-NEXT: .result v128{{$}}
; CHECK-NEXT: i32.const $push[[L0:[0-9]+]]=, 7{{$}}
; CHECK-NEXT: i8x16.shl $push[[L1:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; CHECK-NEXT: i32.const $push[[L2:[0-9]+]]=, 7{{$}}
; CHECK-NEXT: i8x16.shr_s $push[[L3:[0-9]+]]=, $pop[[L1]], $pop[[L2]]{{$}}
; CHECK-NEXT: v128.bitselect $push[[R:[0-9]+]]=, $1, $2, $pop[[L3]]{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <16 x i8> @vselect_v16i8(<16 x i1> %c, <16 x i8> %x, <16 x i8> %y) {
  %res = select <16 x i1> %c, <16 x i8> %x, <16 x i8> %y
  ret <16 x i8> %res
}

; CHECK-LABEL: vselect_v8i16:
; CHECK-NEXT: .param v128, v128, v128{{$}}
; CHECK-NEXT: .result v128{{$}}
; CHECK-NEXT: i32.const $push[[L0:[0-9]+]]=, 15{{$}}
; CHECK-NEXT: i16x8.shl $push[[L1:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; CHECK-NEXT: i32.const $push[[L2:[0-9]+]]=, 15{{$}}
; CHECK-NEXT: i16x8.shr_s $push[[L3:[0-9]+]]=, $pop[[L1]], $pop[[L2]]{{$}}
; CHECK-NEXT: v128.bitselect $push[[R:[0-9]+]]=, $1, $2, $pop[[L3]]{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @vselect_v8i16(<8 x i1> %c, <8 x i16> %x, <8 x i16> %y) {
  %res = select <8 x i1> %c, <8 x i16> %x, <8 x i16> %y
  ret <8 x i16> %res
}

; CHECK-LABEL: vselect_v4i32:
; CHECK-NEXT: .param v128, v128, v128{{$}}
; CHECK-NEXT: .result v128{{$}}
; CHECK-NEXT: i32.const $push[[L0:[0-9]+]]=, 31{{$}}
; CHECK-NEXT: i32x4.shl $push[[L1:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; CHECK-NEXT: i32.const $push[[L2:[0-9]+]]=, 31{{$}}
; CHECK-NEXT: i32x4.shr_s $push[[L3:[0-9]+]]=, $pop[[L1]], $pop[[L2]]{{$}}
; CHECK-NEXT: v128.bitselect $push[[R:[0-9]+]]=, $1, $2, $pop[[L3]]{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @vselect_v4i32(<4 x i1> %c, <4 x i32> %x, <4 x i32> %y) {
  %res = select <4 x i1> %c, <4 x i32> %x, <4 x i32> %y
  ret <4 x i32> %res
}

; CHECK-LABEL: vselect_v2i64:
; CHECK-NEXT: .param v128, v128, v128{{$}}
; CHECK-NEXT: .result v128{{$}}
; CHECK-NEXT: i32.const $push[[L0:[0-9]+]]=, 63{{$}}
; CHECK-NEXT: i64x2.shl $push[[L1:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; CHECK-NEXT: i32.const $push[[L2:[0-9]+]]=, 63{{$}}
; CHECK-NEXT: i64x2.shr_s $push[[L3:[0-9]+]]=, $pop[[L1]], $pop[[L2]]{{$}}
; CHECK-NEXT: v128.bitselect $push[[R:[0-9]+]]=, $1, $2, $pop[[L3]]{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @vselect_v2i64(<2 x i1> %c, <2 x i64> %x, <2 x i64> %y) {
  %res = select <2 x i1> %c, <2 x i64> %x, <2 x i64> %y
  ret <2 x i64> %res
}

; CHECK-LABEL: vselect_v4f32:
; CHECK-NEXT: .param v128, v128, v128{{$}}
; CHECK-NEXT: .result v128{{$}}
; CHECK-NEXT: i32.const $push[[L0:[0-9]+]]=, 31{{$}}
; CHECK-NEXT: i32x4.shl $push[[L1:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; CHECK-NEXT: i32.const $push[[L2:[0-9]+]]=, 31{{$}}
; CHECK-NEXT: i32x4.shr_s $push[[L3:[0-9]+]]=, $pop[[L1]], $pop[[L2]]{{$}}
; CHECK-NEXT: v128.bitselect $push[[R:[0-9]+]]=, $1, $2, $pop[[L3]]{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <4 x float> @vselect_v4f32(<4 x i1> %c, <4 x float> %x, <4 x float> %y) {
  %res = select <4 x i1> %c, <4 x float> %x, <4 x float> %y
  ret <4 x float> %res
}

; CHECK-LABEL: vselect_v2f64:
; CHECK-NEXT: .param v128, v128, v128{{$}}
; CHECK-NEXT: .result v128{{$}}
; CHECK-NEXT: i32.const $push[[L0:[0-9]+]]=, 63{{$}}
; CHECK-NEXT: i64x2.shl $push[[L1:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; CHECK-NEXT: i32.const $push[[L2:[0-9]+]]=, 63{{$}}
; CHECK-NEXT: i64x2.shr_s $push[[L3:[0-9]+]]=, $pop[[L1]], $pop[[L2]]{{$}}
; CHECK-NEXT: v128.bitselect $push[[R:[0-9]+]]=, $1, $2, $pop[[L3]]{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <2 x double> @vselect_v2f64(<2 x i1> %c, <2 x double> %x, <2 x double> %y) {
  %res = select <2 x i1> %c, <2 x double> %x, <2 x double> %y
  ret <2 x double> %res
}
