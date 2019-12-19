; RUN: llc < %s -asm-verbose=false -verify-machineinstrs -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers -mattr=+unimplemented-simd128 | FileCheck %s --check-prefixes CHECK,SIMD128
; RUN: llc < %s -asm-verbose=false -verify-machineinstrs -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers -mattr=+unimplemented-simd128 -fast-isel | FileCheck %s --check-prefixes CHECK,SIMD128

; Test that SIMD128 intrinsics lower as expected. These intrinsics are
; only expected to lower successfully if the simd128 attribute is
; enabled and legal types are used.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; ==============================================================================
; 16 x i8
; ==============================================================================
; CHECK-LABEL: swizzle_v16i8:
; SIMD128-NEXT: .functype swizzle_v16i8 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: v8x16.swizzle $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
declare <16 x i8> @llvm.wasm.swizzle(<16 x i8>, <16 x i8>)
define <16 x i8> @swizzle_v16i8(<16 x i8> %x, <16 x i8> %y) {
  %a = call <16 x i8> @llvm.wasm.swizzle(<16 x i8> %x, <16 x i8> %y)
  ret <16 x i8> %a
}

; CHECK-LABEL: add_sat_s_v16i8:
; SIMD128-NEXT: .functype add_sat_s_v16i8 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i8x16.add_saturate_s $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
declare <16 x i8> @llvm.sadd.sat.v16i8(<16 x i8>, <16 x i8>)
define <16 x i8> @add_sat_s_v16i8(<16 x i8> %x, <16 x i8> %y) {
  %a = call <16 x i8> @llvm.sadd.sat.v16i8(<16 x i8> %x, <16 x i8> %y)
  ret <16 x i8> %a
}

; CHECK-LABEL: add_sat_u_v16i8:
; SIMD128-NEXT: .functype add_sat_u_v16i8 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i8x16.add_saturate_u $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
declare <16 x i8> @llvm.uadd.sat.v16i8(<16 x i8>, <16 x i8>)
define <16 x i8> @add_sat_u_v16i8(<16 x i8> %x, <16 x i8> %y) {
  %a = call <16 x i8> @llvm.uadd.sat.v16i8(<16 x i8> %x, <16 x i8> %y)
  ret <16 x i8> %a
}

; CHECK-LABEL: sub_sat_s_v16i8:
; SIMD128-NEXT: .functype sub_sat_s_v16i8 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i8x16.sub_saturate_s $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
declare <16 x i8> @llvm.wasm.sub.saturate.signed.v16i8(<16 x i8>, <16 x i8>)
define <16 x i8> @sub_sat_s_v16i8(<16 x i8> %x, <16 x i8> %y) {
  %a = call <16 x i8> @llvm.wasm.sub.saturate.signed.v16i8(
    <16 x i8> %x, <16 x i8> %y
  )
  ret <16 x i8> %a
}

; CHECK-LABEL: sub_sat_u_v16i8:
; SIMD128-NEXT: .functype sub_sat_u_v16i8 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i8x16.sub_saturate_u $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
declare <16 x i8> @llvm.wasm.sub.saturate.unsigned.v16i8(<16 x i8>, <16 x i8>)
define <16 x i8> @sub_sat_u_v16i8(<16 x i8> %x, <16 x i8> %y) {
  %a = call <16 x i8> @llvm.wasm.sub.saturate.unsigned.v16i8(
    <16 x i8> %x, <16 x i8> %y
  )
  ret <16 x i8> %a
}

; CHECK-LABEL: avgr_u_v16i8:
; SIMD128-NEXT: .functype avgr_u_v16i8 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i8x16.avgr_u $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
declare <16 x i8> @llvm.wasm.avgr.unsigned.v16i8(<16 x i8>, <16 x i8>)
define <16 x i8> @avgr_u_v16i8(<16 x i8> %x, <16 x i8> %y) {
  %a = call <16 x i8> @llvm.wasm.avgr.unsigned.v16i8(<16 x i8> %x, <16 x i8> %y)
  ret <16 x i8> %a
}

; CHECK-LABEL: any_v16i8:
; SIMD128-NEXT: .functype any_v16i8 (v128) -> (i32){{$}}
; SIMD128-NEXT: i8x16.any_true $push[[R:[0-9]+]]=, $0{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
declare i32 @llvm.wasm.anytrue.v16i8(<16 x i8>)
define i32 @any_v16i8(<16 x i8> %x) {
  %a = call i32 @llvm.wasm.anytrue.v16i8(<16 x i8> %x)
  ret i32 %a
}

; CHECK-LABEL: all_v16i8:
; SIMD128-NEXT: .functype all_v16i8 (v128) -> (i32){{$}}
; SIMD128-NEXT: i8x16.all_true $push[[R:[0-9]+]]=, $0{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
declare i32 @llvm.wasm.alltrue.v16i8(<16 x i8>)
define i32 @all_v16i8(<16 x i8> %x) {
  %a = call i32 @llvm.wasm.alltrue.v16i8(<16 x i8> %x)
  ret i32 %a
}

; CHECK-LABEL: bitselect_v16i8:
; SIMD128-NEXT: .functype bitselect_v16i8 (v128, v128, v128) -> (v128){{$}}
; SIMD128-NEXT: v128.bitselect $push[[R:[0-9]+]]=, $0, $1, $2{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
declare <16 x i8> @llvm.wasm.bitselect.v16i8(<16 x i8>, <16 x i8>, <16 x i8>)
define <16 x i8> @bitselect_v16i8(<16 x i8> %v1, <16 x i8> %v2, <16 x i8> %c) {
  %a = call <16 x i8> @llvm.wasm.bitselect.v16i8(
     <16 x i8> %v1, <16 x i8> %v2, <16 x i8> %c
  )
  ret <16 x i8> %a
}

; CHECK-LABEL: narrow_signed_v16i8:
; SIMD128-NEXT: .functype narrow_signed_v16i8 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i8x16.narrow_i16x8_s $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
declare <16 x i8> @llvm.wasm.narrow.signed.v16i8.v8i16(<8 x i16>, <8 x i16>)
define <16 x i8> @narrow_signed_v16i8(<8 x i16> %low, <8 x i16> %high) {
  %a = call <16 x i8> @llvm.wasm.narrow.signed.v16i8.v8i16(
    <8 x i16> %low, <8 x i16> %high
  )
  ret <16 x i8> %a
}

; CHECK-LABEL: narrow_unsigned_v16i8:
; SIMD128-NEXT: .functype narrow_unsigned_v16i8 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i8x16.narrow_i16x8_u $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
declare <16 x i8> @llvm.wasm.narrow.unsigned.v16i8.v8i16(<8 x i16>, <8 x i16>)
define <16 x i8> @narrow_unsigned_v16i8(<8 x i16> %low, <8 x i16> %high) {
  %a = call <16 x i8> @llvm.wasm.narrow.unsigned.v16i8.v8i16(
    <8 x i16> %low, <8 x i16> %high
  )
  ret <16 x i8> %a
}

; ==============================================================================
; 8 x i16
; ==============================================================================
; CHECK-LABEL: add_sat_s_v8i16:
; SIMD128-NEXT: .functype add_sat_s_v8i16 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i16x8.add_saturate_s $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
declare <8 x i16> @llvm.sadd.sat.v8i16(<8 x i16>, <8 x i16>)
define <8 x i16> @add_sat_s_v8i16(<8 x i16> %x, <8 x i16> %y) {
  %a = call <8 x i16> @llvm.sadd.sat.v8i16(<8 x i16> %x, <8 x i16> %y)
  ret <8 x i16> %a
}

; CHECK-LABEL: add_sat_u_v8i16:
; SIMD128-NEXT: .functype add_sat_u_v8i16 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i16x8.add_saturate_u $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
declare <8 x i16> @llvm.uadd.sat.v8i16(<8 x i16>, <8 x i16>)
define <8 x i16> @add_sat_u_v8i16(<8 x i16> %x, <8 x i16> %y) {
  %a = call <8 x i16> @llvm.uadd.sat.v8i16(<8 x i16> %x, <8 x i16> %y)
  ret <8 x i16> %a
}

; CHECK-LABEL: sub_sat_s_v8i16:
; SIMD128-NEXT: .functype sub_sat_s_v8i16 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i16x8.sub_saturate_s $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
declare <8 x i16> @llvm.wasm.sub.saturate.signed.v8i16(<8 x i16>, <8 x i16>)
define <8 x i16> @sub_sat_s_v8i16(<8 x i16> %x, <8 x i16> %y) {
  %a = call <8 x i16> @llvm.wasm.sub.saturate.signed.v8i16(
    <8 x i16> %x, <8 x i16> %y
  )
  ret <8 x i16> %a
}

; CHECK-LABEL: sub_sat_u_v8i16:
; SIMD128-NEXT: .functype sub_sat_u_v8i16 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i16x8.sub_saturate_u $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
declare <8 x i16> @llvm.wasm.sub.saturate.unsigned.v8i16(<8 x i16>, <8 x i16>)
define <8 x i16> @sub_sat_u_v8i16(<8 x i16> %x, <8 x i16> %y) {
  %a = call <8 x i16> @llvm.wasm.sub.saturate.unsigned.v8i16(
    <8 x i16> %x, <8 x i16> %y
  )
  ret <8 x i16> %a
}

; CHECK-LABEL: avgr_u_v8i16:
; SIMD128-NEXT: .functype avgr_u_v8i16 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i16x8.avgr_u $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
declare <8 x i16> @llvm.wasm.avgr.unsigned.v8i16(<8 x i16>, <8 x i16>)
define <8 x i16> @avgr_u_v8i16(<8 x i16> %x, <8 x i16> %y) {
  %a = call <8 x i16> @llvm.wasm.avgr.unsigned.v8i16(<8 x i16> %x, <8 x i16> %y)
  ret <8 x i16> %a
}

; CHECK-LABEL: any_v8i16:
; SIMD128-NEXT: .functype any_v8i16 (v128) -> (i32){{$}}
; SIMD128-NEXT: i16x8.any_true $push[[R:[0-9]+]]=, $0{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
declare i32 @llvm.wasm.anytrue.v8i16(<8 x i16>)
define i32 @any_v8i16(<8 x i16> %x) {
  %a = call i32 @llvm.wasm.anytrue.v8i16(<8 x i16> %x)
  ret i32 %a
}

; CHECK-LABEL: all_v8i16:
; SIMD128-NEXT: .functype all_v8i16 (v128) -> (i32){{$}}
; SIMD128-NEXT: i16x8.all_true $push[[R:[0-9]+]]=, $0{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
declare i32 @llvm.wasm.alltrue.v8i16(<8 x i16>)
define i32 @all_v8i16(<8 x i16> %x) {
  %a = call i32 @llvm.wasm.alltrue.v8i16(<8 x i16> %x)
  ret i32 %a
}

; CHECK-LABEL: bitselect_v8i16:
; SIMD128-NEXT: .functype bitselect_v8i16 (v128, v128, v128) -> (v128){{$}}
; SIMD128-NEXT: v128.bitselect $push[[R:[0-9]+]]=, $0, $1, $2{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
declare <8 x i16> @llvm.wasm.bitselect.v8i16(<8 x i16>, <8 x i16>, <8 x i16>)
define <8 x i16> @bitselect_v8i16(<8 x i16> %v1, <8 x i16> %v2, <8 x i16> %c) {
  %a = call <8 x i16> @llvm.wasm.bitselect.v8i16(
    <8 x i16> %v1, <8 x i16> %v2, <8 x i16> %c
  )
  ret <8 x i16> %a
}

; CHECK-LABEL: narrow_signed_v8i16:
; SIMD128-NEXT: .functype narrow_signed_v8i16 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i16x8.narrow_i32x4_s $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
declare <8 x i16> @llvm.wasm.narrow.signed.v8i16.v4i32(<4 x i32>, <4 x i32>)
define <8 x i16> @narrow_signed_v8i16(<4 x i32> %low, <4 x i32> %high) {
  %a = call <8 x i16> @llvm.wasm.narrow.signed.v8i16.v4i32(
    <4 x i32> %low, <4 x i32> %high
  )
  ret <8 x i16> %a
}

; CHECK-LABEL: narrow_unsigned_v8i16:
; SIMD128-NEXT: .functype narrow_unsigned_v8i16 (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i16x8.narrow_i32x4_u $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
declare <8 x i16> @llvm.wasm.narrow.unsigned.v8i16.v4i32(<4 x i32>, <4 x i32>)
define <8 x i16> @narrow_unsigned_v8i16(<4 x i32> %low, <4 x i32> %high) {
  %a = call <8 x i16> @llvm.wasm.narrow.unsigned.v8i16.v4i32(
    <4 x i32> %low, <4 x i32> %high
  )
  ret <8 x i16> %a
}

; CHECK-LABEL: widen_low_signed_v8i16:
; SIMD128-NEXT: .functype widen_low_signed_v8i16 (v128) -> (v128){{$}}
; SIMD128-NEXT: i16x8.widen_low_i8x16_s $push[[R:[0-9]+]]=, $0{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
declare <8 x i16> @llvm.wasm.widen.low.signed.v8i16.v16i8(<16 x i8>)
define <8 x i16> @widen_low_signed_v8i16(<16 x i8> %v) {
  %a = call <8 x i16> @llvm.wasm.widen.low.signed.v8i16.v16i8(<16 x i8> %v)
  ret <8 x i16> %a
}

; CHECK-LABEL: widen_high_signed_v8i16:
; SIMD128-NEXT: .functype widen_high_signed_v8i16 (v128) -> (v128){{$}}
; SIMD128-NEXT: i16x8.widen_high_i8x16_s $push[[R:[0-9]+]]=, $0{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
declare <8 x i16> @llvm.wasm.widen.high.signed.v8i16.v16i8(<16 x i8>)
define <8 x i16> @widen_high_signed_v8i16(<16 x i8> %v) {
  %a = call <8 x i16> @llvm.wasm.widen.high.signed.v8i16.v16i8(<16 x i8> %v)
  ret <8 x i16> %a
}

; CHECK-LABEL: widen_low_unsigned_v8i16:
; SIMD128-NEXT: .functype widen_low_unsigned_v8i16 (v128) -> (v128){{$}}
; SIMD128-NEXT: i16x8.widen_low_i8x16_u $push[[R:[0-9]+]]=, $0{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
declare <8 x i16> @llvm.wasm.widen.low.unsigned.v8i16.v16i8(<16 x i8>)
define <8 x i16> @widen_low_unsigned_v8i16(<16 x i8> %v) {
  %a = call <8 x i16> @llvm.wasm.widen.low.unsigned.v8i16.v16i8(<16 x i8> %v)
  ret <8 x i16> %a
}

; CHECK-LABEL: widen_high_unsigned_v8i16:
; SIMD128-NEXT: .functype widen_high_unsigned_v8i16 (v128) -> (v128){{$}}
; SIMD128-NEXT: i16x8.widen_high_i8x16_u $push[[R:[0-9]+]]=, $0{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
declare <8 x i16> @llvm.wasm.widen.high.unsigned.v8i16.v16i8(<16 x i8>)
define <8 x i16> @widen_high_unsigned_v8i16(<16 x i8> %v) {
  %a = call <8 x i16> @llvm.wasm.widen.high.unsigned.v8i16.v16i8(<16 x i8> %v)
  ret <8 x i16> %a
}

; ==============================================================================
; 4 x i32
; ==============================================================================
; CHECK-LABEL: dot:
; SIMD128-NEXT: .functype dot (v128, v128) -> (v128){{$}}
; SIMD128-NEXT: i32x4.dot_i16x8_s $push[[R:[0-9]+]]=, $0, $1{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
declare <4 x i32> @llvm.wasm.dot(<8 x i16>, <8 x i16>)
define <4 x i32> @dot(<8 x i16> %x, <8 x i16> %y) {
  %a = call <4 x i32> @llvm.wasm.dot(<8 x i16> %x, <8 x i16> %y)
  ret <4 x i32> %a
}

; CHECK-LABEL: any_v4i32:
; SIMD128-NEXT: .functype any_v4i32 (v128) -> (i32){{$}}
; SIMD128-NEXT: i32x4.any_true $push[[R:[0-9]+]]=, $0{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
declare i32 @llvm.wasm.anytrue.v4i32(<4 x i32>)
define i32 @any_v4i32(<4 x i32> %x) {
  %a = call i32 @llvm.wasm.anytrue.v4i32(<4 x i32> %x)
  ret i32 %a
}

; CHECK-LABEL: all_v4i32:
; SIMD128-NEXT: .functype all_v4i32 (v128) -> (i32){{$}}
; SIMD128-NEXT: i32x4.all_true $push[[R:[0-9]+]]=, $0{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
declare i32 @llvm.wasm.alltrue.v4i32(<4 x i32>)
define i32 @all_v4i32(<4 x i32> %x) {
  %a = call i32 @llvm.wasm.alltrue.v4i32(<4 x i32> %x)
  ret i32 %a
}

; CHECK-LABEL: bitselect_v4i32:
; SIMD128-NEXT: .functype bitselect_v4i32 (v128, v128, v128) -> (v128){{$}}
; SIMD128-NEXT: v128.bitselect $push[[R:[0-9]+]]=, $0, $1, $2{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
declare <4 x i32> @llvm.wasm.bitselect.v4i32(<4 x i32>, <4 x i32>, <4 x i32>)
define <4 x i32> @bitselect_v4i32(<4 x i32> %v1, <4 x i32> %v2, <4 x i32> %c) {
  %a = call <4 x i32> @llvm.wasm.bitselect.v4i32(
    <4 x i32> %v1, <4 x i32> %v2, <4 x i32> %c
  )
  ret <4 x i32> %a
}

; CHECK-LABEL: trunc_sat_s_v4i32:
; NO-SIMD128-NOT: f32x4
; SIMD128-NEXT: .functype trunc_sat_s_v4i32 (v128) -> (v128){{$}}
; SIMD128-NEXT: i32x4.trunc_sat_f32x4_s $push[[R:[0-9]+]]=, $0
; SIMD128-NEXT: return $pop[[R]]
declare <4 x i32> @llvm.wasm.trunc.saturate.signed.v4i32.v4f32(<4 x float>)
define <4 x i32> @trunc_sat_s_v4i32(<4 x float> %x) {
  %a = call <4 x i32> @llvm.wasm.trunc.saturate.signed.v4i32.v4f32(<4 x float> %x)
  ret <4 x i32> %a
}

; CHECK-LABEL: trunc_sat_u_v4i32:
; NO-SIMD128-NOT: f32x4
; SIMD128-NEXT: .functype trunc_sat_u_v4i32 (v128) -> (v128){{$}}
; SIMD128-NEXT: i32x4.trunc_sat_f32x4_u $push[[R:[0-9]+]]=, $0
; SIMD128-NEXT: return $pop[[R]]
declare <4 x i32> @llvm.wasm.trunc.saturate.unsigned.v4i32.v4f32(<4 x float>)
define <4 x i32> @trunc_sat_u_v4i32(<4 x float> %x) {
  %a = call <4 x i32> @llvm.wasm.trunc.saturate.unsigned.v4i32.v4f32(<4 x float> %x)
  ret <4 x i32> %a
}

; CHECK-LABEL: widen_low_signed_v4i32:
; SIMD128-NEXT: .functype widen_low_signed_v4i32 (v128) -> (v128){{$}}
; SIMD128-NEXT: i32x4.widen_low_i16x8_s $push[[R:[0-9]+]]=, $0{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
declare <4 x i32> @llvm.wasm.widen.low.signed.v4i32.v8i16(<8 x i16>)
define <4 x i32> @widen_low_signed_v4i32(<8 x i16> %v) {
  %a = call <4 x i32> @llvm.wasm.widen.low.signed.v4i32.v8i16(<8 x i16> %v)
  ret <4 x i32> %a
}

; CHECK-LABEL: widen_high_signed_v4i32:
; SIMD128-NEXT: .functype widen_high_signed_v4i32 (v128) -> (v128){{$}}
; SIMD128-NEXT: i32x4.widen_high_i16x8_s $push[[R:[0-9]+]]=, $0{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
declare <4 x i32> @llvm.wasm.widen.high.signed.v4i32.v8i16(<8 x i16>)
define <4 x i32> @widen_high_signed_v4i32(<8 x i16> %v) {
  %a = call <4 x i32> @llvm.wasm.widen.high.signed.v4i32.v8i16(<8 x i16> %v)
  ret <4 x i32> %a
}

; CHECK-LABEL: widen_low_unsigned_v4i32:
; SIMD128-NEXT: .functype widen_low_unsigned_v4i32 (v128) -> (v128){{$}}
; SIMD128-NEXT: i32x4.widen_low_i16x8_u $push[[R:[0-9]+]]=, $0{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
declare <4 x i32> @llvm.wasm.widen.low.unsigned.v4i32.v8i16(<8 x i16>)
define <4 x i32> @widen_low_unsigned_v4i32(<8 x i16> %v) {
  %a = call <4 x i32> @llvm.wasm.widen.low.unsigned.v4i32.v8i16(<8 x i16> %v)
  ret <4 x i32> %a
}

; CHECK-LABEL: widen_high_unsigned_v4i32:
; SIMD128-NEXT: .functype widen_high_unsigned_v4i32 (v128) -> (v128){{$}}
; SIMD128-NEXT: i32x4.widen_high_i16x8_u $push[[R:[0-9]+]]=, $0{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
declare <4 x i32> @llvm.wasm.widen.high.unsigned.v4i32.v8i16(<8 x i16>)
define <4 x i32> @widen_high_unsigned_v4i32(<8 x i16> %v) {
  %a = call <4 x i32> @llvm.wasm.widen.high.unsigned.v4i32.v8i16(<8 x i16> %v)
  ret <4 x i32> %a
}

; ==============================================================================
; 2 x i64
; ==============================================================================
; CHECK-LABEL: any_v2i64:
; SIMD128-NEXT: .functype any_v2i64 (v128) -> (i32){{$}}
; SIMD128-NEXT: i64x2.any_true $push[[R:[0-9]+]]=, $0{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
declare i32 @llvm.wasm.anytrue.v2i64(<2 x i64>)
define i32 @any_v2i64(<2 x i64> %x) {
  %a = call i32 @llvm.wasm.anytrue.v2i64(<2 x i64> %x)
  ret i32 %a
}

; CHECK-LABEL: all_v2i64:
; SIMD128-NEXT: .functype all_v2i64 (v128) -> (i32){{$}}
; SIMD128-NEXT: i64x2.all_true $push[[R:[0-9]+]]=, $0{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
declare i32 @llvm.wasm.alltrue.v2i64(<2 x i64>)
define i32 @all_v2i64(<2 x i64> %x) {
  %a = call i32 @llvm.wasm.alltrue.v2i64(<2 x i64> %x)
  ret i32 %a
}

; CHECK-LABEL: bitselect_v2i64:
; SIMD128-NEXT: .functype bitselect_v2i64 (v128, v128, v128) -> (v128){{$}}
; SIMD128-NEXT: v128.bitselect $push[[R:[0-9]+]]=, $0, $1, $2{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
declare <2 x i64> @llvm.wasm.bitselect.v2i64(<2 x i64>, <2 x i64>, <2 x i64>)
define <2 x i64> @bitselect_v2i64(<2 x i64> %v1, <2 x i64> %v2, <2 x i64> %c) {
  %a = call <2 x i64> @llvm.wasm.bitselect.v2i64(
    <2 x i64> %v1, <2 x i64> %v2, <2 x i64> %c
  )
  ret <2 x i64> %a
}

; CHECK-LABEL: trunc_sat_s_v2i64:
; NO-SIMD128-NOT: f32x4
; SIMD128-NEXT: .functype trunc_sat_s_v2i64 (v128) -> (v128){{$}}
; SIMD128-NEXT: i64x2.trunc_sat_f64x2_s $push[[R:[0-9]+]]=, $0
; SIMD128-NEXT: return $pop[[R]]
declare <2 x i64> @llvm.wasm.trunc.saturate.signed.v2i64.v2f64(<2 x double>)
define <2 x i64> @trunc_sat_s_v2i64(<2 x double> %x) {
  %a = call <2 x i64> @llvm.wasm.trunc.saturate.signed.v2i64.v2f64(<2 x double> %x)
  ret <2 x i64> %a
}

; CHECK-LABEL: trunc_sat_u_v2i64:
; NO-SIMD128-NOT: f32x4
; SIMD128-NEXT: .functype trunc_sat_u_v2i64 (v128) -> (v128){{$}}
; SIMD128-NEXT: i64x2.trunc_sat_f64x2_u $push[[R:[0-9]+]]=, $0
; SIMD128-NEXT: return $pop[[R]]
declare <2 x i64> @llvm.wasm.trunc.saturate.unsigned.v2i64.v2f64(<2 x double>)
define <2 x i64> @trunc_sat_u_v2i64(<2 x double> %x) {
  %a = call <2 x i64> @llvm.wasm.trunc.saturate.unsigned.v2i64.v2f64(<2 x double> %x)
  ret <2 x i64> %a
}

; ==============================================================================
; 4 x f32
; ==============================================================================
; CHECK-LABEL: bitselect_v4f32:
; SIMD128-NEXT: .functype bitselect_v4f32 (v128, v128, v128) -> (v128){{$}}
; SIMD128-NEXT: v128.bitselect $push[[R:[0-9]+]]=, $0, $1, $2{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
declare <4 x float> @llvm.wasm.bitselect.v4f32(<4 x float>, <4 x float>, <4 x float>)
define <4 x float> @bitselect_v4f32(<4 x float> %v1, <4 x float> %v2, <4 x float> %c) {
  %a = call <4 x float> @llvm.wasm.bitselect.v4f32(
    <4 x float> %v1, <4 x float> %v2, <4 x float> %c
  )
  ret <4 x float> %a
}

; CHECK-LABEL: qfma_v4f32:
; SIMD128-NEXT: .functype qfma_v4f32 (v128, v128, v128) -> (v128){{$}}
; SIMD128-NEXT: f32x4.qfma $push[[R:[0-9]+]]=, $0, $1, $2{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
declare <4 x float> @llvm.wasm.qfma.v4f32(<4 x float>, <4 x float>, <4 x float>)
define <4 x float> @qfma_v4f32(<4 x float> %a, <4 x float> %b, <4 x float> %c) {
  %v = call <4 x float> @llvm.wasm.qfma.v4f32(
    <4 x float> %a, <4 x float> %b, <4 x float> %c
  )
  ret <4 x float> %v
}

; CHECK-LABEL: qfms_v4f32:
; SIMD128-NEXT: .functype qfms_v4f32 (v128, v128, v128) -> (v128){{$}}
; SIMD128-NEXT: f32x4.qfms $push[[R:[0-9]+]]=, $0, $1, $2{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
declare <4 x float> @llvm.wasm.qfms.v4f32(<4 x float>, <4 x float>, <4 x float>)
define <4 x float> @qfms_v4f32(<4 x float> %a, <4 x float> %b, <4 x float> %c) {
  %v = call <4 x float> @llvm.wasm.qfms.v4f32(
    <4 x float> %a, <4 x float> %b, <4 x float> %c
  )
  ret <4 x float> %v
}

; ==============================================================================
; 2 x f64
; ==============================================================================
; CHECK-LABEL: bitselect_v2f64:
; SIMD128-NEXT: .functype bitselect_v2f64 (v128, v128, v128) -> (v128){{$}}
; SIMD128-NEXT: v128.bitselect $push[[R:[0-9]+]]=, $0, $1, $2{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
declare <2 x double> @llvm.wasm.bitselect.v2f64(<2 x double>, <2 x double>, <2 x double>)
define <2 x double> @bitselect_v2f64(<2 x double> %v1, <2 x double> %v2, <2 x double> %c) {
  %a = call <2 x double> @llvm.wasm.bitselect.v2f64(
    <2 x double> %v1, <2 x double> %v2, <2 x double> %c
  )
  ret <2 x double> %a
}

; CHECK-LABEL: qfma_v2f64:
; SIMD128-NEXT: .functype qfma_v2f64 (v128, v128, v128) -> (v128){{$}}
; SIMD128-NEXT: f64x2.qfma $push[[R:[0-9]+]]=, $0, $1, $2{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
declare <2 x double> @llvm.wasm.qfma.v2f64(<2 x double>, <2 x double>, <2 x double>)
define <2 x double> @qfma_v2f64(<2 x double> %a, <2 x double> %b, <2 x double> %c) {
  %v = call <2 x double> @llvm.wasm.qfma.v2f64(
    <2 x double> %a, <2 x double> %b, <2 x double> %c
  )
  ret <2 x double> %v
}

; CHECK-LABEL: qfms_v2f64:
; SIMD128-NEXT: .functype qfms_v2f64 (v128, v128, v128) -> (v128){{$}}
; SIMD128-NEXT: f64x2.qfms $push[[R:[0-9]+]]=, $0, $1, $2{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
declare <2 x double> @llvm.wasm.qfms.v2f64(<2 x double>, <2 x double>, <2 x double>)
define <2 x double> @qfms_v2f64(<2 x double> %a, <2 x double> %b, <2 x double> %c) {
  %v = call <2 x double> @llvm.wasm.qfms.v2f64(
    <2 x double> %a, <2 x double> %b, <2 x double> %c
  )
  ret <2 x double> %v
}
