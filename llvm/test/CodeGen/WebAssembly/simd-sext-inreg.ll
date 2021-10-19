; RUN: llc < %s -asm-verbose=false -verify-machineinstrs -wasm-keep-registers -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -mattr=+simd128 | FileCheck %s --check-prefixes CHECK,SIMD128
; RUN: llc < %s -asm-verbose=false -verify-machineinstrs -wasm-keep-registers -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals | FileCheck %s --check-prefixes CHECK,NO-SIMD128

; Test that vector sign extensions lower to shifts

target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: sext_v16i8:
; NO-SIMD128-NOT: i8x16
; SIMD128-NEXT: .functype sext_v16i8 (v128) -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[T0:[0-9]+]]=, 7{{$}}
; SIMD128-NEXT: i8x16.shl $push[[T1:[0-9]+]]=, $0, $pop[[T0]]{{$}}
; SIMD128-NEXT: i32.const $push[[T2:[0-9]+]]=, 7{{$}}
; SIMD128-NEXT: i8x16.shr_s $push[[R:[0-9]+]]=, $pop[[T1]], $pop[[T2]]{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <16 x i8> @sext_v16i8(<16 x i1> %x) {
  %res = sext <16 x i1> %x to <16 x i8>
  ret <16 x i8> %res
}

; CHECK-LABEL: sext_v8i16:
; NO-SIMD128-NOT: i16x8
; SIMD128-NEXT: .functype sext_v8i16 (v128) -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[T0:[0-9]+]]=, 15{{$}}
; SIMD128-NEXT: i16x8.shl $push[[T1:[0-9]+]]=, $0, $pop[[T0]]{{$}}
; SIMD128-NEXT: i32.const $push[[T2:[0-9]+]]=, 15{{$}}
; SIMD128-NEXT: i16x8.shr_s $push[[R:[0-9]+]]=, $pop[[T1]], $pop[[T2]]{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @sext_v8i16(<8 x i1> %x) {
  %res = sext <8 x i1> %x to <8 x i16>
  ret <8 x i16> %res
}

; CHECK-LABEL: sext_v4i32:
; NO-SIMD128-NOT: i32x4
; SIMD128-NEXT: .functype sext_v4i32 (v128) -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[T0:[0-9]+]]=, 31{{$}}
; SIMD128-NEXT: i32x4.shl $push[[T1:[0-9]+]]=, $0, $pop[[T0]]{{$}}
; SIMD128-NEXT: i32.const $push[[T2:[0-9]+]]=, 31{{$}}
; SIMD128-NEXT: i32x4.shr_s $push[[R:[0-9]+]]=, $pop[[T1]], $pop[[T2]]{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @sext_v4i32(<4 x i1> %x) {
  %res = sext <4 x i1> %x to <4 x i32>
  ret <4 x i32> %res
}

; CHECK-LABEL: sext_v2i64:
; NO-SIMD128-NOT: i64x2
; SIMD128-NEXT: .functype sext_v2i64 (v128) -> (v128){{$}}
; SIMD128-NEXT: i32.const $push[[T0:[0-9]+]]=, 63{{$}}
; SIMD128-NEXT: i64x2.shl $push[[T1:[0-9]+]]=, $0, $pop[[T0]]{{$}}
; SIMD128-NEXT: i32.const $push[[T2:[0-9]+]]=, 63{{$}}
; SIMD128-NEXT: i64x2.shr_s $push[[R:[0-9]+]]=, $pop[[T1]], $pop[[T2]]{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @sext_v2i64(<2 x i1> %x) {
  %res = sext <2 x i1> %x to <2 x i64>
  ret <2 x i64> %res
}

; CHECK-LABEL: sext_inreg_i8_to_i16:
; SIMD128-NEXT: .functype sext_inreg_i8_to_i16 (v128) -> (i32){{$}}
; SIMD128-NEXT: i8x16.extract_lane_s $push[[R:[0-9]+]]=, $0, 2{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define i16 @sext_inreg_i8_to_i16(<8 x i16> %x) {
  %lane = extractelement <8 x i16> %x, i32 1
  %a = shl i16 %lane, 8
  %res = ashr i16 %a, 8
  ret i16 %res
}

; CHECK-LABEL: sext_inreg_i8_to_i32:
; SIMD128-NEXT: .functype sext_inreg_i8_to_i32 (v128) -> (i32){{$}}
; SIMD128-NEXT: i8x16.extract_lane_s $push[[R:[0-9]+]]=, $0, 4{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define i32 @sext_inreg_i8_to_i32(<4 x i32> %x) {
  %lane = extractelement <4 x i32> %x, i32 1
  %a = shl i32 %lane, 24
  %res = ashr i32 %a, 24
  ret i32 %res
}

; CHECK-LABEL: sext_inreg_i16_to_i32:
; SIMD128-NEXT: .functype sext_inreg_i16_to_i32 (v128) -> (i32){{$}}
; SIMD128-NEXT: i16x8.extract_lane_s $push[[R:[0-9]+]]=, $0, 2{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define i32 @sext_inreg_i16_to_i32(<4 x i32> %x) {
  %lane = extractelement <4 x i32> %x, i32 1
  %a = shl i32 %lane, 16
  %res = ashr i32 %a, 16
  ret i32 %res
}

; CHECK-LABEL: sext_inreg_i8_to_i64:
; SIMD128-NEXT: .functype sext_inreg_i8_to_i64 (v128) -> (i64){{$}}
; SIMD128-NEXT: i64x2.extract_lane $push[[T0:[0-9]+]]=, $0, 1{{$}}
; SIMD128-NEXT: i64.const $push[[T1:[0-9]+]]=, 56{{$}}
; SIMD128-NEXT: i64.shl $push[[T2:[0-9]+]]=, $pop[[T0]], $pop[[T1]]{{$}}
; SIMD128-NEXT: i64.const $push[[T3:[0-9]+]]=, 56{{$}}
; SIMD128-NEXT: i64.shr_s $push[[R:[0-9]+]]=, $pop[[T2]], $pop[[T3]]{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define i64 @sext_inreg_i8_to_i64(<2 x i64> %x) {
  %lane = extractelement <2 x i64> %x, i32 1
  %a = shl i64 %lane, 56
  %res = ashr i64 %a, 56
  ret i64 %res
}

; CHECK-LABEL: sext_inreg_i16_to_i64:
; SIMD128-NEXT: .functype sext_inreg_i16_to_i64 (v128) -> (i64){{$}}
; SIMD128-NEXT: i64x2.extract_lane $push[[T0:[0-9]+]]=, $0, 1{{$}}
; SIMD128-NEXT: i64.const $push[[T1:[0-9]+]]=, 48{{$}}
; SIMD128-NEXT: i64.shl $push[[T2:[0-9]+]]=, $pop[[T0]], $pop[[T1]]{{$}}
; SIMD128-NEXT: i64.const $push[[T3:[0-9]+]]=, 48{{$}}
; SIMD128-NEXT: i64.shr_s $push[[R:[0-9]+]]=, $pop[[T2]], $pop[[T3]]{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define i64 @sext_inreg_i16_to_i64(<2 x i64> %x) {
  %lane = extractelement <2 x i64> %x, i32 1
  %a = shl i64 %lane, 48
  %res = ashr i64 %a, 48
  ret i64 %res
}

; CHECK-LABEL: sext_inreg_i32_to_i64:
; NO-SIMD128-NOT: i64x2
; SIMD128-NEXT: .functype sext_inreg_i32_to_i64 (v128) -> (i64){{$}}
; SIMD128-NEXT: i64x2.extract_lane $push[[T0:[0-9]+]]=, $0, 1{{$}}
; SIMD128-NEXT: i64.const $push[[T1:[0-9]+]]=, 32{{$}}
; SIMD128-NEXT: i64.shl $push[[T2:[0-9]+]]=, $pop[[T0]], $pop[[T1]]{{$}}
; SIMD128-NEXT: i64.const $push[[T3:[0-9]+]]=, 32{{$}}
; SIMD128-NEXT: i64.shr_s $push[[R:[0-9]+]]=, $pop[[T2]], $pop[[T3]]{{$}}
; SIMD128-NEXT: return $pop[[R]]{{$}}
define i64 @sext_inreg_i32_to_i64(<2 x i64> %x) {
  %lane = extractelement <2 x i64> %x, i32 1
  %a = shl i64 %lane, 32
  %res = ashr i64 %a, 32
  ret i64 %res
}
