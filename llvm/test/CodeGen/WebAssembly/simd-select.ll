; RUN: llc < %s -asm-verbose=false -verify-machineinstrs -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers -mattr=+unimplemented-simd128 | FileCheck %s

; Test that vector selects of various varieties lower correctly to bitselects.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; ==============================================================================
; 16 x i8
; ==============================================================================
; CHECK-LABEL: vselect_v16i8:
; CHECK-NEXT: .functype vselect_v16i8 (v128, v128, v128) -> (v128){{$}}
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

; CHECK-LABEL: vselect_cmp_v16i8:
; CHECK-NEXT: .functype vselect_cmp_v16i8 (v128, v128, v128, v128) -> (v128){{$}}
; CHECK-NEXT: i8x16.lt_s $push[[L0:[0-9]+]]=, $0, $1{{$}}
; CHECK-NEXT: v128.bitselect $push[[R:[0-9]+]]=, $2, $3, $pop[[L0]]{{$}}
; CHECK-NEXT: return  $pop[[R]]{{$}}
define <16 x i8> @vselect_cmp_v16i8(<16 x i8> %a, <16 x i8> %b,
                                    <16 x i8> %x, <16 x i8> %y) {
  %c = icmp slt <16 x i8> %a, %b
  %res = select <16 x i1> %c, <16 x i8> %x, <16 x i8> %y
  ret <16 x i8> %res
}

; CHECK-LABEL: select_v16i8:
; CHECK-NEXT: .functype select_v16i8 (i32, v128, v128) -> (v128){{$}}
; CHECK-NEXT: i32.const $push[[L0:[0-9]+]]=, -1{{$}}
; CHECK-NEXT: i32.const $push[[L1:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: i32.select $push[[L2:[0-9]+]]=, $pop[[L0]], $pop[[L1]], $0{{$}}
; CHECK-NEXT: i8x16.splat $push[[L3:[0-9]+]]=, $pop[[L2]]{{$}}
; CHECK-NEXT: v128.bitselect $push[[R:[0-9]+]]=, $1, $2, $pop[[L3]]{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <16 x i8> @select_v16i8(i1 zeroext %c, <16 x i8> %x, <16 x i8> %y) {
  %res = select i1 %c, <16 x i8> %x, <16 x i8> %y
  ret <16 x i8> %res
}

; CHECK-LABEL: select_cmp_v16i8:
; CHECK-NEXT: .functype select_cmp_v16i8 (i32, v128, v128) -> (v128){{$}}
; CHECK-NEXT: i32.const $push[[L0:[0-9]+]]=, 31
; CHECK-NEXT: i32.shr_s $push[[L1:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; CHECK-NEXT: i8x16.splat $push[[L2:[0-9]+]]=, $pop[[L1]]{{$}}
; CHECK-NEXT: v128.bitselect $push[[R:[0-9]+]]=, $1, $2, $pop[[L2]]{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <16 x i8> @select_cmp_v16i8(i32 %i, <16 x i8> %x, <16 x i8> %y) {
  %c = icmp slt i32 %i, 0
  %res = select i1 %c, <16 x i8> %x, <16 x i8> %y
  ret <16 x i8> %res
}

; CHECK-LABEL: select_ne_v16i8:
; CHECK-NEXT: .functype select_ne_v16i8 (i32, v128, v128) -> (v128){{$}}
; CHECK-NEXT: i32.const $push[[L0:[0-9]+]]=, -1{{$}}
; CHECK-NEXT: i32.const $push[[L1:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: i32.select $push[[L2:[0-9]+]]=, $pop[[L0]], $pop[[L1]], $0{{$}}
; CHECK-NEXT: i8x16.splat $push[[L3:[0-9]+]]=, $pop[[L2]]{{$}}
; CHECK-NEXT: v128.bitselect $push[[R:[0-9]+]]=, $1, $2, $pop[[L3]]{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <16 x i8> @select_ne_v16i8(i32 %i, <16 x i8> %x, <16 x i8> %y) {
  %c = icmp ne i32 %i, 0
  %res = select i1 %c, <16 x i8> %x, <16 x i8> %y
  ret <16 x i8> %res
}

; CHECK-LABEL: select_eq_v16i8:
; CHECK-NEXT: .functype select_eq_v16i8 (i32, v128, v128) -> (v128){{$}}
; CHECK-NEXT: i32.const $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: i32.const $push[[L1:[0-9]+]]=, -1{{$}}
; CHECK-NEXT: i32.select $push[[L2:[0-9]+]]=, $pop[[L0]], $pop[[L1]], $0{{$}}
; CHECK-NEXT: i8x16.splat $push[[L3:[0-9]+]]=, $pop[[L2]]{{$}}
; CHECK-NEXT: v128.bitselect $push[[R:[0-9]+]]=, $1, $2, $pop[[L3]]{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <16 x i8> @select_eq_v16i8(i32 %i, <16 x i8> %x, <16 x i8> %y) {
  %c = icmp eq i32 %i, 0
  %res = select i1 %c, <16 x i8> %x, <16 x i8> %y
  ret <16 x i8> %res
}

; ==============================================================================
; 8 x i16
; ==============================================================================
; CHECK-LABEL: vselect_v8i16:
; CHECK-NEXT: .functype vselect_v8i16 (v128, v128, v128) -> (v128){{$}}
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

; CHECK-LABEL: vselect_cmp_v8i16:
; CHECK-NEXT: .functype vselect_cmp_v8i16 (v128, v128, v128, v128) -> (v128){{$}}
; CHECK-NEXT: i16x8.lt_s $push[[L0:[0-9]+]]=, $0, $1{{$}}
; CHECK-NEXT: v128.bitselect $push[[R:[0-9]+]]=, $2, $3, $pop[[L0]]{{$}}
; CHECK-NEXT: return  $pop[[R]]{{$}}
define <8 x i16> @vselect_cmp_v8i16(<8 x i16> %a, <8 x i16> %b,
                                           <8 x i16> %x, <8 x i16> %y) {
  %c = icmp slt <8 x i16> %a, %b
  %res = select <8 x i1> %c, <8 x i16> %x, <8 x i16> %y
  ret <8 x i16> %res
}

; CHECK-LABEL: select_v8i16:
; CHECK-NEXT: .functype select_v8i16 (i32, v128, v128) -> (v128){{$}}
; CHECK-NEXT: i32.const $push[[L0:[0-9]+]]=, -1{{$}}
; CHECK-NEXT: i32.const $push[[L1:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: i32.select $push[[L2:[0-9]+]]=, $pop[[L0]], $pop[[L1]], $0{{$}}
; CHECK-NEXT: i16x8.splat $push[[L3:[0-9]+]]=, $pop[[L2]]{{$}}
; CHECK-NEXT: v128.bitselect $push[[R:[0-9]+]]=, $1, $2, $pop[[L3]]{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @select_v8i16(i1 zeroext %c, <8 x i16> %x, <8 x i16> %y) {
  %res = select i1 %c, <8 x i16> %x, <8 x i16> %y
  ret <8 x i16> %res
}

; CHECK-LABEL: select_cmp_v8i16:
; CHECK-NEXT: .functype select_cmp_v8i16 (i32, v128, v128) -> (v128){{$}}
; CHECK-NEXT: i32.const $push[[L0:[0-9]+]]=, 31{{$}}
; CHECK-NEXT: i32.shr_s $push[[L1:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; CHECK-NEXT: i16x8.splat $push[[L2:[0-9]+]]=, $pop[[L1]]{{$}}
; CHECK-NEXT: v128.bitselect $push[[R:[0-9]+]]=, $1, $2, $pop[[L2]]{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @select_cmp_v8i16(i32 %i, <8 x i16> %x, <8 x i16> %y) {
  %c = icmp slt i32 %i, 0
  %res = select i1 %c, <8 x i16> %x, <8 x i16> %y
  ret <8 x i16> %res
}

; CHECK-LABEL: select_ne_v8i16:
; CHECK-NEXT: .functype select_ne_v8i16 (i32, v128, v128) -> (v128){{$}}
; CHECK-NEXT: i32.const $push[[L0:[0-9]+]]=, -1{{$}}
; CHECK-NEXT: i32.const $push[[L1:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: i32.select $push[[L2:[0-9]+]]=, $pop[[L0]], $pop[[L1]], $0{{$}}
; CHECK-NEXT: i16x8.splat $push[[L3:[0-9]+]]=, $pop[[L2]]{{$}}
; CHECK-NEXT: v128.bitselect $push[[R:[0-9]+]]=, $1, $2, $pop[[L3]]{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @select_ne_v8i16(i32 %i, <8 x i16> %x, <8 x i16> %y) {
  %c = icmp ne i32 %i, 0
  %res = select i1 %c, <8 x i16> %x, <8 x i16> %y
  ret <8 x i16> %res
}

; CHECK-LABEL: select_eq_v8i16:
; CHECK-NEXT: .functype select_eq_v8i16 (i32, v128, v128) -> (v128){{$}}
; CHECK-NEXT: i32.const $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: i32.const $push[[L1:[0-9]+]]=, -1{{$}}
; CHECK-NEXT: i32.select $push[[L2:[0-9]+]]=, $pop[[L0]], $pop[[L1]], $0{{$}}
; CHECK-NEXT: i16x8.splat $push[[L3:[0-9]+]]=, $pop[[L2]]{{$}}
; CHECK-NEXT: v128.bitselect $push[[R:[0-9]+]]=, $1, $2, $pop[[L3]]{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <8 x i16> @select_eq_v8i16(i32 %i, <8 x i16> %x, <8 x i16> %y) {
  %c = icmp eq i32 %i, 0
  %res = select i1 %c, <8 x i16> %x, <8 x i16> %y
  ret <8 x i16> %res
}

; ==============================================================================
; 4 x i32
; ==============================================================================
; CHECK-LABEL: vselect_v4i32:
; CHECK-NEXT: .functype vselect_v4i32 (v128, v128, v128) -> (v128){{$}}
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

; CHECK-LABEL: vselect_cmp_v4i32:
; CHECK-NEXT: .functype vselect_cmp_v4i32 (v128, v128, v128, v128) -> (v128){{$}}
; CHECK-NEXT: i32x4.lt_s $push[[L0:[0-9]+]]=, $0, $1{{$}}
; CHECK-NEXT: v128.bitselect $push[[R:[0-9]+]]=, $2, $3, $pop[[L0]]{{$}}
; CHECK-NEXT: return  $pop[[R]]{{$}}
define <4 x i32> @vselect_cmp_v4i32(<4 x i32> %a, <4 x i32> %b,
                                    <4 x i32> %x, <4 x i32> %y) {
  %c = icmp slt <4 x i32> %a, %b
  %res = select <4 x i1> %c, <4 x i32> %x, <4 x i32> %y
  ret <4 x i32> %res
}

; CHECK-LABEL: select_v4i32:
; CHECK-NEXT: .functype select_v4i32 (i32, v128, v128) -> (v128){{$}}
; CHECK-NEXT: i32.const $push[[L0:[0-9]+]]=, -1{{$}}
; CHECK-NEXT: i32.const $push[[L1:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: i32.select $push[[L2:[0-9]+]]=, $pop[[L0]], $pop[[L1]], $0{{$}}
; CHECK-NEXT: i32x4.splat $push[[L3:[0-9]+]]=, $pop[[L2]]{{$}}
; CHECK-NEXT: v128.bitselect $push[[R:[0-9]+]]=, $1, $2, $pop[[L3]]{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @select_v4i32(i1 zeroext %c, <4 x i32> %x, <4 x i32> %y) {
  %res = select i1 %c, <4 x i32> %x, <4 x i32> %y
  ret <4 x i32> %res
}

; CHECK-LABEL: select_cmp_v4i32:
; CHECK-NEXT: .functype select_cmp_v4i32 (i32, v128, v128) -> (v128){{$}}
; CHECK-NEXT: i32.const $push[[L0:[0-9]+]]=, 31{{$}}
; CHECK-NEXT: i32.shr_s $push[[L1:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; CHECK-NEXT: i32x4.splat $push[[L2:[0-9]+]]=, $pop[[L1]]{{$}}
; CHECK-NEXT: v128.bitselect $push[[R:[0-9]+]]=, $1, $2, $pop[[L2]]{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @select_cmp_v4i32(i32 %i, <4 x i32> %x, <4 x i32> %y) {
  %c = icmp slt i32 %i, 0
  %res = select i1 %c, <4 x i32> %x, <4 x i32> %y
  ret <4 x i32> %res
}

; CHECK-LABEL: select_ne_v4i32:
; CHECK-NEXT: .functype select_ne_v4i32 (i32, v128, v128) -> (v128){{$}}
; CHECK-NEXT: i32.const $push[[L0:[0-9]+]]=, -1{{$}}
; CHECK-NEXT: i32.const $push[[L1:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: i32.select $push[[L2:[0-9]+]]=, $pop[[L0]], $pop[[L1]], $0{{$}}
; CHECK-NEXT: i32x4.splat $push[[L3:[0-9]+]]=, $pop[[L2]]{{$}}
; CHECK-NEXT: v128.bitselect $push[[R:[0-9]+]]=, $1, $2, $pop[[L3]]{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @select_ne_v4i32(i32 %i, <4 x i32> %x, <4 x i32> %y) {
  %c = icmp ne i32 %i, 0
  %res = select i1 %c, <4 x i32> %x, <4 x i32> %y
  ret <4 x i32> %res
}

; CHECK-LABEL: select_eq_v4i32:
; CHECK-NEXT: .functype select_eq_v4i32 (i32, v128, v128) -> (v128){{$}}
; CHECK-NEXT: i32.const $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: i32.const $push[[L1:[0-9]+]]=, -1{{$}}
; CHECK-NEXT: i32.select $push[[L2:[0-9]+]]=, $pop[[L0]], $pop[[L1]], $0{{$}}
; CHECK-NEXT: i32x4.splat $push[[L3:[0-9]+]]=, $pop[[L2]]{{$}}
; CHECK-NEXT: v128.bitselect $push[[R:[0-9]+]]=, $1, $2, $pop[[L3]]{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <4 x i32> @select_eq_v4i32(i32 %i, <4 x i32> %x, <4 x i32> %y) {
  %c = icmp eq i32 %i, 0
  %res = select i1 %c, <4 x i32> %x, <4 x i32> %y
  ret <4 x i32> %res
}

; ==============================================================================
; 2 x i64
; ==============================================================================
; CHECK-LABEL: vselect_v2i64:
; CHECK-NEXT: .functype vselect_v2i64 (v128, v128, v128) -> (v128){{$}}
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

; CHECK-LABEL: vselect_cmp_v2i64:
; CHECK-NEXT: .functype vselect_cmp_v2i64 (v128, v128, v128, v128) -> (v128){{$}}
; CHECK-NEXT: i64.const $push[[L0:[0-9]+]]=, -1{{$}}
; CHECK-NEXT: i64.const $push[[L1:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: i64x2.extract_lane $push[[L2:[0-9]+]]=, $0, 0{{$}}
; CHECK-NEXT: i64x2.extract_lane $push[[L3:[0-9]+]]=, $1, 0{{$}}
; CHECK-NEXT: i64.lt_s $push[[L4:[0-9]+]]=, $pop[[L2]], $pop[[L3]]{{$}}
; CHECK-NEXT: i64.select $push[[L5:[0-9]+]]=, $pop[[L0]], $pop[[L1]], $pop[[L4]]{{$}}
; CHECK-NEXT: i64x2.splat $push[[L6:[0-9]+]]=, $pop[[L5]]{{$}}
; CHECK-NEXT: i64.const $push[[L7:[0-9]+]]=, -1{{$}}
; CHECK-NEXT: i64.const $push[[L8:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: i64x2.extract_lane $push[[L9:[0-9]+]]=, $0, 1{{$}}
; CHECK-NEXT: i64x2.extract_lane $push[[L10:[0-9]+]]=, $1, 1{{$}}
; CHECK-NEXT: i64.lt_s $push[[L11:[0-9]+]]=, $pop[[L9]], $pop[[L10]]{{$}}
; CHECK-NEXT: i64.select $push[[L12:[0-9]+]]=, $pop[[L7]], $pop[[L8]], $pop[[L11]]{{$}}
; CHECK-NEXT: i64x2.replace_lane $push[[L13:[0-9]+]]=, $pop[[L6]], 1, $pop[[L12]]{{$}}
; CHECK-NEXT: v128.bitselect $push[[R:[0-9]+]]=, $2, $3, $pop[[L13]]{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @vselect_cmp_v2i64(<2 x i64> %a, <2 x i64> %b,
                                    <2 x i64> %x, <2 x i64> %y) {
  %c = icmp slt <2 x i64> %a, %b
  %res = select <2 x i1> %c, <2 x i64> %x, <2 x i64> %y
  ret <2 x i64> %res
}

; CHECK-LABEL: select_v2i64:
; CHECK-NEXT: .functype select_v2i64 (i32, v128, v128) -> (v128){{$}}
; CHECK-NEXT: i64.const $push[[L0:[0-9]+]]=, -1{{$}}
; CHECK-NEXT: i64.const $push[[L1:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: i64.select $push[[L2:[0-9]+]]=, $pop[[L0]], $pop[[L1]], $0{{$}}
; CHECK-NEXT: i64x2.splat $push[[L3:[0-9]+]]=, $pop[[L2]]{{$}}
; CHECK-NEXT: v128.bitselect $push[[R:[0-9]+]]=, $1, $2, $pop[[L3]]{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @select_v2i64(i1 zeroext %c, <2 x i64> %x, <2 x i64> %y) {
  %res = select i1 %c, <2 x i64> %x, <2 x i64> %y
  ret <2 x i64> %res
}

; CHECK-LABEL: select_cmp_v2i64:
; CHECK-NEXT: .functype select_cmp_v2i64 (i32, v128, v128) -> (v128){{$}}
; CHECK-NEXT: i64.const $push[[L0:[0-9]+]]=, -1{{$}}
; CHECK-NEXT: i64.const $push[[L1:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: i32.const $push[[L2:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: i32.lt_s $push[[L3:[0-9]+]]=, $0, $pop[[L2]]{{$}}
; CHECK-NEXT: i64.select $push[[L4:[0-9]+]]=, $pop[[L0]], $pop[[L1]], $pop[[L3]]{{$}}
; CHECK-NEXT: i64x2.splat $push[[L5:[0-9]+]]=, $pop[[L4]]{{$}}
; CHECK-NEXT: v128.bitselect $push[[R:[0-9]+]]=, $1, $2, $pop[[L5]]{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @select_cmp_v2i64(i32 %i, <2 x i64> %x, <2 x i64> %y) {
  %c = icmp slt i32 %i, 0
  %res = select i1 %c, <2 x i64> %x, <2 x i64> %y
  ret <2 x i64> %res
}

; CHECK-LABEL: select_ne_v2i64:
; CHECK-NEXT: .functype select_ne_v2i64 (i32, v128, v128) -> (v128){{$}}
; CHECK-NEXT: i64.const $push[[L0:[0-9]+]]=, -1{{$}}
; CHECK-NEXT: i64.const $push[[L1:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: i64.select $push[[L2:[0-9]+]]=, $pop[[L0]], $pop[[L1]], $0{{$}}
; CHECK-NEXT: i64x2.splat $push[[L3:[0-9]+]]=, $pop[[L2]]{{$}}
; CHECK-NEXT: v128.bitselect $push[[R:[0-9]+]]=, $1, $2, $pop[[L3]]{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @select_ne_v2i64(i32 %i, <2 x i64> %x, <2 x i64> %y) {
  %c = icmp ne i32 %i, 0
  %res = select i1 %c, <2 x i64> %x, <2 x i64> %y
  ret <2 x i64> %res
}

; CHECK-LABEL: select_eq_v2i64:
; CHECK-NEXT: .functype select_eq_v2i64 (i32, v128, v128) -> (v128){{$}}
; CHECK-NEXT: i64.const $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: i64.const $push[[L1:[0-9]+]]=, -1{{$}}
; CHECK-NEXT: i64.select $push[[L2:[0-9]+]]=, $pop[[L0]], $pop[[L1]], $0{{$}}
; CHECK-NEXT: i64x2.splat $push[[L3:[0-9]+]]=, $pop[[L2]]{{$}}
; CHECK-NEXT: v128.bitselect $push[[R:[0-9]+]]=, $1, $2, $pop[[L3]]{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <2 x i64> @select_eq_v2i64(i32 %i, <2 x i64> %x, <2 x i64> %y) {
  %c = icmp eq i32 %i, 0
  %res = select i1 %c, <2 x i64> %x, <2 x i64> %y
  ret <2 x i64> %res
}

; ==============================================================================
; 4 x float
; ==============================================================================
; CHECK-LABEL: vselect_v4f32:
; CHECK-NEXT: .functype vselect_v4f32 (v128, v128, v128) -> (v128){{$}}
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

; CHECK-LABEL: vselect_cmp_v4f32:
; CHECK-NEXT: .functype vselect_cmp_v4f32 (v128, v128, v128, v128) -> (v128){{$}}
; CHECK-NEXT: f32x4.lt $push[[L0:[0-9]+]]=, $0, $1{{$}}
; CHECK-NEXT: v128.bitselect $push[[R:[0-9]+]]=, $2, $3, $pop[[L0]]{{$}}
; CHECK-NEXT: return  $pop[[R]]{{$}}
define <4 x float> @vselect_cmp_v4f32(<4 x float> %a, <4 x float> %b,
                                      <4 x float> %x, <4 x float> %y) {
  %c = fcmp olt <4 x float> %a, %b
  %res = select <4 x i1> %c, <4 x float> %x, <4 x float> %y
  ret <4 x float> %res
}

; CHECK-LABEL: select_v4f32:
; CHECK-NEXT: .functype select_v4f32 (i32, v128, v128) -> (v128){{$}}
; CHECK-NEXT: i32.const $push[[L0:[0-9]+]]=, -1{{$}}
; CHECK-NEXT: i32.const $push[[L1:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: i32.select $push[[L2:[0-9]+]]=, $pop[[L0]], $pop[[L1]], $0{{$}}
; CHECK-NEXT: i32x4.splat $push[[L3:[0-9]+]]=, $pop[[L2]]{{$}}
; CHECK-NEXT: v128.bitselect $push[[R:[0-9]+]]=, $1, $2, $pop[[L3]]{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <4 x float> @select_v4f32(i1 zeroext %c, <4 x float> %x, <4 x float> %y) {
  %res = select i1 %c, <4 x float> %x, <4 x float> %y
  ret <4 x float> %res
}

; CHECK-LABEL: select_cmp_v4f32:
; CHECK-NEXT: .functype select_cmp_v4f32 (i32, v128, v128) -> (v128){{$}}
; CHECK-NEXT: i32.const $push[[L0:[0-9]+]]=, 31{{$}}
; CHECK-NEXT: i32.shr_s $push[[L1:[0-9]+]]=, $0, $pop[[L0]]{{$}}
; CHECK-NEXT: i32x4.splat $push[[L2:[0-9]+]]=, $pop[[L1]]{{$}}
; CHECK-NEXT: v128.bitselect $push[[R:[0-9]+]]=, $1, $2, $pop[[L2]]{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <4 x float> @select_cmp_v4f32(i32 %i, <4 x float> %x, <4 x float> %y) {
  %c = icmp slt i32 %i, 0
  %res = select i1 %c, <4 x float> %x, <4 x float> %y
  ret <4 x float> %res
}

; CHECK-LABEL: select_ne_v4f32:
; CHECK-NEXT: .functype select_ne_v4f32 (i32, v128, v128) -> (v128){{$}}
; CHECK-NEXT: i32.const $push[[L0:[0-9]+]]=, -1{{$}}
; CHECK-NEXT: i32.const $push[[L1:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: i32.select $push[[L2:[0-9]+]]=, $pop[[L0]], $pop[[L1]], $0{{$}}
; CHECK-NEXT: i32x4.splat $push[[L3:[0-9]+]]=, $pop[[L2]]{{$}}
; CHECK-NEXT: v128.bitselect $push[[R:[0-9]+]]=, $1, $2, $pop[[L3]]{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <4 x float> @select_ne_v4f32(i32 %i, <4 x float> %x, <4 x float> %y) {
  %c = icmp ne i32 %i, 0
  %res = select i1 %c, <4 x float> %x, <4 x float> %y
  ret <4 x float> %res
}

; CHECK-LABEL: select_eq_v4f32:
; CHECK-NEXT: .functype select_eq_v4f32 (i32, v128, v128) -> (v128){{$}}
; CHECK-NEXT: i32.const $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: i32.const $push[[L1:[0-9]+]]=, -1{{$}}
; CHECK-NEXT: i32.select $push[[L2:[0-9]+]]=, $pop[[L0]], $pop[[L1]], $0{{$}}
; CHECK-NEXT: i32x4.splat $push[[L3:[0-9]+]]=, $pop[[L2]]{{$}}
; CHECK-NEXT: v128.bitselect $push[[R:[0-9]+]]=, $1, $2, $pop[[L3]]{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <4 x float> @select_eq_v4f32(i32 %i, <4 x float> %x, <4 x float> %y) {
  %c = icmp eq i32 %i, 0
  %res = select i1 %c, <4 x float> %x, <4 x float> %y
  ret <4 x float> %res
}

; ==============================================================================
; 2 x double
; ==============================================================================
; CHECK-LABEL: vselect_v2f64:
; CHECK-NEXT: .functype vselect_v2f64 (v128, v128, v128) -> (v128){{$}}
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

; CHECK-LABEL: vselect_cmp_v2f64:
; CHECK-NEXT: .functype vselect_cmp_v2f64 (v128, v128, v128, v128) -> (v128){{$}}
; CHECK-NEXT: f64x2.lt $push[[L0:[0-9]+]]=, $0, $1{{$}}
; CHECK-NEXT: v128.bitselect $push[[R:[0-9]+]]=, $2, $3, $pop[[L0]]{{$}}
; CHECK-NEXT: return  $pop[[R]]{{$}}
define <2 x double> @vselect_cmp_v2f64(<2 x double> %a, <2 x double> %b,
                                       <2 x double> %x, <2 x double> %y) {
  %c = fcmp olt <2 x double> %a, %b
  %res = select <2 x i1> %c, <2 x double> %x, <2 x double> %y
  ret <2 x double> %res
}

; CHECK-LABEL: select_v2f64:
; CHECK-NEXT: .functype select_v2f64 (i32, v128, v128) -> (v128){{$}}
; CHECK-NEXT: i64.const $push[[L0:[0-9]+]]=, -1{{$}}
; CHECK-NEXT: i64.const $push[[L1:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: i64.select $push[[L2:[0-9]+]]=, $pop[[L0]], $pop[[L1]], $0{{$}}
; CHECK-NEXT: i64x2.splat $push[[L3:[0-9]+]]=, $pop[[L2]]{{$}}
; CHECK-NEXT: v128.bitselect $push[[R:[0-9]+]]=, $1, $2, $pop[[L3]]{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <2 x double> @select_v2f64(i1 zeroext %c, <2 x double> %x, <2 x double> %y) {
  %res = select i1 %c, <2 x double> %x, <2 x double> %y
  ret <2 x double> %res
}

; CHECK-LABEL: select_cmp_v2f64:
; CHECK-NEXT: .functype select_cmp_v2f64 (i32, v128, v128) -> (v128){{$}}
; CHECK-NEXT: i64.const $push[[L0:[0-9]+]]=, -1{{$}}
; CHECK-NEXT: i64.const $push[[L1:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: i32.const $push[[L2:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: i32.lt_s $push[[L3:[0-9]+]]=, $0, $pop[[L2]]{{$}}
; CHECK-NEXT: i64.select $push[[L4:[0-9]+]]=, $pop[[L0]], $pop[[L1]], $pop[[L3]]{{$}}
; CHECK-NEXT: i64x2.splat $push[[L5:[0-9]+]]=, $pop[[L4]]{{$}}
; CHECK-NEXT: v128.bitselect $push[[R:[0-9]+]]=, $1, $2, $pop[[L5]]{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <2 x double> @select_cmp_v2f64(i32 %i, <2 x double> %x, <2 x double> %y) {
  %c = icmp slt i32 %i, 0
  %res = select i1 %c, <2 x double> %x, <2 x double> %y
  ret <2 x double> %res
}

; CHECK-LABEL: select_ne_v2f64:
; CHECK-NEXT: .functype select_ne_v2f64 (i32, v128, v128) -> (v128){{$}}
; CHECK-NEXT: i64.const $push[[L0:[0-9]+]]=, -1{{$}}
; CHECK-NEXT: i64.const $push[[L1:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: i64.select $push[[L2:[0-9]+]]=, $pop[[L0]], $pop[[L1]], $0{{$}}
; CHECK-NEXT: i64x2.splat $push[[L3:[0-9]+]]=, $pop[[L2]]{{$}}
; CHECK-NEXT: v128.bitselect $push[[R:[0-9]+]]=, $1, $2, $pop[[L3]]{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <2 x double> @select_ne_v2f64(i32 %i, <2 x double> %x, <2 x double> %y) {
  %c = icmp ne i32 %i, 0
  %res = select i1 %c, <2 x double> %x, <2 x double> %y
  ret <2 x double> %res
}

; CHECK-LABEL: select_eq_v2f64:
; CHECK-NEXT: .functype select_eq_v2f64 (i32, v128, v128) -> (v128){{$}}
; CHECK-NEXT: i64.const $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: i64.const $push[[L1:[0-9]+]]=, -1{{$}}
; CHECK-NEXT: i64.select $push[[L2:[0-9]+]]=, $pop[[L0]], $pop[[L1]], $0{{$}}
; CHECK-NEXT: i64x2.splat $push[[L3:[0-9]+]]=, $pop[[L2]]{{$}}
; CHECK-NEXT: v128.bitselect $push[[R:[0-9]+]]=, $1, $2, $pop[[L3]]{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <2 x double> @select_eq_v2f64(i32 %i, <2 x double> %x, <2 x double> %y) {
  %c = icmp eq i32 %i, 0
  %res = select i1 %c, <2 x double> %x, <2 x double> %y
  ret <2 x double> %res
}
