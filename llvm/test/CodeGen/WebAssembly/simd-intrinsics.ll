; RUN: llc < %s -asm-verbose=false -verify-machineinstrs -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers -mattr=+simd128 | FileCheck %s
; RUN: llc < %s -asm-verbose=false -verify-machineinstrs -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers -mattr=+simd128 -fast-isel | FileCheck %s

; Test that SIMD128 intrinsics lower as expected. These intrinsics are
; only expected to lower successfully if the simd128 attribute is
; enabled and legal types are used.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; ==============================================================================
; 16 x i8
; ==============================================================================
; CHECK-LABEL: swizzle_v16i8:
; CHECK-NEXT: .functype swizzle_v16i8 (v128, v128) -> (v128){{$}}
; CHECK-NEXT: i8x16.swizzle $push[[R:[0-9]+]]=, $0, $1{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
declare <16 x i8> @llvm.wasm.swizzle(<16 x i8>, <16 x i8>)
define <16 x i8> @swizzle_v16i8(<16 x i8> %x, <16 x i8> %y) {
  %a = call <16 x i8> @llvm.wasm.swizzle(<16 x i8> %x, <16 x i8> %y)
  ret <16 x i8> %a
}

; CHECK-LABEL: add_sat_s_v16i8:
; CHECK-NEXT: .functype add_sat_s_v16i8 (v128, v128) -> (v128){{$}}
; CHECK-NEXT: i8x16.add_sat_s $push[[R:[0-9]+]]=, $0, $1{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
declare <16 x i8> @llvm.sadd.sat.v16i8(<16 x i8>, <16 x i8>)
define <16 x i8> @add_sat_s_v16i8(<16 x i8> %x, <16 x i8> %y) {
  %a = call <16 x i8> @llvm.sadd.sat.v16i8(<16 x i8> %x, <16 x i8> %y)
  ret <16 x i8> %a
}

; CHECK-LABEL: add_sat_u_v16i8:
; CHECK-NEXT: .functype add_sat_u_v16i8 (v128, v128) -> (v128){{$}}
; CHECK-NEXT: i8x16.add_sat_u $push[[R:[0-9]+]]=, $0, $1{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
declare <16 x i8> @llvm.uadd.sat.v16i8(<16 x i8>, <16 x i8>)
define <16 x i8> @add_sat_u_v16i8(<16 x i8> %x, <16 x i8> %y) {
  %a = call <16 x i8> @llvm.uadd.sat.v16i8(<16 x i8> %x, <16 x i8> %y)
  ret <16 x i8> %a
}

; CHECK-LABEL: sub_sat_s_v16i8:
; CHECK-NEXT: .functype sub_sat_s_v16i8 (v128, v128) -> (v128){{$}}
; CHECK-NEXT: i8x16.sub_sat_s $push[[R:[0-9]+]]=, $0, $1{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
declare <16 x i8> @llvm.wasm.sub.sat.signed.v16i8(<16 x i8>, <16 x i8>)
define <16 x i8> @sub_sat_s_v16i8(<16 x i8> %x, <16 x i8> %y) {
  %a = call <16 x i8> @llvm.wasm.sub.sat.signed.v16i8(
    <16 x i8> %x, <16 x i8> %y
  )
  ret <16 x i8> %a
}

; CHECK-LABEL: sub_sat_u_v16i8:
; CHECK-NEXT: .functype sub_sat_u_v16i8 (v128, v128) -> (v128){{$}}
; CHECK-NEXT: i8x16.sub_sat_u $push[[R:[0-9]+]]=, $0, $1{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
declare <16 x i8> @llvm.wasm.sub.sat.unsigned.v16i8(<16 x i8>, <16 x i8>)
define <16 x i8> @sub_sat_u_v16i8(<16 x i8> %x, <16 x i8> %y) {
  %a = call <16 x i8> @llvm.wasm.sub.sat.unsigned.v16i8(
    <16 x i8> %x, <16 x i8> %y
  )
  ret <16 x i8> %a
}

; CHECK-LABEL: avgr_u_v16i8:
; CHECK-NEXT: .functype avgr_u_v16i8 (v128, v128) -> (v128){{$}}
; CHECK-NEXT: i8x16.avgr_u $push[[R:[0-9]+]]=, $0, $1{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
declare <16 x i8> @llvm.wasm.avgr.unsigned.v16i8(<16 x i8>, <16 x i8>)
define <16 x i8> @avgr_u_v16i8(<16 x i8> %x, <16 x i8> %y) {
  %a = call <16 x i8> @llvm.wasm.avgr.unsigned.v16i8(<16 x i8> %x, <16 x i8> %y)
  ret <16 x i8> %a
}

; CHECK-LABEL: popcnt_v16i8:
; CHECK-NEXT: .functype popcnt_v16i8 (v128) -> (v128){{$}}
; CHECK-NEXT: i8x16.popcnt $push[[R:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
declare <16 x i8> @llvm.wasm.popcnt(<16 x i8>)
define <16 x i8> @popcnt_v16i8(<16 x i8> %x) {
 %a = call <16 x i8> @llvm.wasm.popcnt(<16 x i8> %x)
 ret <16 x i8> %a
}

; CHECK-LABEL: any_v16i8:
; CHECK-NEXT: .functype any_v16i8 (v128) -> (i32){{$}}
; CHECK-NEXT: i8x16.any_true $push[[R:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
declare i32 @llvm.wasm.anytrue.v16i8(<16 x i8>)
define i32 @any_v16i8(<16 x i8> %x) {
  %a = call i32 @llvm.wasm.anytrue.v16i8(<16 x i8> %x)
  ret i32 %a
}

; CHECK-LABEL: all_v16i8:
; CHECK-NEXT: .functype all_v16i8 (v128) -> (i32){{$}}
; CHECK-NEXT: i8x16.all_true $push[[R:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
declare i32 @llvm.wasm.alltrue.v16i8(<16 x i8>)
define i32 @all_v16i8(<16 x i8> %x) {
  %a = call i32 @llvm.wasm.alltrue.v16i8(<16 x i8> %x)
  ret i32 %a
}

; CHECK-LABEL: bitmask_v16i8:
; CHECK-NEXT: .functype bitmask_v16i8 (v128) -> (i32){{$}}
; CHECK-NEXT: i8x16.bitmask $push[[R:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
declare i32 @llvm.wasm.bitmask.v16i8(<16 x i8>)
define i32 @bitmask_v16i8(<16 x i8> %x) {
  %a = call i32 @llvm.wasm.bitmask.v16i8(<16 x i8> %x)
  ret i32 %a
}

; CHECK-LABEL: bitselect_v16i8:
; CHECK-NEXT: .functype bitselect_v16i8 (v128, v128, v128) -> (v128){{$}}
; CHECK-NEXT: v128.bitselect $push[[R:[0-9]+]]=, $0, $1, $2{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
declare <16 x i8> @llvm.wasm.bitselect.v16i8(<16 x i8>, <16 x i8>, <16 x i8>)
define <16 x i8> @bitselect_v16i8(<16 x i8> %v1, <16 x i8> %v2, <16 x i8> %c) {
  %a = call <16 x i8> @llvm.wasm.bitselect.v16i8(
     <16 x i8> %v1, <16 x i8> %v2, <16 x i8> %c
  )
  ret <16 x i8> %a
}

; CHECK-LABEL: narrow_signed_v16i8:
; CHECK-NEXT: .functype narrow_signed_v16i8 (v128, v128) -> (v128){{$}}
; CHECK-NEXT: i8x16.narrow_i16x8_s $push[[R:[0-9]+]]=, $0, $1{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
declare <16 x i8> @llvm.wasm.narrow.signed.v16i8.v8i16(<8 x i16>, <8 x i16>)
define <16 x i8> @narrow_signed_v16i8(<8 x i16> %low, <8 x i16> %high) {
  %a = call <16 x i8> @llvm.wasm.narrow.signed.v16i8.v8i16(
    <8 x i16> %low, <8 x i16> %high
  )
  ret <16 x i8> %a
}

; CHECK-LABEL: narrow_unsigned_v16i8:
; CHECK-NEXT: .functype narrow_unsigned_v16i8 (v128, v128) -> (v128){{$}}
; CHECK-NEXT: i8x16.narrow_i16x8_u $push[[R:[0-9]+]]=, $0, $1{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
declare <16 x i8> @llvm.wasm.narrow.unsigned.v16i8.v8i16(<8 x i16>, <8 x i16>)
define <16 x i8> @narrow_unsigned_v16i8(<8 x i16> %low, <8 x i16> %high) {
  %a = call <16 x i8> @llvm.wasm.narrow.unsigned.v16i8.v8i16(
    <8 x i16> %low, <8 x i16> %high
  )
  ret <16 x i8> %a
}

; CHECK-LABEL: shuffle_v16i8:
; NO-CHECK-NOT: i8x16
; CHECK-NEXT: .functype shuffle_v16i8 (v128, v128) -> (v128){{$}}
; CHECK-NEXT: i8x16.shuffle $push[[R:[0-9]+]]=, $0, $1,
; CHECK-SAME: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
declare <16 x i8> @llvm.wasm.shuffle(
  <16 x i8>, <16 x i8>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32,
  i32, i32, i32, i32, i32)
define <16 x i8> @shuffle_v16i8(<16 x i8> %x, <16 x i8> %y) {
  %res = call <16 x i8> @llvm.wasm.shuffle(<16 x i8> %x, <16 x i8> %y,
      i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7,
      i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 35)
  ret <16 x i8> %res
}

; CHECK-LABEL: shuffle_undef_v16i8:
; NO-CHECK-NOT: i8x16
; CHECK-NEXT: .functype shuffle_undef_v16i8 (v128, v128) -> (v128){{$}}
; CHECK-NEXT: i8x16.shuffle $push[[R:[0-9]+]]=, $0, $1,
; CHECK-SAME: 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
define <16 x i8> @shuffle_undef_v16i8(<16 x i8> %x, <16 x i8> %y) {
  %res = call <16 x i8> @llvm.wasm.shuffle(<16 x i8> %x, <16 x i8> %y,
      i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef,
      i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef,
      i32 undef, i32 undef, i32 undef, i32 2)
  ret <16 x i8> %res
}

; ==============================================================================
; 8 x i16
; ==============================================================================
; CHECK-LABEL: add_sat_s_v8i16:
; CHECK-NEXT: .functype add_sat_s_v8i16 (v128, v128) -> (v128){{$}}
; CHECK-NEXT: i16x8.add_sat_s $push[[R:[0-9]+]]=, $0, $1{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
declare <8 x i16> @llvm.sadd.sat.v8i16(<8 x i16>, <8 x i16>)
define <8 x i16> @add_sat_s_v8i16(<8 x i16> %x, <8 x i16> %y) {
  %a = call <8 x i16> @llvm.sadd.sat.v8i16(<8 x i16> %x, <8 x i16> %y)
  ret <8 x i16> %a
}

; CHECK-LABEL: add_sat_u_v8i16:
; CHECK-NEXT: .functype add_sat_u_v8i16 (v128, v128) -> (v128){{$}}
; CHECK-NEXT: i16x8.add_sat_u $push[[R:[0-9]+]]=, $0, $1{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
declare <8 x i16> @llvm.uadd.sat.v8i16(<8 x i16>, <8 x i16>)
define <8 x i16> @add_sat_u_v8i16(<8 x i16> %x, <8 x i16> %y) {
  %a = call <8 x i16> @llvm.uadd.sat.v8i16(<8 x i16> %x, <8 x i16> %y)
  ret <8 x i16> %a
}

; CHECK-LABEL: sub_sat_s_v8i16:
; CHECK-NEXT: .functype sub_sat_s_v8i16 (v128, v128) -> (v128){{$}}
; CHECK-NEXT: i16x8.sub_sat_s $push[[R:[0-9]+]]=, $0, $1{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
declare <8 x i16> @llvm.wasm.sub.sat.signed.v8i16(<8 x i16>, <8 x i16>)
define <8 x i16> @sub_sat_s_v8i16(<8 x i16> %x, <8 x i16> %y) {
  %a = call <8 x i16> @llvm.wasm.sub.sat.signed.v8i16(
    <8 x i16> %x, <8 x i16> %y
  )
  ret <8 x i16> %a
}

; CHECK-LABEL: sub_sat_u_v8i16:
; CHECK-NEXT: .functype sub_sat_u_v8i16 (v128, v128) -> (v128){{$}}
; CHECK-NEXT: i16x8.sub_sat_u $push[[R:[0-9]+]]=, $0, $1{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
declare <8 x i16> @llvm.wasm.sub.sat.unsigned.v8i16(<8 x i16>, <8 x i16>)
define <8 x i16> @sub_sat_u_v8i16(<8 x i16> %x, <8 x i16> %y) {
  %a = call <8 x i16> @llvm.wasm.sub.sat.unsigned.v8i16(
    <8 x i16> %x, <8 x i16> %y
  )
  ret <8 x i16> %a
}

; CHECK-LABEL: avgr_u_v8i16:
; CHECK-NEXT: .functype avgr_u_v8i16 (v128, v128) -> (v128){{$}}
; CHECK-NEXT: i16x8.avgr_u $push[[R:[0-9]+]]=, $0, $1{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
declare <8 x i16> @llvm.wasm.avgr.unsigned.v8i16(<8 x i16>, <8 x i16>)
define <8 x i16> @avgr_u_v8i16(<8 x i16> %x, <8 x i16> %y) {
  %a = call <8 x i16> @llvm.wasm.avgr.unsigned.v8i16(<8 x i16> %x, <8 x i16> %y)
  ret <8 x i16> %a
}

; CHECK-LABEL: q15mulr_sat_s_v8i16:
; CHECK-NEXT: .functype q15mulr_sat_s_v8i16 (v128, v128) -> (v128){{$}}
; CHECK-NEXT: i16x8.q15mulr_sat_s $push[[R:[0-9]+]]=, $0, $1{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
declare <8 x i16> @llvm.wasm.q15mulr.sat.signed(<8 x i16>, <8 x i16>)
define <8 x i16> @q15mulr_sat_s_v8i16(<8 x i16> %x, <8 x i16> %y) {
  %a = call <8 x i16> @llvm.wasm.q15mulr.sat.signed(<8 x i16> %x,
                                                         <8 x i16> %y)
  ret <8 x i16> %a
}

; CHECK-LABEL: extmul_low_s_v8i16:
; CHECK-NEXT: .functype extmul_low_s_v8i16 (v128, v128) -> (v128){{$}}
; CHECK-NEXT: i16x8.extmul_low_i8x16_s $push[[R:[0-9]+]]=, $0, $1{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
declare <8 x i16> @llvm.wasm.extmul.low.signed.v8i16(<16 x i8>, <16 x i8>)
define <8 x i16> @extmul_low_s_v8i16(<16 x i8> %x, <16 x i8> %y) {
  %a = call <8 x i16> @llvm.wasm.extmul.low.signed.v8i16(
    <16 x i8> %x, <16 x i8> %y
  )
  ret <8 x i16> %a
}

; CHECK-LABEL: extmul_high_s_v8i16:
; CHECK-NEXT: .functype extmul_high_s_v8i16 (v128, v128) -> (v128){{$}}
; CHECK-NEXT: i16x8.extmul_high_i8x16_s $push[[R:[0-9]+]]=, $0, $1{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
declare <8 x i16> @llvm.wasm.extmul.high.signed.v8i16(<16 x i8>, <16 x i8>)
define <8 x i16> @extmul_high_s_v8i16(<16 x i8> %x, <16 x i8> %y) {
  %a = call <8 x i16> @llvm.wasm.extmul.high.signed.v8i16(
    <16 x i8> %x, <16 x i8> %y
  )
  ret <8 x i16> %a
}

; CHECK-LABEL: extmul_low_u_v8i16:
; CHECK-NEXT: .functype extmul_low_u_v8i16 (v128, v128) -> (v128){{$}}
; CHECK-NEXT: i16x8.extmul_low_i8x16_u $push[[R:[0-9]+]]=, $0, $1{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
declare <8 x i16> @llvm.wasm.extmul.low.unsigned.v8i16(<16 x i8>, <16 x i8>)
define <8 x i16> @extmul_low_u_v8i16(<16 x i8> %x, <16 x i8> %y) {
  %a = call <8 x i16> @llvm.wasm.extmul.low.unsigned.v8i16(
    <16 x i8> %x, <16 x i8> %y
  )
  ret <8 x i16> %a
}

; CHECK-LABEL: extmul_high_u_v8i16:
; CHECK-NEXT: .functype extmul_high_u_v8i16 (v128, v128) -> (v128){{$}}
; CHECK-NEXT: i16x8.extmul_high_i8x16_u $push[[R:[0-9]+]]=, $0, $1{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
declare <8 x i16> @llvm.wasm.extmul.high.unsigned.v8i16(<16 x i8>, <16 x i8>)
define <8 x i16> @extmul_high_u_v8i16(<16 x i8> %x, <16 x i8> %y) {
  %a = call <8 x i16> @llvm.wasm.extmul.high.unsigned.v8i16(
    <16 x i8> %x, <16 x i8> %y
  )
  ret <8 x i16> %a
}

; CHECK-LABEL: extadd_pairwise_s_v8i16:
; CHECK-NEXT: .functype extadd_pairwise_s_v8i16 (v128) -> (v128){{$}}
; CHECK-NEXT: i16x8.extadd_pairwise_i8x16_s $push[[R:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
declare <8 x i16> @llvm.wasm.extadd.pairwise.signed.v8i16(<16 x i8>)
define <8 x i16> @extadd_pairwise_s_v8i16(<16 x i8> %x) {
  %a = call <8 x i16> @llvm.wasm.extadd.pairwise.signed.v8i16(<16 x i8> %x)
  ret <8 x i16> %a
}

; CHECK-LABEL: extadd_pairwise_u_v8i16:
; CHECK-NEXT: .functype extadd_pairwise_u_v8i16 (v128) -> (v128){{$}}
; CHECK-NEXT: i16x8.extadd_pairwise_i8x16_u $push[[R:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
declare <8 x i16> @llvm.wasm.extadd.pairwise.unsigned.v8i16(<16 x i8>)
define <8 x i16> @extadd_pairwise_u_v8i16(<16 x i8> %x) {
  %a = call <8 x i16> @llvm.wasm.extadd.pairwise.unsigned.v8i16(<16 x i8> %x)
  ret <8 x i16> %a
}

; CHECK-LABEL: any_v8i16:
; CHECK-NEXT: .functype any_v8i16 (v128) -> (i32){{$}}
; CHECK-NEXT: i16x8.any_true $push[[R:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
declare i32 @llvm.wasm.anytrue.v8i16(<8 x i16>)
define i32 @any_v8i16(<8 x i16> %x) {
  %a = call i32 @llvm.wasm.anytrue.v8i16(<8 x i16> %x)
  ret i32 %a
}

; CHECK-LABEL: all_v8i16:
; CHECK-NEXT: .functype all_v8i16 (v128) -> (i32){{$}}
; CHECK-NEXT: i16x8.all_true $push[[R:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
declare i32 @llvm.wasm.alltrue.v8i16(<8 x i16>)
define i32 @all_v8i16(<8 x i16> %x) {
  %a = call i32 @llvm.wasm.alltrue.v8i16(<8 x i16> %x)
  ret i32 %a
}

; CHECK-LABEL: bitmask_v8i16:
; CHECK-NEXT: .functype bitmask_v8i16 (v128) -> (i32){{$}}
; CHECK-NEXT: i16x8.bitmask $push[[R:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
declare i32 @llvm.wasm.bitmask.v8i16(<8 x i16>)
define i32 @bitmask_v8i16(<8 x i16> %x) {
  %a = call i32 @llvm.wasm.bitmask.v8i16(<8 x i16> %x)
  ret i32 %a
}

; CHECK-LABEL: bitselect_v8i16:
; CHECK-NEXT: .functype bitselect_v8i16 (v128, v128, v128) -> (v128){{$}}
; CHECK-NEXT: v128.bitselect $push[[R:[0-9]+]]=, $0, $1, $2{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
declare <8 x i16> @llvm.wasm.bitselect.v8i16(<8 x i16>, <8 x i16>, <8 x i16>)
define <8 x i16> @bitselect_v8i16(<8 x i16> %v1, <8 x i16> %v2, <8 x i16> %c) {
  %a = call <8 x i16> @llvm.wasm.bitselect.v8i16(
    <8 x i16> %v1, <8 x i16> %v2, <8 x i16> %c
  )
  ret <8 x i16> %a
}

; CHECK-LABEL: narrow_signed_v8i16:
; CHECK-NEXT: .functype narrow_signed_v8i16 (v128, v128) -> (v128){{$}}
; CHECK-NEXT: i16x8.narrow_i32x4_s $push[[R:[0-9]+]]=, $0, $1{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
declare <8 x i16> @llvm.wasm.narrow.signed.v8i16.v4i32(<4 x i32>, <4 x i32>)
define <8 x i16> @narrow_signed_v8i16(<4 x i32> %low, <4 x i32> %high) {
  %a = call <8 x i16> @llvm.wasm.narrow.signed.v8i16.v4i32(
    <4 x i32> %low, <4 x i32> %high
  )
  ret <8 x i16> %a
}

; CHECK-LABEL: narrow_unsigned_v8i16:
; CHECK-NEXT: .functype narrow_unsigned_v8i16 (v128, v128) -> (v128){{$}}
; CHECK-NEXT: i16x8.narrow_i32x4_u $push[[R:[0-9]+]]=, $0, $1{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
declare <8 x i16> @llvm.wasm.narrow.unsigned.v8i16.v4i32(<4 x i32>, <4 x i32>)
define <8 x i16> @narrow_unsigned_v8i16(<4 x i32> %low, <4 x i32> %high) {
  %a = call <8 x i16> @llvm.wasm.narrow.unsigned.v8i16.v4i32(
    <4 x i32> %low, <4 x i32> %high
  )
  ret <8 x i16> %a
}

; ==============================================================================
; 4 x i32
; ==============================================================================
; CHECK-LABEL: dot:
; CHECK-NEXT: .functype dot (v128, v128) -> (v128){{$}}
; CHECK-NEXT: i32x4.dot_i16x8_s $push[[R:[0-9]+]]=, $0, $1{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
declare <4 x i32> @llvm.wasm.dot(<8 x i16>, <8 x i16>)
define <4 x i32> @dot(<8 x i16> %x, <8 x i16> %y) {
  %a = call <4 x i32> @llvm.wasm.dot(<8 x i16> %x, <8 x i16> %y)
  ret <4 x i32> %a
}


; CHECK-LABEL: extmul_low_s_v4i32:
; CHECK-NEXT: .functype extmul_low_s_v4i32 (v128, v128) -> (v128){{$}}
; CHECK-NEXT: i32x4.extmul_low_i16x8_s $push[[R:[0-9]+]]=, $0, $1{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
declare <4 x i32> @llvm.wasm.extmul.low.signed.v4i32(<8 x i16>, <8 x i16>)
define <4 x i32> @extmul_low_s_v4i32(<8 x i16> %x, <8 x i16> %y) {
  %a = call <4 x i32> @llvm.wasm.extmul.low.signed.v4i32(
    <8 x i16> %x, <8 x i16> %y
  )
  ret <4 x i32> %a
}

; CHECK-LABEL: extmul_high_s_v4i32:
; CHECK-NEXT: .functype extmul_high_s_v4i32 (v128, v128) -> (v128){{$}}
; CHECK-NEXT: i32x4.extmul_high_i16x8_s $push[[R:[0-9]+]]=, $0, $1{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
declare <4 x i32> @llvm.wasm.extmul.high.signed.v4i32(<8 x i16>, <8 x i16>)
define <4 x i32> @extmul_high_s_v4i32(<8 x i16> %x, <8 x i16> %y) {
  %a = call <4 x i32> @llvm.wasm.extmul.high.signed.v4i32(
    <8 x i16> %x, <8 x i16> %y
  )
  ret <4 x i32> %a
}

; CHECK-LABEL: extmul_low_u_v4i32:
; CHECK-NEXT: .functype extmul_low_u_v4i32 (v128, v128) -> (v128){{$}}
; CHECK-NEXT: i32x4.extmul_low_i16x8_u $push[[R:[0-9]+]]=, $0, $1{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
declare <4 x i32> @llvm.wasm.extmul.low.unsigned.v4i32(<8 x i16>, <8 x i16>)
define <4 x i32> @extmul_low_u_v4i32(<8 x i16> %x, <8 x i16> %y) {
  %a = call <4 x i32> @llvm.wasm.extmul.low.unsigned.v4i32(
    <8 x i16> %x, <8 x i16> %y
  )
  ret <4 x i32> %a
}

; CHECK-LABEL: extmul_high_u_v4i32:
; CHECK-NEXT: .functype extmul_high_u_v4i32 (v128, v128) -> (v128){{$}}
; CHECK-NEXT: i32x4.extmul_high_i16x8_u $push[[R:[0-9]+]]=, $0, $1{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
declare <4 x i32> @llvm.wasm.extmul.high.unsigned.v4i32(<8 x i16>, <8 x i16>)
define <4 x i32> @extmul_high_u_v4i32(<8 x i16> %x, <8 x i16> %y) {
  %a = call <4 x i32> @llvm.wasm.extmul.high.unsigned.v4i32(
    <8 x i16> %x, <8 x i16> %y
  )
  ret <4 x i32> %a
}

; CHECK-LABEL: extadd_pairwise_s_v4i32:
; CHECK-NEXT: .functype extadd_pairwise_s_v4i32 (v128) -> (v128){{$}}
; CHECK-NEXT: i32x4.extadd_pairwise_i16x8_s $push[[R:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
declare <4 x i32> @llvm.wasm.extadd.pairwise.signed.v4i32(<8 x i16>)
define <4 x i32> @extadd_pairwise_s_v4i32(<8 x i16> %x) {
  %a = call <4 x i32> @llvm.wasm.extadd.pairwise.signed.v4i32(<8 x i16> %x)
  ret <4 x i32> %a
}

; CHECK-LABEL: extadd_pairwise_u_v4i32:
; CHECK-NEXT: .functype extadd_pairwise_u_v4i32 (v128) -> (v128){{$}}
; CHECK-NEXT: i32x4.extadd_pairwise_i16x8_u $push[[R:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
declare <4 x i32> @llvm.wasm.extadd.pairwise.unsigned.v4i32(<8 x i16>)
define <4 x i32> @extadd_pairwise_u_v4i32(<8 x i16> %x) {
  %a = call <4 x i32> @llvm.wasm.extadd.pairwise.unsigned.v4i32(<8 x i16> %x)
  ret <4 x i32> %a
}


; CHECK-LABEL: any_v4i32:
; CHECK-NEXT: .functype any_v4i32 (v128) -> (i32){{$}}
; CHECK-NEXT: i32x4.any_true $push[[R:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
declare i32 @llvm.wasm.anytrue.v4i32(<4 x i32>)
define i32 @any_v4i32(<4 x i32> %x) {
  %a = call i32 @llvm.wasm.anytrue.v4i32(<4 x i32> %x)
  ret i32 %a
}

; CHECK-LABEL: all_v4i32:
; CHECK-NEXT: .functype all_v4i32 (v128) -> (i32){{$}}
; CHECK-NEXT: i32x4.all_true $push[[R:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
declare i32 @llvm.wasm.alltrue.v4i32(<4 x i32>)
define i32 @all_v4i32(<4 x i32> %x) {
  %a = call i32 @llvm.wasm.alltrue.v4i32(<4 x i32> %x)
  ret i32 %a
}

; CHECK-LABEL: bitmask_v4i32:
; CHECK-NEXT: .functype bitmask_v4i32 (v128) -> (i32){{$}}
; CHECK-NEXT: i32x4.bitmask $push[[R:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
declare i32 @llvm.wasm.bitmask.v4i32(<4 x i32>)
define i32 @bitmask_v4i32(<4 x i32> %x) {
  %a = call i32 @llvm.wasm.bitmask.v4i32(<4 x i32> %x)
  ret i32 %a
}

; CHECK-LABEL: bitselect_v4i32:
; CHECK-NEXT: .functype bitselect_v4i32 (v128, v128, v128) -> (v128){{$}}
; CHECK-NEXT: v128.bitselect $push[[R:[0-9]+]]=, $0, $1, $2{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
declare <4 x i32> @llvm.wasm.bitselect.v4i32(<4 x i32>, <4 x i32>, <4 x i32>)
define <4 x i32> @bitselect_v4i32(<4 x i32> %v1, <4 x i32> %v2, <4 x i32> %c) {
  %a = call <4 x i32> @llvm.wasm.bitselect.v4i32(
    <4 x i32> %v1, <4 x i32> %v2, <4 x i32> %c
  )
  ret <4 x i32> %a
}

; CHECK-LABEL: trunc_sat_s_v4i32:
; NO-CHECK-NOT: f32x4
; CHECK-NEXT: .functype trunc_sat_s_v4i32 (v128) -> (v128){{$}}
; CHECK-NEXT: i32x4.trunc_sat_f32x4_s $push[[R:[0-9]+]]=, $0
; CHECK-NEXT: return $pop[[R]]
declare <4 x i32> @llvm.wasm.trunc.saturate.signed.v4i32.v4f32(<4 x float>)
define <4 x i32> @trunc_sat_s_v4i32(<4 x float> %x) {
  %a = call <4 x i32> @llvm.wasm.trunc.saturate.signed.v4i32.v4f32(<4 x float> %x)
  ret <4 x i32> %a
}

; CHECK-LABEL: trunc_sat_u_v4i32:
; NO-CHECK-NOT: f32x4
; CHECK-NEXT: .functype trunc_sat_u_v4i32 (v128) -> (v128){{$}}
; CHECK-NEXT: i32x4.trunc_sat_f32x4_u $push[[R:[0-9]+]]=, $0
; CHECK-NEXT: return $pop[[R]]
declare <4 x i32> @llvm.wasm.trunc.saturate.unsigned.v4i32.v4f32(<4 x float>)
define <4 x i32> @trunc_sat_u_v4i32(<4 x float> %x) {
  %a = call <4 x i32> @llvm.wasm.trunc.saturate.unsigned.v4i32.v4f32(<4 x float> %x)
  ret <4 x i32> %a
}

; CHECK-LABEL: trunc_sat_zero_signed_v4i32:
; CHECK-NEXT: .functype trunc_sat_zero_signed_v4i32 (v128) -> (v128){{$}}
; CHECK-NEXT: i32x4.trunc_sat_zero_f64x2_s $push[[R:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
declare <4 x i32> @llvm.wasm.trunc.sat.zero.signed(<2 x double>)
define <4 x i32> @trunc_sat_zero_signed_v4i32(<2 x double> %a) {
  %v = call <4 x i32> @llvm.wasm.trunc.sat.zero.signed(<2 x double> %a)
  ret <4 x i32> %v
}

; CHECK-LABEL: trunc_sat_zero_unsigned_v4i32:
; CHECK-NEXT: .functype trunc_sat_zero_unsigned_v4i32 (v128) -> (v128){{$}}
; CHECK-NEXT: i32x4.trunc_sat_zero_f64x2_u $push[[R:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
declare <4 x i32> @llvm.wasm.trunc.sat.zero.unsigned(<2 x double>)
define <4 x i32> @trunc_sat_zero_unsigned_v4i32(<2 x double> %a) {
  %v = call <4 x i32> @llvm.wasm.trunc.sat.zero.unsigned(<2 x double> %a)
  ret <4 x i32> %v
}

; ==============================================================================
; 2 x i64
; ==============================================================================
; CHECK-LABEL: extend_low_s_v2i64:
; CHECK-NEXT: .functype extend_low_s_v2i64 (v128) -> (v128){{$}}
; CHECK-NEXT: i64x2.extend_low_i32x4_s $push[[R:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
declare <2 x i64> @llvm.wasm.extend.low.signed(<4 x i32>)
define <2 x i64> @extend_low_s_v2i64(<4 x i32> %x) {
  %a = call <2 x i64> @llvm.wasm.extend.low.signed(<4 x i32> %x)
  ret <2 x i64> %a
}

; CHECK-LABEL: extend_high_s_v2i64:
; CHECK-NEXT: .functype extend_high_s_v2i64 (v128) -> (v128){{$}}
; CHECK-NEXT: i64x2.extend_high_i32x4_s $push[[R:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
declare <2 x i64> @llvm.wasm.extend.high.signed(<4 x i32>)
define <2 x i64> @extend_high_s_v2i64(<4 x i32> %x) {
  %a = call <2 x i64> @llvm.wasm.extend.high.signed(<4 x i32> %x)
  ret <2 x i64> %a
}

; CHECK-LABEL: extend_low_u_v2i64:
; CHECK-NEXT: .functype extend_low_u_v2i64 (v128) -> (v128){{$}}
; CHECK-NEXT: i64x2.extend_low_i32x4_u $push[[R:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
declare <2 x i64> @llvm.wasm.extend.low.unsigned(<4 x i32>)
define <2 x i64> @extend_low_u_v2i64(<4 x i32> %x) {
  %a = call <2 x i64> @llvm.wasm.extend.low.unsigned(<4 x i32> %x)
  ret <2 x i64> %a
}

; CHECK-LABEL: extend_high_u_v2i64:
; CHECK-NEXT: .functype extend_high_u_v2i64 (v128) -> (v128){{$}}
; CHECK-NEXT: i64x2.extend_high_i32x4_u $push[[R:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
declare <2 x i64> @llvm.wasm.extend.high.unsigned(<4 x i32>)
define <2 x i64> @extend_high_u_v2i64(<4 x i32> %x) {
  %a = call <2 x i64> @llvm.wasm.extend.high.unsigned(<4 x i32> %x)
  ret <2 x i64> %a
}

; CHECK-LABEL: extmul_low_s_v2i64:
; CHECK-NEXT: .functype extmul_low_s_v2i64 (v128, v128) -> (v128){{$}}
; CHECK-NEXT: i64x2.extmul_low_i32x4_s $push[[R:[0-9]+]]=, $0, $1{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
declare <2 x i64> @llvm.wasm.extmul.low.signed.v2i64(<4 x i32>, <4 x i32>)
define <2 x i64> @extmul_low_s_v2i64(<4 x i32> %x, <4 x i32> %y) {
  %a = call <2 x i64> @llvm.wasm.extmul.low.signed.v2i64(
    <4 x i32> %x, <4 x i32> %y
  )
  ret <2 x i64> %a
}

; CHECK-LABEL: extmul_high_s_v2i64:
; CHECK-NEXT: .functype extmul_high_s_v2i64 (v128, v128) -> (v128){{$}}
; CHECK-NEXT: i64x2.extmul_high_i32x4_s $push[[R:[0-9]+]]=, $0, $1{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
declare <2 x i64> @llvm.wasm.extmul.high.signed.v2i64(<4 x i32>, <4 x i32>)
define <2 x i64> @extmul_high_s_v2i64(<4 x i32> %x, <4 x i32> %y) {
  %a = call <2 x i64> @llvm.wasm.extmul.high.signed.v2i64(
    <4 x i32> %x, <4 x i32> %y
  )
  ret <2 x i64> %a
}

; CHECK-LABEL: extmul_low_u_v2i64:
; CHECK-NEXT: .functype extmul_low_u_v2i64 (v128, v128) -> (v128){{$}}
; CHECK-NEXT: i64x2.extmul_low_i32x4_u $push[[R:[0-9]+]]=, $0, $1{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
declare <2 x i64> @llvm.wasm.extmul.low.unsigned.v2i64(<4 x i32>, <4 x i32>)
define <2 x i64> @extmul_low_u_v2i64(<4 x i32> %x, <4 x i32> %y) {
  %a = call <2 x i64> @llvm.wasm.extmul.low.unsigned.v2i64(
    <4 x i32> %x, <4 x i32> %y
  )
  ret <2 x i64> %a
}

; CHECK-LABEL: extmul_high_u_v2i64:
; CHECK-NEXT: .functype extmul_high_u_v2i64 (v128, v128) -> (v128){{$}}
; CHECK-NEXT: i64x2.extmul_high_i32x4_u $push[[R:[0-9]+]]=, $0, $1{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
declare <2 x i64> @llvm.wasm.extmul.high.unsigned.v2i64(<4 x i32>, <4 x i32>)
define <2 x i64> @extmul_high_u_v2i64(<4 x i32> %x, <4 x i32> %y) {
  %a = call <2 x i64> @llvm.wasm.extmul.high.unsigned.v2i64(
    <4 x i32> %x, <4 x i32> %y
  )
  ret <2 x i64> %a
}

; CHECK-LABEL: any_v2i64:
; CHECK-NEXT: .functype any_v2i64 (v128) -> (i32){{$}}
; CHECK-NEXT: i64x2.any_true $push[[R:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
declare i32 @llvm.wasm.anytrue.v2i64(<2 x i64>)
define i32 @any_v2i64(<2 x i64> %x) {
  %a = call i32 @llvm.wasm.anytrue.v2i64(<2 x i64> %x)
  ret i32 %a
}

; CHECK-LABEL: all_v2i64:
; CHECK-NEXT: .functype all_v2i64 (v128) -> (i32){{$}}
; CHECK-NEXT: i64x2.all_true $push[[R:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
declare i32 @llvm.wasm.alltrue.v2i64(<2 x i64>)
define i32 @all_v2i64(<2 x i64> %x) {
  %a = call i32 @llvm.wasm.alltrue.v2i64(<2 x i64> %x)
  ret i32 %a
}

; CHECK-LABEL: bitmask_v2i64:
; CHECK-NEXT: .functype bitmask_v2i64 (v128) -> (i32){{$}}
; CHECK-NEXT: i64x2.bitmask $push[[R:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
declare i32 @llvm.wasm.bitmask.v2i64(<2 x i64>)
define i32 @bitmask_v2i64(<2 x i64> %x) {
  %a = call i32 @llvm.wasm.bitmask.v2i64(<2 x i64> %x)
  ret i32 %a
}

; CHECK-LABEL: bitselect_v2i64:
; CHECK-NEXT: .functype bitselect_v2i64 (v128, v128, v128) -> (v128){{$}}
; CHECK-NEXT: v128.bitselect $push[[R:[0-9]+]]=, $0, $1, $2{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
declare <2 x i64> @llvm.wasm.bitselect.v2i64(<2 x i64>, <2 x i64>, <2 x i64>)
define <2 x i64> @bitselect_v2i64(<2 x i64> %v1, <2 x i64> %v2, <2 x i64> %c) {
  %a = call <2 x i64> @llvm.wasm.bitselect.v2i64(
    <2 x i64> %v1, <2 x i64> %v2, <2 x i64> %c
  )
  ret <2 x i64> %a
}

; ==============================================================================
; 4 x f32
; ==============================================================================
; CHECK-LABEL: bitselect_v4f32:
; CHECK-NEXT: .functype bitselect_v4f32 (v128, v128, v128) -> (v128){{$}}
; CHECK-NEXT: v128.bitselect $push[[R:[0-9]+]]=, $0, $1, $2{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
declare <4 x float> @llvm.wasm.bitselect.v4f32(<4 x float>, <4 x float>, <4 x float>)
define <4 x float> @bitselect_v4f32(<4 x float> %v1, <4 x float> %v2, <4 x float> %c) {
  %a = call <4 x float> @llvm.wasm.bitselect.v4f32(
    <4 x float> %v1, <4 x float> %v2, <4 x float> %c
  )
  ret <4 x float> %a
}

; CHECK-LABEL: pmin_v4f32:
; CHECK-NEXT: .functype pmin_v4f32 (v128, v128) -> (v128){{$}}
; CHECK-NEXT: f32x4.pmin $push[[R:[0-9]+]]=, $0, $1{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
declare <4 x float> @llvm.wasm.pmin.v4f32(<4 x float>, <4 x float>)
define <4 x float> @pmin_v4f32(<4 x float> %a, <4 x float> %b) {
  %v = call <4 x float> @llvm.wasm.pmin.v4f32(<4 x float> %a, <4 x float> %b)
  ret <4 x float> %v
}

; CHECK-LABEL: pmax_v4f32:
; CHECK-NEXT: .functype pmax_v4f32 (v128, v128) -> (v128){{$}}
; CHECK-NEXT: f32x4.pmax $push[[R:[0-9]+]]=, $0, $1{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
declare <4 x float> @llvm.wasm.pmax.v4f32(<4 x float>, <4 x float>)
define <4 x float> @pmax_v4f32(<4 x float> %a, <4 x float> %b) {
  %v = call <4 x float> @llvm.wasm.pmax.v4f32(<4 x float> %a, <4 x float> %b)
  ret <4 x float> %v
}

; CHECK-LABEL: ceil_v4f32:
; CHECK-NEXT: .functype ceil_v4f32 (v128) -> (v128){{$}}
; CHECK-NEXT: f32x4.ceil $push[[R:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
declare <4 x float> @llvm.wasm.ceil.v4f32(<4 x float>)
define <4 x float> @ceil_v4f32(<4 x float> %a) {
  %v = call <4 x float> @llvm.wasm.ceil.v4f32(<4 x float> %a)
  ret <4 x float> %v
}

; CHECK-LABEL: floor_v4f32:
; CHECK-NEXT: .functype floor_v4f32 (v128) -> (v128){{$}}
; CHECK-NEXT: f32x4.floor $push[[R:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
declare <4 x float> @llvm.wasm.floor.v4f32(<4 x float>)
define <4 x float> @floor_v4f32(<4 x float> %a) {
  %v = call <4 x float> @llvm.wasm.floor.v4f32(<4 x float> %a)
  ret <4 x float> %v
}

; CHECK-LABEL: trunc_v4f32:
; CHECK-NEXT: .functype trunc_v4f32 (v128) -> (v128){{$}}
; CHECK-NEXT: f32x4.trunc $push[[R:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
declare <4 x float> @llvm.wasm.trunc.v4f32(<4 x float>)
define <4 x float> @trunc_v4f32(<4 x float> %a) {
  %v = call <4 x float> @llvm.wasm.trunc.v4f32(<4 x float> %a)
  ret <4 x float> %v
}

; CHECK-LABEL: nearest_v4f32:
; CHECK-NEXT: .functype nearest_v4f32 (v128) -> (v128){{$}}
; CHECK-NEXT: f32x4.nearest $push[[R:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
declare <4 x float> @llvm.wasm.nearest.v4f32(<4 x float>)
define <4 x float> @nearest_v4f32(<4 x float> %a) {
  %v = call <4 x float> @llvm.wasm.nearest.v4f32(<4 x float> %a)
  ret <4 x float> %v
}

; CHECK-LABEL: demote_zero_v4f32:
; CHECK-NEXT: .functype demote_zero_v4f32 (v128) -> (v128){{$}}
; CHECK-NEXT: f32x4.demote_zero_f64x2 $push[[R:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
declare <4 x float> @llvm.wasm.demote.zero(<2 x double>)
define <4 x float> @demote_zero_v4f32(<2 x double> %a) {
  %v = call <4 x float> @llvm.wasm.demote.zero(<2 x double> %a)
  ret <4 x float> %v
}

; ==============================================================================
; 2 x f64
; ==============================================================================
; CHECK-LABEL: bitselect_v2f64:
; CHECK-NEXT: .functype bitselect_v2f64 (v128, v128, v128) -> (v128){{$}}
; CHECK-NEXT: v128.bitselect $push[[R:[0-9]+]]=, $0, $1, $2{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
declare <2 x double> @llvm.wasm.bitselect.v2f64(<2 x double>, <2 x double>, <2 x double>)
define <2 x double> @bitselect_v2f64(<2 x double> %v1, <2 x double> %v2, <2 x double> %c) {
  %a = call <2 x double> @llvm.wasm.bitselect.v2f64(
    <2 x double> %v1, <2 x double> %v2, <2 x double> %c
  )
  ret <2 x double> %a
}

; CHECK-LABEL: pmin_v2f64:
; CHECK-NEXT: .functype pmin_v2f64 (v128, v128) -> (v128){{$}}
; CHECK-NEXT: f64x2.pmin $push[[R:[0-9]+]]=, $0, $1{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
declare <2 x double> @llvm.wasm.pmin.v2f64(<2 x double>, <2 x double>)
define <2 x double> @pmin_v2f64(<2 x double> %a, <2 x double> %b) {
  %v = call <2 x double> @llvm.wasm.pmin.v2f64(<2 x double> %a, <2 x double> %b)
  ret <2 x double> %v
}

; CHECK-LABEL: pmax_v2f64:
; CHECK-NEXT: .functype pmax_v2f64 (v128, v128) -> (v128){{$}}
; CHECK-NEXT: f64x2.pmax $push[[R:[0-9]+]]=, $0, $1{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
declare <2 x double> @llvm.wasm.pmax.v2f64(<2 x double>, <2 x double>)
define <2 x double> @pmax_v2f64(<2 x double> %a, <2 x double> %b) {
  %v = call <2 x double> @llvm.wasm.pmax.v2f64(<2 x double> %a, <2 x double> %b)
  ret <2 x double> %v
}

; CHECK-LABEL: ceil_v2f64:
; CHECK-NEXT: .functype ceil_v2f64 (v128) -> (v128){{$}}
; CHECK-NEXT: f64x2.ceil $push[[R:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
declare <2 x double> @llvm.wasm.ceil.v2f64(<2 x double>)
define <2 x double> @ceil_v2f64(<2 x double> %a) {
  %v = call <2 x double> @llvm.wasm.ceil.v2f64(<2 x double> %a)
  ret <2 x double> %v
}

; CHECK-LABEL: floor_v2f64:
; CHECK-NEXT: .functype floor_v2f64 (v128) -> (v128){{$}}
; CHECK-NEXT: f64x2.floor $push[[R:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
declare <2 x double> @llvm.wasm.floor.v2f64(<2 x double>)
define <2 x double> @floor_v2f64(<2 x double> %a) {
  %v = call <2 x double> @llvm.wasm.floor.v2f64(<2 x double> %a)
  ret <2 x double> %v
}

; CHECK-LABEL: trunc_v2f64:
; CHECK-NEXT: .functype trunc_v2f64 (v128) -> (v128){{$}}
; CHECK-NEXT: f64x2.trunc $push[[R:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
declare <2 x double> @llvm.wasm.trunc.v2f64(<2 x double>)
define <2 x double> @trunc_v2f64(<2 x double> %a) {
  %v = call <2 x double> @llvm.wasm.trunc.v2f64(<2 x double> %a)
  ret <2 x double> %v
}

; CHECK-LABEL: nearest_v2f64:
; CHECK-NEXT: .functype nearest_v2f64 (v128) -> (v128){{$}}
; CHECK-NEXT: f64x2.nearest $push[[R:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
declare <2 x double> @llvm.wasm.nearest.v2f64(<2 x double>)
define <2 x double> @nearest_v2f64(<2 x double> %a) {
  %v = call <2 x double> @llvm.wasm.nearest.v2f64(<2 x double> %a)
  ret <2 x double> %v
}

; CHECK-LABEL: convert_low_signed_v2f64:
; CHECK-NEXT: .functype convert_low_signed_v2f64 (v128) -> (v128){{$}}
; CHECK-NEXT: f64x2.convert_low_i32x4_s $push[[R:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
declare <2 x double> @llvm.wasm.convert.low.signed(<4 x i32>)
define <2 x double> @convert_low_signed_v2f64(<4 x i32> %a) {
  %v = call <2 x double> @llvm.wasm.convert.low.signed(<4 x i32> %a)
  ret <2 x double> %v
}

; CHECK-LABEL: convert_low_unsigned_v2f64:
; CHECK-NEXT: .functype convert_low_unsigned_v2f64 (v128) -> (v128){{$}}
; CHECK-NEXT: f64x2.convert_low_i32x4_u $push[[R:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
declare <2 x double> @llvm.wasm.convert.low.unsigned(<4 x i32>)
define <2 x double> @convert_low_unsigned_v2f64(<4 x i32> %a) {
  %v = call <2 x double> @llvm.wasm.convert.low.unsigned(<4 x i32> %a)
  ret <2 x double> %v
}

; CHECK-LABEL: promote_low_v2f64:
; CHECK-NEXT: .functype promote_low_v2f64 (v128) -> (v128){{$}}
; CHECK-NEXT: f64x2.promote_low_f32x4 $push[[R:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: return $pop[[R]]{{$}}
declare <2 x double> @llvm.wasm.promote.low(<4 x float>)
define <2 x double> @promote_low_v2f64(<4 x float> %a) {
  %v = call <2 x double> @llvm.wasm.promote.low(<4 x float> %a)
  ret <2 x double> %v
}
