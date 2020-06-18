; RUN: llc < %s -asm-verbose=false -verify-machineinstrs -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers -mattr=+unimplemented-simd128 | FileCheck %s --check-prefixes CHECK,SIMD128,SIMD128-SLOW

;; Test that the custom shift unrolling works correctly in cases that
;; cause assertion failures due to illegal types when using
;; DAG.UnrollVectorOp. Regression test for PR45178.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: shl_v16i8:
; CHECK-NEXT: .functype       shl_v16i8 (v128) -> (v128)
; CHECK-NEXT: i8x16.extract_lane_u    $push0=, $0, 0
; CHECK-NEXT: i32.const       $push1=, 3
; CHECK-NEXT: i32.shl         $push2=, $pop0, $pop1
; CHECK-NEXT: i8x16.splat     $push3=, $pop2
; CHECK-NEXT: i8x16.extract_lane_u    $push4=, $0, 1
; CHECK-NEXT: i8x16.replace_lane      $push5=, $pop3, 1, $pop4
; ...
; CHECK:      i8x16.extract_lane_u    $push32=, $0, 15
; CHECK-NEXT: i8x16.replace_lane      $push33=, $pop31, 15, $pop32
; CHECK-NEXT: v8x16.shuffle   $push34=, $pop33, $0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
; CHECK-NEXT: return  $pop34
define <16 x i8> @shl_v16i8(<16 x i8> %in) {
  %out = shl <16 x i8> %in,
    <i8 3, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0,
     i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>
  %ret = shufflevector <16 x i8> %out, <16 x i8> undef, <16 x i32> zeroinitializer
  ret <16 x i8> %ret
}

; CHECK-LABEL: shr_s_v16i8:
; CHECK-NEXT: functype       shr_s_v16i8 (v128) -> (v128)
; CHECK-NEXT: i8x16.extract_lane_s    $push0=, $0, 0
; CHECK-NEXT: i32.const       $push1=, 3
; CHECK-NEXT: i32.shr_s       $push2=, $pop0, $pop1
; CHECK-NEXT: i8x16.splat     $push3=, $pop2
; CHECK-NEXT: i8x16.extract_lane_s    $push4=, $0, 1
; CHECK-NEXT: i8x16.replace_lane      $push5=, $pop3, 1, $pop4
; ...
; CHECK:      i8x16.extract_lane_s    $push32=, $0, 15
; CHECK-NEXT: i8x16.replace_lane      $push33=, $pop31, 15, $pop32
; CHECK-NEXT: v8x16.shuffle   $push34=, $pop33, $0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
; CHECK-NEXT: return  $pop34
define <16 x i8> @shr_s_v16i8(<16 x i8> %in) {
  %out = ashr <16 x i8> %in,
    <i8 3, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0,
     i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>
  %ret = shufflevector <16 x i8> %out, <16 x i8> undef, <16 x i32> zeroinitializer
  ret <16 x i8> %ret
}

; CHECK-LABEL: shr_u_v16i8:
; CHECK-NEXT: functype       shr_u_v16i8 (v128) -> (v128)
; CHECK-NEXT: i8x16.extract_lane_u    $push0=, $0, 0
; CHECK-NEXT: i32.const       $push1=, 3
; CHECK-NEXT: i32.shr_u       $push2=, $pop0, $pop1
; CHECK-NEXT: i8x16.splat     $push3=, $pop2
; CHECK-NEXT: i8x16.extract_lane_u    $push4=, $0, 1
; CHECK-NEXT: i8x16.replace_lane      $push5=, $pop3, 1, $pop4
; ...
; CHECK:      i8x16.extract_lane_u    $push32=, $0, 15
; CHECK-NEXT: i8x16.replace_lane      $push33=, $pop31, 15, $pop32
; CHECK-NEXT: v8x16.shuffle   $push34=, $pop33, $0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
; CHECK-NEXT: return  $pop34
define <16 x i8> @shr_u_v16i8(<16 x i8> %in) {
  %out = lshr <16 x i8> %in,
    <i8 3, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0,
     i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>
  %ret = shufflevector <16 x i8> %out, <16 x i8> undef, <16 x i32> zeroinitializer
  ret <16 x i8> %ret
}

; CHECK-LABEL: shl_v8i16:
; CHECK-NEXT: functype       shl_v8i16 (v128) -> (v128)
; CHECK-NEXT: i16x8.extract_lane_u    $push0=, $0, 0
; CHECK-NEXT: i32.const       $push1=, 9
; CHECK-NEXT: i32.shl         $push2=, $pop0, $pop1
; CHECK-NEXT: i16x8.splat     $push3=, $pop2
; CHECK-NEXT: i16x8.extract_lane_u    $push4=, $0, 1
; CHECK-NEXT: i16x8.replace_lane      $push5=, $pop3, 1, $pop4
; ...
; CHECK:      i16x8.extract_lane_u    $push16=, $0, 7
; CHECK-NEXT: i16x8.replace_lane      $push17=, $pop15, 7, $pop16
; CHECK-NEXT: v8x16.shuffle   $push18=, $pop17, $0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1
; CHECK-NEXT: return  $pop18
define <8 x i16> @shl_v8i16(<8 x i16> %in) {
  %out = shl <8 x i16> %in, <i16 9, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>
  %ret = shufflevector <8 x i16> %out, <8 x i16> undef, <8 x i32> zeroinitializer
  ret <8 x i16> %ret
}

; CHECK-LABEL: shr_s_v8i16:
; CHECK-NEXT: functype       shr_s_v8i16 (v128) -> (v128)
; CHECK-NEXT: i16x8.extract_lane_s    $push0=, $0, 0
; CHECK-NEXT: i32.const       $push1=, 9
; CHECK-NEXT: i32.shr_s       $push2=, $pop0, $pop1
; CHECK-NEXT: i16x8.splat     $push3=, $pop2
; CHECK-NEXT: i16x8.extract_lane_s    $push4=, $0, 1
; CHECK-NEXT: i16x8.replace_lane      $push5=, $pop3, 1, $pop4
; ...
; CHECK:      i16x8.extract_lane_s    $push16=, $0, 7
; CHECK-NEXT: i16x8.replace_lane      $push17=, $pop15, 7, $pop16
; CHECK-NEXT: v8x16.shuffle   $push18=, $pop17, $0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1
; CHECK-NEXT: return  $pop18
define <8 x i16> @shr_s_v8i16(<8 x i16> %in) {
  %out = ashr <8 x i16> %in, <i16 9, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>
  %ret = shufflevector <8 x i16> %out, <8 x i16> undef, <8 x i32> zeroinitializer
  ret <8 x i16> %ret
}

; CHECK-LABEL: shr_u_v8i16:
; CHECK-NEXT: functype       shr_u_v8i16 (v128) -> (v128)
; CHECK-NEXT: i16x8.extract_lane_u    $push0=, $0, 0
; CHECK-NEXT: i32.const       $push1=, 9
; CHECK-NEXT: i32.shr_u       $push2=, $pop0, $pop1
; CHECK-NEXT: i16x8.splat     $push3=, $pop2
; CHECK-NEXT: i16x8.extract_lane_u    $push4=, $0, 1
; CHECK-NEXT: i16x8.replace_lane      $push5=, $pop3, 1, $pop4
; ...
; CHECK:      i16x8.extract_lane_u    $push16=, $0, 7
; CHECK-NEXT: i16x8.replace_lane      $push17=, $pop15, 7, $pop16
; CHECK-NEXT: v8x16.shuffle   $push18=, $pop17, $0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1
; CHECK-NEXT: return  $pop18
define <8 x i16> @shr_u_v8i16(<8 x i16> %in) {
  %out = lshr <8 x i16> %in, <i16 9, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>
  %ret = shufflevector <8 x i16> %out, <8 x i16> undef, <8 x i32> zeroinitializer
  ret <8 x i16> %ret
}
