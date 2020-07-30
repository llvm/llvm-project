; RUN: llc < %s -asm-verbose=false -verify-machineinstrs -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers -mattr=+simd128 | FileCheck %s

; Test that SIMD shifts can be lowered correctly even with shift
; values that are more complex than plain splats.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

;; TODO: Optimize this further by scalarizing the add

; CHECK-LABEL: shl_add:
; CHECK-NEXT: .functype shl_add (v128, i32, i32) -> (v128)
; CHECK-NEXT: i8x16.splat $push1=, $1
; CHECK-NEXT: i8x16.splat $push0=, $2
; CHECK-NEXT: i8x16.add $push2=, $pop1, $pop0
; CHECK-NEXT: i8x16.extract_lane_u $push3=, $pop2, 0
; CHECK-NEXT: i8x16.shl $push4=, $0, $pop3
; CHECK-NEXT: return $pop4
define <16 x i8> @shl_add(<16 x i8> %v, i8 %a, i8 %b) {
  %t1 = insertelement <16 x i8> undef, i8 %a, i32 0
  %va = shufflevector <16 x i8> %t1, <16 x i8> undef, <16 x i32> zeroinitializer
  %t2 = insertelement <16 x i8> undef, i8 %b, i32 0
  %vb = shufflevector <16 x i8> %t2, <16 x i8> undef, <16 x i32> zeroinitializer
  %shift = add <16 x i8> %va, %vb
  %r = shl <16 x i8> %v, %shift
  ret <16 x i8> %r
}

; CHECK-LABEL: shl_abs:
; CHECK-NEXT: .functype shl_abs (v128, i32) -> (v128)
; CHECK-NEXT: i8x16.extract_lane_u $push8=, $0, 0
; CHECK-NEXT: i8x16.splat $push0=, $1
; CHECK-NEXT: i8x16.abs $push98=, $pop0
; CHECK-NEXT: local.tee $push97=, $2=, $pop98
; CHECK-NEXT: i8x16.extract_lane_u $push6=, $pop97, 0
; CHECK-NEXT: i32.const $push2=, 7
; CHECK-NEXT: i32.and $push7=, $pop6, $pop2
; CHECK-NEXT: i32.shl $push9=, $pop8, $pop7
; CHECK-NEXT: i8x16.splat $push10=, $pop9
; CHECK-NEXT: i8x16.extract_lane_u $push4=, $0, 1
; CHECK-NEXT: i8x16.extract_lane_u $push1=, $2, 1
; CHECK-NEXT: i32.const $push96=, 7
; CHECK-NEXT: i32.and $push3=, $pop1, $pop96
; CHECK-NEXT: i32.shl $push5=, $pop4, $pop3
; CHECK-NEXT: i8x16.replace_lane $push11=, $pop10, 1, $pop5
; ...
; CHECK:      i8x16.extract_lane_u $push79=, $0, 15
; CHECK-NEXT: i8x16.extract_lane_u $push77=, $2, 15
; CHECK-NEXT: i32.const $push82=, 7
; CHECK-NEXT: i32.and $push78=, $pop77, $pop82
; CHECK-NEXT: i32.shl $push80=, $pop79, $pop78
; CHECK-NEXT: i8x16.replace_lane $push81=, $pop76, 15, $pop80
; CHECK-NEXT: return $pop81
define <16 x i8> @shl_abs(<16 x i8> %v, i8 %a) {
  %t1 = insertelement <16 x i8> undef, i8 %a, i32 0
  %va = shufflevector <16 x i8> %t1, <16 x i8> undef, <16 x i32> zeroinitializer
  %nva = sub <16 x i8> zeroinitializer, %va
  %c = icmp sgt <16 x i8> %va, zeroinitializer
  %shift = select <16 x i1> %c, <16 x i8> %va, <16 x i8> %nva
  %r = shl <16 x i8> %v, %shift
  ret <16 x i8> %r
}

; CHECK-LABEL: shl_abs_add:
; CHECK-NEXT: .functype shl_abs_add (v128, i32, i32) -> (v128)
; CHECK-NEXT: i8x16.extract_lane_u $push11=, $0, 0
; CHECK-NEXT: i8x16.splat $push1=, $1
; CHECK-NEXT: i8x16.splat $push0=, $2
; CHECK-NEXT: i8x16.add $push2=, $pop1, $pop0
; CHECK-NEXT: v8x16.shuffle $push3=, $pop2, $0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
; CHECK-NEXT: i8x16.abs $push101=, $pop3
; CHECK-NEXT: local.tee $push100=, $3=, $pop101
; CHECK-NEXT: i8x16.extract_lane_u $push9=, $pop100, 0
; CHECK-NEXT: i32.const $push5=, 7
; CHECK-NEXT: i32.and $push10=, $pop9, $pop5
; CHECK-NEXT: i32.shl $push12=, $pop11, $pop10
; CHECK-NEXT: i8x16.splat $push13=, $pop12
; CHECK-NEXT: i8x16.extract_lane_u $push7=, $0, 1
; CHECK-NEXT: i8x16.extract_lane_u $push4=, $3, 1
; CHECK-NEXT: i32.const $push99=, 7
; CHECK-NEXT: i32.and $push6=, $pop4, $pop99
; CHECK-NEXT: i32.shl $push8=, $pop7, $pop6
; CHECK-NEXT: i8x16.replace_lane $push14=, $pop13, 1, $pop8
; ...
; CHECK:      i8x16.extract_lane_u $push82=, $0, 15
; CHECK-NEXT: i8x16.extract_lane_u $push80=, $3, 15
; CHECK-NEXT: i32.const $push85=, 7
; CHECK-NEXT: i32.and $push81=, $pop80, $pop85
; CHECK-NEXT: i32.shl $push83=, $pop82, $pop81
; CHECK-NEXT: i8x16.replace_lane $push84=, $pop79, 15, $pop83
; CHECK-NEXT: return $pop84
define <16 x i8> @shl_abs_add(<16 x i8> %v, i8 %a, i8 %b) {
  %t1 = insertelement <16 x i8> undef, i8 %a, i32 0
  %va = shufflevector <16 x i8> %t1, <16 x i8> undef, <16 x i32> zeroinitializer
  %t2 = insertelement <16 x i8> undef, i8 %b, i32 0
  %vb = shufflevector <16 x i8> %t2, <16 x i8> undef, <16 x i32> zeroinitializer
  %vadd = add <16 x i8> %va, %vb
  %nvadd = sub <16 x i8> zeroinitializer, %vadd
  %c = icmp sgt <16 x i8> %vadd, zeroinitializer
  %shift = select <16 x i1> %c, <16 x i8> %vadd, <16 x i8> %nvadd
  %r = shl <16 x i8> %v, %shift
  ret <16 x i8> %r
}
