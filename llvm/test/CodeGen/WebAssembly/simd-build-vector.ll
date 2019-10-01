; RUN: llc < %s -asm-verbose=false -verify-machineinstrs -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers -mattr=+unimplemented-simd128 | FileCheck %s

; Test that the logic to choose between v128.const vector
; initialization and splat vector initialization and to optimize the
; choice of splat value works correctly.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: same_const_one_replaced_i8x16:
; CHECK-NEXT:  .functype       same_const_one_replaced_i8x16 (i32) -> (v128)
; CHECK-NEXT:  i32.const       $push[[L0:[0-9]+]]=, 42
; CHECK-NEXT:  i16x8.splat     $push[[L1:[0-9]+]]=, $pop[[L0]]
; CHECK-NEXT:  i16x8.replace_lane      $push[[L2:[0-9]+]]=, $pop[[L1]], 5, $0
; CHECK-NEXT:  return          $pop[[L2]]
define <8 x i16> @same_const_one_replaced_i8x16(i16 %x) {
  %v = insertelement
    <8 x i16> <i16 42, i16 42, i16 42, i16 42, i16 42, i16 42, i16 42, i16 42>,
    i16 %x,
    i32 5
  ret <8 x i16> %v
}

; CHECK-LABEL: different_const_one_replaced_i8x16:
; CHECK-NEXT:  .functype       different_const_one_replaced_i8x16 (i32) -> (v128)
; CHECK-NEXT:  v128.const      $push[[L0:[0-9]+]]=, 1, -2, 3, -4, 5, 0, 7, -8
; CHECK-NEXT:  i16x8.replace_lane      $push[[L1:[0-9]+]]=, $pop[[L0]], 5, $0
; CHECK-NEXT:  return          $pop[[L1]]
define <8 x i16> @different_const_one_replaced_i8x16(i16 %x) {
  %v = insertelement
    <8 x i16> <i16 1, i16 -2, i16 3, i16 -4, i16 5, i16 -6, i16 7, i16 -8>,
    i16 %x,
    i32 5
  ret <8 x i16> %v
}

; CHECK-LABEL: same_const_one_replaced_f32x4:
; CHECK-NEXT:  .functype       same_const_one_replaced_f32x4 (f32) -> (v128)
; CHECK-NEXT:  f32.const       $push[[L0:[0-9]+]]=, 0x1.5p5
; CHECK-NEXT:  f32x4.splat     $push[[L1:[0-9]+]]=, $pop[[L0]]
; CHECK-NEXT:  f32x4.replace_lane      $push[[L2:[0-9]+]]=, $pop[[L1]], 2, $0
; CHECK-NEXT:  return          $pop[[L2]]
define <4 x float> @same_const_one_replaced_f32x4(float %x) {
  %v = insertelement
    <4 x float> <float 42., float 42., float 42., float 42.>,
    float %x,
    i32 2
  ret <4 x float> %v
}

; CHECK-LABEL: different_const_one_replaced_f32x4:
; CHECK-NEXT:  .functype       different_const_one_replaced_f32x4 (f32) -> (v128)
; CHECK-NEXT:  v128.const      $push[[L0:[0-9]+]]=, 0x1p0, 0x1p1, 0x0p0, 0x1p2
; CHECK-NEXT:  f32x4.replace_lane      $push[[L1:[0-9]+]]=, $pop[[L0]], 2, $0
; CHECK-NEXT:  return          $pop[[L1]]
define <4 x float> @different_const_one_replaced_f32x4(float %x) {
  %v = insertelement
    <4 x float> <float 1., float 2., float 3., float 4.>,
    float %x,
    i32 2
  ret <4 x float> %v
}

; CHECK-LABEL: splat_common_const_i32x4:
; CHECK-NEXT:  .functype       splat_common_const_i32x4 () -> (v128)
; CHECK-NEXT:  i32.const       $push[[L0:[0-9]+]]=, 3
; CHECK-NEXT:  i32x4.splat     $push[[L1:[0-9]+]]=, $pop[[L0]]
; CHECK-NEXT:  i32.const       $push[[L2:[0-9]+]]=, 1
; CHECK-NEXT:  i32x4.replace_lane      $push[[L3:[0-9]+]]=, $pop[[L1]], 3, $pop[[L2]]
; CHECK-NEXT:  return          $pop[[L3]]
define <4 x i32> @splat_common_const_i32x4() {
  ret <4 x i32> <i32 undef, i32 3, i32 3, i32 1>
}

; CHECK-LABEL: splat_common_arg_i16x8:
; CHECK-NEXT:  .functype       splat_common_arg_i16x8 (i32, i32, i32) -> (v128)
; CHECK-NEXT:  i16x8.splat     $push[[L0:[0-9]+]]=, $2
; CHECK-NEXT:  i16x8.replace_lane      $push[[L1:[0-9]+]]=, $pop[[L0]], 0, $1
; CHECK-NEXT:  i16x8.replace_lane      $push[[L2:[0-9]+]]=, $pop[[L1]], 2, $0
; CHECK-NEXT:  i16x8.replace_lane      $push[[L3:[0-9]+]]=, $pop[[L2]], 4, $1
; CHECK-NEXT:  i16x8.replace_lane      $push[[L4:[0-9]+]]=, $pop[[L3]], 7, $1
; CHECK-NEXT:  return          $pop[[L4]]
define <8 x i16> @splat_common_arg_i16x8(i16 %a, i16 %b, i16 %c) {
  %v0 = insertelement <8 x i16> undef, i16 %b, i32 0
  %v1 = insertelement <8 x i16> %v0, i16 %c, i32 1
  %v2 = insertelement <8 x i16> %v1, i16 %a, i32 2
  %v3 = insertelement <8 x i16> %v2, i16 %c, i32 3
  %v4 = insertelement <8 x i16> %v3, i16 %b, i32 4
  %v5 = insertelement <8 x i16> %v4, i16 %c, i32 5
  %v6 = insertelement <8 x i16> %v5, i16 %c, i32 6
  %v7 = insertelement <8 x i16> %v6, i16 %b, i32 7
  ret <8 x i16> %v7
}

; CHECK-LABEL: undef_const_insert_f32x4:
; CHECK-NEXT:  .functype       undef_const_insert_f32x4 () -> (v128)
; CHECK-NEXT:  f32.const       $push[[L0:[0-9]+]]=, 0x1.5p5
; CHECK-NEXT:  f32x4.splat     $push[[L1:[0-9]+]]=, $pop[[L0]]
; CHECK-NEXT:  return          $pop[[L1]]
define <4 x float> @undef_const_insert_f32x4() {
  %v = insertelement <4 x float> undef, float 42., i32 1
  ret <4 x float> %v
}

; CHECK-LABEL: undef_arg_insert_i32x4:
; CHECK-NEXT:  .functype       undef_arg_insert_i32x4 (i32) -> (v128)
; CHECK-NEXT:  i32x4.splat     $push[[L0:[0-9]+]]=, $0
; CHECK-NEXT:  return          $pop[[L0]]
define <4 x i32> @undef_arg_insert_i32x4(i32 %x) {
  %v = insertelement <4 x i32> undef, i32 %x, i32 3
  ret <4 x i32> %v
}

; CHECK-LABEL: all_undef_i8x16:
; CHECK-NEXT:  .functype       all_undef_i8x16 () -> (v128)
; CHECK-NEXT:  return          $0
define <16 x i8> @all_undef_i8x16() {
  %v = insertelement <16 x i8> undef, i8 undef, i32 4
  ret <16 x i8> %v
}

; CHECK-LABEL: all_undef_f64x2:
; CHECK-NEXT:  .functype       all_undef_f64x2 () -> (v128)
; CHECK-NEXT:  return          $0
define <2 x double> @all_undef_f64x2() {
  ret <2 x double> undef
}
