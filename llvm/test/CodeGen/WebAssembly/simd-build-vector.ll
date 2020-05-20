; RUN: llc < %s -asm-verbose=false -verify-machineinstrs -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers -mattr=+unimplemented-simd128 | FileCheck %s

; Test that the logic to choose between v128.const vector
; initialization and splat vector initialization and to optimize the
; choice of splat value works correctly.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: same_const_one_replaced_i16x8:
; CHECK-NEXT:  .functype       same_const_one_replaced_i16x8 (i32) -> (v128)
; CHECK-NEXT:  v128.const      $push[[L0:[0-9]+]]=, 42, 42, 42, 42, 42, 0, 42, 42
; CHECK-NEXT:  i16x8.replace_lane      $push[[L1:[0-9]+]]=, $pop[[L0]], 5, $0
; CHECK-NEXT:  return          $pop[[L1]]
define <8 x i16> @same_const_one_replaced_i16x8(i16 %x) {
  %v = insertelement
    <8 x i16> <i16 42, i16 42, i16 42, i16 42, i16 42, i16 42, i16 42, i16 42>,
    i16 %x,
    i32 5
  ret <8 x i16> %v
}

; CHECK-LABEL: different_const_one_replaced_i16x8:
; CHECK-NEXT:  .functype       different_const_one_replaced_i16x8 (i32) -> (v128)
; CHECK-NEXT:  v128.const      $push[[L0:[0-9]+]]=, 1, -2, 3, -4, 5, 0, 7, -8
; CHECK-NEXT:  i16x8.replace_lane      $push[[L1:[0-9]+]]=, $pop[[L0]], 5, $0
; CHECK-NEXT:  return          $pop[[L1]]
define <8 x i16> @different_const_one_replaced_i16x8(i16 %x) {
  %v = insertelement
    <8 x i16> <i16 1, i16 -2, i16 3, i16 -4, i16 5, i16 -6, i16 7, i16 -8>,
    i16 %x,
    i32 5
  ret <8 x i16> %v
}

; CHECK-LABEL: same_const_one_replaced_f32x4:
; CHECK-NEXT:  .functype       same_const_one_replaced_f32x4 (f32) -> (v128)
; CHECK-NEXT:  v128.const      $push[[L0:[0-9]+]]=, 0x1.5p5, 0x1.5p5, 0x0p0, 0x1.5p5
; CHECK-NEXT:  f32x4.replace_lane      $push[[L1:[0-9]+]]=, $pop[[L0]], 2, $0
; CHECK-NEXT:  return          $pop[[L1]]
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
; CHECK-NEXT:  v128.const      $push[[L0:[0-9]+]]=, 0, 3, 3, 1
; CHECK-NEXT:  return          $pop[[L0]]
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

; CHECK-LABEL: swizzle_one_i8x16:
; CHECK-NEXT:  .functype       swizzle_one_i8x16 (v128, v128) -> (v128)
; CHECK-NEXT:  v8x16.swizzle   $push[[L0:[0-9]+]]=, $0, $1
; CHECK-NEXT:  return          $pop[[L0]]
define <16 x i8> @swizzle_one_i8x16(<16 x i8> %src, <16 x i8> %mask) {
  %m0 = extractelement <16 x i8> %mask, i32 0
  %s0 = extractelement <16 x i8> %src, i8 %m0
  %v0 = insertelement <16 x i8> undef, i8 %s0, i32 0
  ret <16 x i8> %v0
}

; CHECK-LABEL: swizzle_all_i8x16:
; CHECK-NEXT:  .functype       swizzle_all_i8x16 (v128, v128) -> (v128)
; CHECK-NEXT:  v8x16.swizzle   $push[[L0:[0-9]+]]=, $0, $1
; CHECK-NEXT:  return          $pop[[L0]]
define <16 x i8> @swizzle_all_i8x16(<16 x i8> %src, <16 x i8> %mask) {
  %m0 = extractelement <16 x i8> %mask, i32 0
  %s0 = extractelement <16 x i8> %src, i8 %m0
  %v0 = insertelement <16 x i8> undef, i8 %s0, i32 0
  %m1 = extractelement <16 x i8> %mask, i32 1
  %s1 = extractelement <16 x i8> %src, i8 %m1
  %v1 = insertelement <16 x i8> %v0, i8 %s1, i32 1
  %m2 = extractelement <16 x i8> %mask, i32 2
  %s2 = extractelement <16 x i8> %src, i8 %m2
  %v2 = insertelement <16 x i8> %v1, i8 %s2, i32 2
  %m3 = extractelement <16 x i8> %mask, i32 3
  %s3 = extractelement <16 x i8> %src, i8 %m3
  %v3 = insertelement <16 x i8> %v2, i8 %s3, i32 3
  %m4 = extractelement <16 x i8> %mask, i32 4
  %s4 = extractelement <16 x i8> %src, i8 %m4
  %v4 = insertelement <16 x i8> %v3, i8 %s4, i32 4
  %m5 = extractelement <16 x i8> %mask, i32 5
  %s5 = extractelement <16 x i8> %src, i8 %m5
  %v5 = insertelement <16 x i8> %v4, i8 %s5, i32 5
  %m6 = extractelement <16 x i8> %mask, i32 6
  %s6 = extractelement <16 x i8> %src, i8 %m6
  %v6 = insertelement <16 x i8> %v5, i8 %s6, i32 6
  %m7 = extractelement <16 x i8> %mask, i32 7
  %s7 = extractelement <16 x i8> %src, i8 %m7
  %v7 = insertelement <16 x i8> %v6, i8 %s7, i32 7
  %m8 = extractelement <16 x i8> %mask, i32 8
  %s8 = extractelement <16 x i8> %src, i8 %m8
  %v8 = insertelement <16 x i8> %v7, i8 %s8, i32 8
  %m9 = extractelement <16 x i8> %mask, i32 9
  %s9 = extractelement <16 x i8> %src, i8 %m9
  %v9 = insertelement <16 x i8> %v8, i8 %s9, i32 9
  %m10 = extractelement <16 x i8> %mask, i32 10
  %s10 = extractelement <16 x i8> %src, i8 %m10
  %v10 = insertelement <16 x i8> %v9, i8 %s10, i32 10
  %m11 = extractelement <16 x i8> %mask, i32 11
  %s11 = extractelement <16 x i8> %src, i8 %m11
  %v11 = insertelement <16 x i8> %v10, i8 %s11, i32 11
  %m12 = extractelement <16 x i8> %mask, i32 12
  %s12 = extractelement <16 x i8> %src, i8 %m12
  %v12 = insertelement <16 x i8> %v11, i8 %s12, i32 12
  %m13 = extractelement <16 x i8> %mask, i32 13
  %s13 = extractelement <16 x i8> %src, i8 %m13
  %v13 = insertelement <16 x i8> %v12, i8 %s13, i32 13
  %m14 = extractelement <16 x i8> %mask, i32 14
  %s14 = extractelement <16 x i8> %src, i8 %m14
  %v14 = insertelement <16 x i8> %v13, i8 %s14, i32 14
  %m15 = extractelement <16 x i8> %mask, i32 15
  %s15 = extractelement <16 x i8> %src, i8 %m15
  %v15 = insertelement <16 x i8> %v14, i8 %s15, i32 15
  ret <16 x i8> %v15
}

; CHECK-LABEL: swizzle_one_i16x8:
; CHECK-NEXT:  .functype       swizzle_one_i16x8 (v128, v128) -> (v128)
; CHECK-NOT:    swizzle
; CHECK:        return
define <8 x i16> @swizzle_one_i16x8(<8 x i16> %src, <8 x i16> %mask) {
  %m0 = extractelement <8 x i16> %mask, i32 0
  %s0 = extractelement <8 x i16> %src, i16 %m0
  %v0 = insertelement <8 x i16> undef, i16 %s0, i32 0
  ret <8 x i16> %v0
}

; CHECK-LABEL: mashup_swizzle_i8x16:
; CHECK-NEXT:  .functype       mashup_swizzle_i8x16 (v128, v128, i32) -> (v128)
; CHECK-NEXT:  v8x16.swizzle   $push[[L0:[0-9]+]]=, $0, $1
; CHECK:       i8x16.replace_lane
; CHECK:       i8x16.replace_lane
; CHECK:       i8x16.replace_lane
; CHECK:       i8x16.replace_lane
; CHECK:       return
define <16 x i8> @mashup_swizzle_i8x16(<16 x i8> %src, <16 x i8> %mask, i8 %splatted) {
  ; swizzle 0
  %m0 = extractelement <16 x i8> %mask, i32 0
  %s0 = extractelement <16 x i8> %src, i8 %m0
  %v0 = insertelement <16 x i8> undef, i8 %s0, i32 0
  ; swizzle 7
  %m1 = extractelement <16 x i8> %mask, i32 7
  %s1 = extractelement <16 x i8> %src, i8 %m1
  %v1 = insertelement <16 x i8> %v0, i8 %s1, i32 7
  ; splat 3
  %v2 = insertelement <16 x i8> %v1, i8 %splatted, i32 3
  ; splat 12
  %v3 = insertelement <16 x i8> %v2, i8 %splatted, i32 12
  ; const 4
  %v4 = insertelement <16 x i8> %v3, i8 42, i32 4
  ; const 14
  %v5 = insertelement <16 x i8> %v4, i8 42, i32 14
  ret <16 x i8> %v5
}

; CHECK-LABEL: mashup_const_i8x16:
; CHECK-NEXT:  .functype       mashup_const_i8x16 (v128, v128, i32) -> (v128)
; CHECK:       v128.const      $push[[L0:[0-9]+]]=, 0, 0, 0, 0, 42, 0, 0, 0, 0, 0, 0, 0, 0, 0, 42, 0
; CHECK:       i8x16.replace_lane
; CHECK:       i8x16.replace_lane
; CHECK:       i8x16.replace_lane
; CHECK:       return
define <16 x i8> @mashup_const_i8x16(<16 x i8> %src, <16 x i8> %mask, i8 %splatted) {
  ; swizzle 0
  %m0 = extractelement <16 x i8> %mask, i32 0
  %s0 = extractelement <16 x i8> %src, i8 %m0
  %v0 = insertelement <16 x i8> undef, i8 %s0, i32 0
  ; splat 3
  %v1 = insertelement <16 x i8> %v0, i8 %splatted, i32 3
  ; splat 12
  %v2 = insertelement <16 x i8> %v1, i8 %splatted, i32 12
  ; const 4
  %v3 = insertelement <16 x i8> %v2, i8 42, i32 4
  ; const 14
  %v4 = insertelement <16 x i8> %v3, i8 42, i32 14
  ret <16 x i8> %v4
}

; CHECK-LABEL: mashup_splat_i8x16:
; CHECK-NEXT:  .functype       mashup_splat_i8x16 (v128, v128, i32) -> (v128)
; CHECK:       i8x16.splat     $push[[L0:[0-9]+]]=, $2
; CHECK:       i8x16.replace_lane
; CHECK:       i8x16.replace_lane
; CHECK:       return
define <16 x i8> @mashup_splat_i8x16(<16 x i8> %src, <16 x i8> %mask, i8 %splatted) {
  ; swizzle 0
  %m0 = extractelement <16 x i8> %mask, i32 0
  %s0 = extractelement <16 x i8> %src, i8 %m0
  %v0 = insertelement <16 x i8> undef, i8 %s0, i32 0
  ; splat 3
  %v1 = insertelement <16 x i8> %v0, i8 %splatted, i32 3
  ; splat 12
  %v2 = insertelement <16 x i8> %v1, i8 %splatted, i32 12
  ; const 4
  %v3 = insertelement <16 x i8> %v2, i8 42, i32 4
  ret <16 x i8> %v3
}

; CHECK-LABEL: undef_const_insert_f32x4:
; CHECK-NEXT:  .functype       undef_const_insert_f32x4 () -> (v128)
; CHECK-NEXT:  v128.const      $push[[L0:[0-9]+]]=, 0x0p0, 0x1.5p5, 0x0p0, 0x0p0
; CHECK-NEXT:  return          $pop[[L0]]
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
