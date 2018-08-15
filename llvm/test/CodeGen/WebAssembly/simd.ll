; RUN: llc < %s -disable-wasm-fallthrough-return-opt -disable-wasm-explicit-locals -wasm-enable-unimplemented-simd -mattr=+simd128,+sign-ext --show-mc-encoding | FileCheck %s --check-prefixes CHECK,SIMD128
; RUN: llc < %s -disable-wasm-fallthrough-return-opt -disable-wasm-explicit-locals -mattr=+simd128,+sign-ext --show-mc-encoding | FileCheck %s --check-prefixes CHECK,SIMD128-VM
; RUN: llc < %s -disable-wasm-fallthrough-return-opt -disable-wasm-explicit-locals -mattr=-simd128,+sign-ext --show-mc-encoding | FileCheck %s --check-prefixes CHECK,NO-SIMD128

; Test that basic SIMD128 vector manipulation operations assemble as expected.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; ==============================================================================
; 16 x i8
; ==============================================================================
; CHECK-LABEL: splat_v16i8:
; NO-SIMD128-NOT: i8x16
; SIMD128: .param i32{{$}}
; SIMD128: .result v128{{$}}
; SIMD128: i8x16.splat $push0=, $0 # encoding: [0xfd,0x03]{{$}}
; SIMD128: return $pop0 # encoding: [0x0f]{{$}}
define <16 x i8> @splat_v16i8(i8 %x) {
  %v = insertelement <16 x i8> undef, i8 %x, i32 0
  %res = shufflevector <16 x i8> %v, <16 x i8> undef,
    <16 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0,
                i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <16 x i8> %res
}

; CHECK-LABEL: extract_v16i8_s:
; NO-SIMD128-NOT: i8x16
; SIMD128: .param v128{{$}}
; SIMD128: .result i32{{$}}
; SIMD128: i8x16.extract_lane_s $push0=, $0, 13 # encoding: [0xfd,0x09,0x0d]{{$}}
; SIMD128: return $pop0 #
define i32 @extract_v16i8_s(<16 x i8> %v) {
  %elem = extractelement <16 x i8> %v, i8 13
  %a = sext i8 %elem to i32
  ret i32 %a
}

; CHECK-LABEL: extract_v16i8_u:
; NO-SIMD128-NOT: i8x16
; SIMD128: .param v128{{$}}
; SIMD128: .result i32{{$}}
; SIMD128: i8x16.extract_lane_u $push0=, $0, 13 # encoding: [0xfd,0x0a,0x0d]{{$}}
; SIMD128: return $pop0 #
define i32 @extract_v16i8_u(<16 x i8> %v) {
  %elem = extractelement <16 x i8> %v, i8 13
  %a = zext i8 %elem to i32
  ret i32 %a
}

; CHECK-LABEL: extract_v16i8:
; NO-SIMD128-NOT: i8x16
; SIMD128: .param v128{{$}}
; SIMD128: .result i32{{$}}
; SIMD128: i8x16.extract_lane_u $push0=, $0, 13 # encoding: [0xfd,0x0a,0x0d]{{$}}
; SIMD128: return $pop0 #
define i8 @extract_v16i8(<16 x i8> %v) {
  %elem = extractelement <16 x i8> %v, i8 13
  ret i8 %elem
}

; ==============================================================================
; 8 x i16
; ==============================================================================
; CHECK-LABEL: splat_v8i16:
; NO-SIMD128-NOT: i16x8
; SIMD128: .param i32{{$}}
; SIMD128: .result v128{{$}}
; SIMD128: i16x8.splat $push0=, $0 # encoding: [0xfd,0x04]{{$}}
; SIMD128: return $pop0 # encoding: [0x0f]{{$}}
define <8 x i16> @splat_v8i16(i16 %x) {
  %v = insertelement <8 x i16> undef, i16 %x, i32 0
  %res = shufflevector <8 x i16> %v, <8 x i16> undef,
    <8 x i32> <i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0>
  ret <8 x i16> %res
}

; CHECK-LABEL: extract_v8i16_s:
; NO-SIMD128-NOT: i16x8
; SIMD128: .param v128{{$}}
; SIMD128: .result i32{{$}}
; SIMD128: i16x8.extract_lane_s $push0=, $0, 5 # encoding: [0xfd,0x0b,0x05]{{$}}
; SIMD128: return $pop0 #
define i32 @extract_v8i16_s(<8 x i16> %v) {
  %elem = extractelement <8 x i16> %v, i16 5
  %a = sext i16 %elem to i32
  ret i32 %a
}

; CHECK-LABEL: extract_v8i16_u:
; NO-SIMD128-NOT: i16x8
; SIMD128: .param v128{{$}}
; SIMD128: .result i32{{$}}
; SIMD128: i16x8.extract_lane_u $push0=, $0, 5 # encoding: [0xfd,0x0c,0x05]{{$}}
; SIMD128: return $pop0 #
define i32 @extract_v8i16_u(<8 x i16> %v) {
  %elem = extractelement <8 x i16> %v, i16 5
  %a = zext i16 %elem to i32
  ret i32 %a
}

; CHECK-LABEL: extract_v8i16:
; NO-SIMD128-NOT: i16x8
; SIMD128: .param v128{{$}}
; SIMD128: .result i32{{$}}
; SIMD128: i16x8.extract_lane_u $push0=, $0, 5 # encoding: [0xfd,0x0c,0x05]{{$}}
; SIMD128: return $pop0 #
define i16 @extract_v8i16(<8 x i16> %v) {
  %elem = extractelement <8 x i16> %v, i16 5
  ret i16 %elem
}

; ==============================================================================
; 4 x i32
; ==============================================================================
; CHECK-LABEL: splat_v4i32:
; NO-SIMD128-NOT: i32x4
; SIMD128: .param i32{{$}}
; SIMD128: .result v128{{$}}
; SIMD128: i32x4.splat $push0=, $0 # encoding: [0xfd,0x05]{{$}}
; SIMD128: return $pop0 # encoding: [0x0f]{{$}}
define <4 x i32> @splat_v4i32(i32 %x) {
  %v = insertelement <4 x i32> undef, i32 %x, i32 0
  %res = shufflevector <4 x i32> %v, <4 x i32> undef,
    <4 x i32> <i32 0, i32 0, i32 0, i32 0>
  ret <4 x i32> %res
}

; CHECK-LABEL: extract_v4i32:
; NO-SIMD128-NOT: i32x4
; SIMD128: .param v128{{$}}
; SIMD128: .result i32{{$}}
; SIMD128: i32x4.extract_lane $push0=, $0, 3 # encoding: [0xfd,0x0d,0x03]{{$}}
; SIMD128: return $pop0 #
define i32 @extract_v4i32(<4 x i32> %v) {
  %elem = extractelement <4 x i32> %v, i32 3
  ret i32 %elem
}

; ==============================================================================
; 2 x i64
; ==============================================================================
; CHECK-LABEL: splat_v2i64:
; NO-SIMD128-NOT: i64x2
; SIMD128-VM-NOT: i64x2
; SIMD128: .param i64{{$}}
; SIMD128: .result v128{{$}}
; SIMD128: i64x2.splat $push0=, $0 # encoding: [0xfd,0x06]{{$}}
; SIMD128: return $pop0 # encoding: [0x0f]{{$}}
define <2 x i64> @splat_v2i64(i64 %x) {
  %t1 = insertelement <2 x i64> zeroinitializer, i64 %x, i32 0
  %res = insertelement <2 x i64> %t1, i64 %x, i32 1
  ret <2 x i64> %res
}

; CHECK-LABEL: extract_v2i64:
; NO-SIMD128-NOT: i64x2
; SIMD128-VM-NOT: i64x2
; SIMD128: .param v128{{$}}
; SIMD128: .result i64{{$}}
; SIMD128: i64x2.extract_lane $push0=, $0, 1 # encoding: [0xfd,0x0e,0x01]{{$}}
; SIMD128: return $pop0 #
define i64 @extract_v2i64(<2 x i64> %v) {
  %elem = extractelement <2 x i64> %v, i64 1
  ret i64 %elem
}

; ==============================================================================
; 4 x f32
; ==============================================================================
; CHECK-LABEL: splat_v4f32:
; NO-SIMD128-NOT: f32x4
; SIMD128: .param f32{{$}}
; SIMD128: .result v128{{$}}
; SIMD128: f32x4.splat $push0=, $0 # encoding: [0xfd,0x07]{{$}}
; SIMD128: return $pop0 # encoding: [0x0f]{{$}}
define <4 x float> @splat_v4f32(float %x) {
  %v = insertelement <4 x float> undef, float %x, i32 0
  %res = shufflevector <4 x float> %v, <4 x float> undef,
    <4 x i32> <i32 0, i32 0, i32 0, i32 0>
  ret <4 x float> %res
}

; CHECK-LABEL: extract_v4f32:
; NO-SIMD128-NOT: f32x4
; SIMD128: .param v128{{$}}
; SIMD128: .result f32{{$}}
; SIMD128: f32x4.extract_lane $push0=, $0, 3 # encoding: [0xfd,0x0f,0x03]{{$}}
; SIMD128: return $pop0 #
define float @extract_v4f32(<4 x float> %v) {
  %elem = extractelement <4 x float> %v, i32 3
  ret float %elem
}

; ==============================================================================
; 2 x f64
; ==============================================================================
; CHECK-LABEL: splat_v2f64:
; NO-SIMD128-NOT: f64x2
; SIMD128-VM-NOT: f64x2
; SIMD128: .param f64{{$}}
; SIMD128: .result v128{{$}}
; SIMD128: f64x2.splat $push0=, $0 # encoding: [0xfd,0x08]{{$}}
; SIMD128: return $pop0 # encoding: [0x0f]{{$}}
define <2 x double> @splat_v2f64(double %x) {
  %t1 = insertelement <2 x double> zeroinitializer, double %x, i3 0
  %res = insertelement <2 x double> %t1, double %x, i32 1
  ret <2 x double> %res
}

; CHECK-LABEL: extract_v2f64:
; NO-SIMD128-NOT: f64x2
; SIMD128-VM-NOT: f64x2
; SIMD128: .param v128{{$}}
; SIMD128: .result f64{{$}}
; SIMD128: f64x2.extract_lane $push0=, $0, 1 # encoding: [0xfd,0x10,0x01]{{$}}
; SIMD128: return $pop0 #
define double @extract_v2f64(<2 x double> %v) {
  %elem = extractelement <2 x double> %v, i32 1
  ret double %elem
}
