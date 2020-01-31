; RUN: llc < %s -asm-verbose=false -verify-machineinstrs -disable-wasm-fallthrough-return-opt -wasm-keep-registers -wasm-disable-explicit-locals -mattr=+simd128 | FileCheck %s --check-prefixes CHECK,SIMD128
; RUN: llc < %s -asm-verbose=false -verify-machineinstrs -disable-wasm-fallthrough-return-opt -wasm-keep-registers -wasm-disable-explicit-locals | FileCheck %s --check-prefixes CHECK,NO-SIMD128

; Test that bitcasts between vector types are lowered to zero instructions

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: v16i8_to_v16i8:
; NO-SIMD128-NOT: return $0
; SIMD128: return $0
define <16 x i8> @v16i8_to_v16i8(<16 x i8> %v) {
  %res = bitcast <16 x i8> %v to <16 x i8>
  ret <16 x i8> %res
}

; CHECK-LABEL: v16i8_to_v8i16:
; NO-SIMD128-NOT: return $0
; SIMD128: return $0
define <8 x i16> @v16i8_to_v8i16(<16 x i8> %v) {
  %res = bitcast <16 x i8> %v to <8 x i16>
  ret <8 x i16> %res
}

; CHECK-LABEL: v16i8_to_v4i32:
; NO-SIMD128-NOT: return $0
; SIMD128: return $0
define <4 x i32> @v16i8_to_v4i32(<16 x i8> %v) {
  %res = bitcast <16 x i8> %v to <4 x i32>
  ret <4 x i32> %res
}

; CHECK-LABEL: v16i8_to_v2i64:
; NO-SIMD128-NOT: return $0
; SIMD128: return $0
define <2 x i64> @v16i8_to_v2i64(<16 x i8> %v) {
  %res = bitcast <16 x i8> %v to <2 x i64>
  ret <2 x i64> %res
}

; CHECK-LABEL: v16i8_to_v4f32:
; NO-SIMD128-NOT: return $0
; SIMD128: return $0
define <4 x float> @v16i8_to_v4f32(<16 x i8> %v) {
  %res = bitcast <16 x i8> %v to <4 x float>
  ret <4 x float> %res
}

; CHECK-LABEL: v16i8_to_v2f64:
; NO-SIMD128-NOT: return $0
; SIMD128: return $0
define <2 x double> @v16i8_to_v2f64(<16 x i8> %v) {
  %res = bitcast <16 x i8> %v to <2 x double>
  ret <2 x double> %res
}

; CHECK-LABEL: v8i16_to_v16i8:
; NO-SIMD128-NOT: return $0
; SIMD128: return $0
define <16 x i8> @v8i16_to_v16i8(<8 x i16> %v) {
  %res = bitcast <8 x i16> %v to <16 x i8>
  ret <16 x i8> %res
}

; CHECK-LABEL: v8i16_to_v8i16:
; NO-SIMD128-NOT: return $0
; SIMD128: return $0
define <8 x i16> @v8i16_to_v8i16(<8 x i16> %v) {
  %res = bitcast <8 x i16> %v to <8 x i16>
  ret <8 x i16> %res
}

; CHECK-LABEL: v8i16_to_v4i32:
; NO-SIMD128-NOT: return $0
; SIMD128: return $0
define <4 x i32> @v8i16_to_v4i32(<8 x i16> %v) {
  %res = bitcast <8 x i16> %v to <4 x i32>
  ret <4 x i32> %res
}

; CHECK-LABEL: v8i16_to_v2i64:
; NO-SIMD128-NOT: return $0
; SIMD128: return $0
define <2 x i64> @v8i16_to_v2i64(<8 x i16> %v) {
  %res = bitcast <8 x i16> %v to <2 x i64>
  ret <2 x i64> %res
}

; CHECK-LABEL: v8i16_to_v4f32:
; NO-SIMD128-NOT: return $0
; SIMD128: return $0
define <4 x float> @v8i16_to_v4f32(<8 x i16> %v) {
  %res = bitcast <8 x i16> %v to <4 x float>
  ret <4 x float> %res
}

; CHECK-LABEL: v8i16_to_v2f64:
; NO-SIMD128-NOT: return $0
; SIMD128: return $0
define <2 x double> @v8i16_to_v2f64(<8 x i16> %v) {
  %res = bitcast <8 x i16> %v to <2 x double>
  ret <2 x double> %res
}

; CHECK-LABEL: v4i32_to_v16i8:
; NO-SIMD128-NOT: return $0
; SIMD128: return $0
define <16 x i8> @v4i32_to_v16i8(<4 x i32> %v) {
  %res = bitcast <4 x i32> %v to <16 x i8>
  ret <16 x i8> %res
}

; CHECK-LABEL: v4i32_to_v8i16:
; NO-SIMD128-NOT: return $0
; SIMD128: return $0
define <8 x i16> @v4i32_to_v8i16(<4 x i32> %v) {
  %res = bitcast <4 x i32> %v to <8 x i16>
  ret <8 x i16> %res
}

; CHECK-LABEL: v4i32_to_v4i32:
; NO-SIMD128-NOT: return $0
; SIMD128: return $0
define <4 x i32> @v4i32_to_v4i32(<4 x i32> %v) {
  %res = bitcast <4 x i32> %v to <4 x i32>
  ret <4 x i32> %res
}

; CHECK-LABEL: v4i32_to_v2i64:
; NO-SIMD128-NOT: return $0
; SIMD128: return $0
define <2 x i64> @v4i32_to_v2i64(<4 x i32> %v) {
  %res = bitcast <4 x i32> %v to <2 x i64>
  ret <2 x i64> %res
}

; CHECK-LABEL: v4i32_to_v4f32:
; NO-SIMD128-NOT: return $0
; SIMD128: return $0
define <4 x float> @v4i32_to_v4f32(<4 x i32> %v) {
  %res = bitcast <4 x i32> %v to <4 x float>
  ret <4 x float> %res
}

; CHECK-LABEL: v4i32_to_v2f64:
; NO-SIMD128-NOT: return $0
; SIMD128: return $0
define <2 x double> @v4i32_to_v2f64(<4 x i32> %v) {
  %res = bitcast <4 x i32> %v to <2 x double>
  ret <2 x double> %res
}

; CHECK-LABEL: v2i64_to_v16i8:
; NO-SIMD128-NOT: return $0
; SIMD128: return $0
define <16 x i8> @v2i64_to_v16i8(<2 x i64> %v) {
  %res = bitcast <2 x i64> %v to <16 x i8>
  ret <16 x i8> %res
}

; CHECK-LABEL: v2i64_to_v8i16:
; NO-SIMD128-NOT: return $0
; SIMD128: return $0
define <8 x i16> @v2i64_to_v8i16(<2 x i64> %v) {
  %res = bitcast <2 x i64> %v to <8 x i16>
  ret <8 x i16> %res
}

; CHECK-LABEL: v2i64_to_v4i32:
; NO-SIMD128-NOT: return $0
; SIMD128: return $0
define <4 x i32> @v2i64_to_v4i32(<2 x i64> %v) {
  %res = bitcast <2 x i64> %v to <4 x i32>
  ret <4 x i32> %res
}

; CHECK-LABEL: v2i64_to_v2i64:
; NO-SIMD128-NOT: return $0
; SIMD128: return $0
define <2 x i64> @v2i64_to_v2i64(<2 x i64> %v) {
  %res = bitcast <2 x i64> %v to <2 x i64>
  ret <2 x i64> %res
}

; CHECK-LABEL: v2i64_to_v4f32:
; NO-SIMD128-NOT: return $0
; SIMD128: return $0
define <4 x float> @v2i64_to_v4f32(<2 x i64> %v) {
  %res = bitcast <2 x i64> %v to <4 x float>
  ret <4 x float> %res
}

; CHECK-LABEL: v2i64_to_v2f64:
; NO-SIMD128-NOT: return $0
; SIMD128: return $0
define <2 x double> @v2i64_to_v2f64(<2 x i64> %v) {
  %res = bitcast <2 x i64> %v to <2 x double>
  ret <2 x double> %res
}

; CHECK-LABEL: v4f32_to_v16i8:
; NO-SIMD128-NOT: return $0
; SIMD128: return $0
define <16 x i8> @v4f32_to_v16i8(<4 x float> %v) {
  %res = bitcast <4 x float> %v to <16 x i8>
  ret <16 x i8> %res
}

; CHECK-LABEL: v4f32_to_v8i16:
; NO-SIMD128-NOT: return $0
; SIMD128: return $0
define <8 x i16> @v4f32_to_v8i16(<4 x float> %v) {
  %res = bitcast <4 x float> %v to <8 x i16>
  ret <8 x i16> %res
}

; CHECK-LABEL: v4f32_to_v4i32:
; NO-SIMD128-NOT: return $0
; SIMD128: return $0
define <4 x i32> @v4f32_to_v4i32(<4 x float> %v) {
  %res = bitcast <4 x float> %v to <4 x i32>
  ret <4 x i32> %res
}

; CHECK-LABEL: v4f32_to_v2i64:
; NO-SIMD128-NOT: return $0
; SIMD128: return $0
define <2 x i64> @v4f32_to_v2i64(<4 x float> %v) {
  %res = bitcast <4 x float> %v to <2 x i64>
  ret <2 x i64> %res
}

; CHECK-LABEL: v4f32_to_v4f32:
; NO-SIMD128-NOT: return $0
; SIMD128: return $0
define <4 x float> @v4f32_to_v4f32(<4 x float> %v) {
  %res = bitcast <4 x float> %v to <4 x float>
  ret <4 x float> %res
}

; CHECK-LABEL: v4f32_to_v2f64:
; NO-SIMD128-NOT: return $0
; SIMD128: return $0
define <2 x double> @v4f32_to_v2f64(<4 x float> %v) {
  %res = bitcast <4 x float> %v to <2 x double>
  ret <2 x double> %res
}

; CHECK-LABEL: v2f64_to_v16i8:
; NO-SIMD128-NOT: return $0
; SIMD128: return $0
define <16 x i8> @v2f64_to_v16i8(<2 x double> %v) {
  %res = bitcast <2 x double> %v to <16 x i8>
  ret <16 x i8> %res
}

; CHECK-LABEL: v2f64_to_v8i16:
; NO-SIMD128-NOT: return $0
; SIMD128: return $0
define <8 x i16> @v2f64_to_v8i16(<2 x double> %v) {
  %res = bitcast <2 x double> %v to <8 x i16>
  ret <8 x i16> %res
}

; CHECK-LABEL: v2f64_to_v4i32:
; NO-SIMD128-NOT: return $0
; SIMD128: return $0
define <4 x i32> @v2f64_to_v4i32(<2 x double> %v) {
  %res = bitcast <2 x double> %v to <4 x i32>
  ret <4 x i32> %res
}

; CHECK-LABEL: v2f64_to_v2i64:
; NO-SIMD128-NOT: return $0
; SIMD128: return $0
define <2 x i64> @v2f64_to_v2i64(<2 x double> %v) {
  %res = bitcast <2 x double> %v to <2 x i64>
  ret <2 x i64> %res
}

; CHECK-LABEL: v2f64_to_v4f32:
; NO-SIMD128-NOT: return $0
; SIMD128: return $0
define <4 x float> @v2f64_to_v4f32(<2 x double> %v) {
  %res = bitcast <2 x double> %v to <4 x float>
  ret <4 x float> %res
}

; CHECK-LABEL: v2f64_to_v2f64:
; NO-SIMD128-NOT: return $0
; SIMD128: return $0
define <2 x double> @v2f64_to_v2f64(<2 x double> %v) {
  %res = bitcast <2 x double> %v to <2 x double>
  ret <2 x double> %res
}
