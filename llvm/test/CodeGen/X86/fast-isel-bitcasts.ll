; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+sse2 -fast-isel -fast-isel-abort=1 -asm-verbose=0 | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+avx -fast-isel -fast-isel-abort=1 -asm-verbose=0 | FileCheck %s
;
; Bitcasts between 128-bit vector types are no-ops since no instruction is
; needed for the conversion.

define <2 x i64> @v4i32_to_v2i64(<4 x i32> %a) {
;CHECK-LABEL: v4i32_to_v2i64:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <4 x i32> %a to <2 x i64>
  ret <2 x i64> %1
}

define <2 x i64> @v8i16_to_v2i64(<8 x i16> %a) {
;CHECK-LABEL: v8i16_to_v2i64:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <8 x i16> %a to <2 x i64>
  ret <2 x i64> %1
}

define <2 x i64> @v16i8_to_v2i64(<16 x i8> %a) {
;CHECK-LABEL: v16i8_to_v2i64:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <16 x i8> %a to <2 x i64>
  ret <2 x i64> %1
}

define <2 x i64> @v2f64_to_v2i64(<2 x double> %a) {
;CHECK-LABEL: v2f64_to_v2i64:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <2 x double> %a to <2 x i64>
  ret <2 x i64> %1
}

define <2 x i64> @v4f32_to_v2i64(<4 x float> %a) {
;CHECK-LABEL: v4f32_to_v2i64:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <4 x float> %a to <2 x i64>
  ret <2 x i64> %1
}

define <4 x i32> @v2i64_to_v4i32(<2 x i64> %a) {
;CHECK-LABEL: v2i64_to_v4i32:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <2 x i64> %a to <4 x i32>
  ret <4 x i32> %1
}

define <4 x i32> @v8i16_to_v4i32(<8 x i16> %a) {
;CHECK-LABEL: v8i16_to_v4i32:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <8 x i16> %a to <4 x i32>
  ret <4 x i32> %1
}

define <4 x i32> @v16i8_to_v4i32(<16 x i8> %a) {
;CHECK-LABEL: v16i8_to_v4i32:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <16 x i8> %a to <4 x i32>
  ret <4 x i32> %1
}

define <4 x i32> @v2f64_to_v4i32(<2 x double> %a) {
;CHECK-LABEL: v2f64_to_v4i32:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <2 x double> %a to <4 x i32>
  ret <4 x i32> %1
}

define <4 x i32> @v4f32_to_v4i32(<4 x float> %a) {
;CHECK-LABEL: v4f32_to_v4i32:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <4 x float> %a to <4 x i32>
  ret <4 x i32> %1
}

define <8 x i16> @v2i64_to_v8i16(<2 x i64> %a) {
;CHECK-LABEL: v2i64_to_v8i16:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <2 x i64> %a to <8 x i16>
  ret <8 x i16> %1
}

define <8 x i16> @v4i32_to_v8i16(<4 x i32> %a) {
;CHECK-LABEL: v4i32_to_v8i16:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <4 x i32> %a to <8 x i16>
  ret <8 x i16> %1
}

define <8 x i16> @v16i8_to_v8i16(<16 x i8> %a) {
;CHECK-LABEL: v16i8_to_v8i16:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <16 x i8> %a to <8 x i16>
  ret <8 x i16> %1
}

define <8 x i16> @v2f64_to_v8i16(<2 x double> %a) {
;CHECK-LABEL: v2f64_to_v8i16:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <2 x double> %a to <8 x i16>
  ret <8 x i16> %1
}

define <8 x i16> @v4f32_to_v8i16(<4 x float> %a) {
;CHECK-LABEL: v4f32_to_v8i16:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <4 x float> %a to <8 x i16>
  ret <8 x i16> %1
}

define <16 x i8> @v8i16_to_v16i8(<8 x i16> %a) {
;CHECK-LABEL: v8i16_to_v16i8:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <8 x i16> %a to <16 x i8>
  ret <16 x i8> %1
}

define <16 x i8> @v2i64_to_v16i8(<2 x i64> %a) {
;CHECK-LABEL: v2i64_to_v16i8:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <2 x i64> %a to <16 x i8>
  ret <16 x i8> %1
}

define <16 x i8> @v4i32_to_v16i8(<4 x i32> %a) {
;CHECK-LABEL: v4i32_to_v16i8:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <4 x i32> %a to <16 x i8>
  ret <16 x i8> %1
}

define <16 x i8> @v2f64_to_v16i8(<2 x double> %a) {
;CHECK-LABEL: v2f64_to_v16i8:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <2 x double> %a to <16 x i8>
  ret <16 x i8> %1
}

define <16 x i8> @v4f32_to_v16i8(<4 x float> %a) {
;CHECK-LABEL: v4f32_to_v16i8:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <4 x float> %a to <16 x i8>
  ret <16 x i8> %1
}

define <4 x float> @v16i8_to_v4f32(<16 x i8> %a) {
;CHECK-LABEL: v16i8_to_v4f32:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <16 x i8> %a to <4 x float>
  ret <4 x float> %1
}

define <4 x float> @v8i16_to_v4f32(<8 x i16> %a) {
;CHECK-LABEL: v8i16_to_v4f32:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <8 x i16> %a to <4 x float>
  ret <4 x float> %1
}

define <4 x float> @v2i64_to_v4f32(<2 x i64> %a) {
;CHECK-LABEL: v2i64_to_v4f32:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <2 x i64> %a to <4 x float>
  ret <4 x float> %1
}

define <4 x float> @v4i32_to_v4f32(<4 x i32> %a) {
;CHECK-LABEL: v4i32_to_v4f32:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <4 x i32> %a to <4 x float>
  ret <4 x float> %1
}

define <4 x float> @v2f64_to_v4f32(<2 x double> %a) {
;CHECK-LABEL: v2f64_to_v4f32:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <2 x double> %a to <4 x float>
  ret <4 x float> %1
}

define <2 x double> @v4f32_to_v2f64(<4 x float> %a) {
;CHECK-LABEL: v4f32_to_v2f64:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <4 x float> %a to <2 x double>
  ret <2 x double> %1
}

define <2 x double> @v16i8_to_v2f64(<16 x i8> %a) {
;CHECK-LABEL: v16i8_to_v2f64:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <16 x i8> %a to <2 x double>
  ret <2 x double> %1
}

define <2 x double> @v8i16_to_v2f64(<8 x i16> %a) {
;CHECK-LABEL: v8i16_to_v2f64:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <8 x i16> %a to <2 x double>
  ret <2 x double> %1
}

define <2 x double> @v2i64_to_v2f64(<2 x i64> %a) {
;CHECK-LABEL: v2i64_to_v2f64:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <2 x i64> %a to <2 x double>
  ret <2 x double> %1
}

define <2 x double> @v4i32_to_v2f64(<4 x i32> %a) {
;CHECK-LABEL: v4i32_to_v2f64:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <4 x i32> %a to <2 x double>
  ret <2 x double> %1
}
