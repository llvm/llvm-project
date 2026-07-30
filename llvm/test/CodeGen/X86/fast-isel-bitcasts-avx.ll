; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+avx -fast-isel -fast-isel-abort=1 -asm-verbose=0 | FileCheck %s
;
; Bitcasts between 256-bit vector types are no-ops since no instruction is
; needed for the conversion.

define <4 x i64> @v8i32_to_v4i64(<8 x i32> %a) {
;CHECK-LABEL: v8i32_to_v4i64:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <8 x i32> %a to <4 x i64>
  ret <4 x i64> %1
}

define <4 x i64> @v16i16_to_v4i64(<16 x i16> %a) {
;CHECK-LABEL: v16i16_to_v4i64:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <16 x i16> %a to <4 x i64>
  ret <4 x i64> %1
}

define <4 x i64> @v32i8_to_v4i64(<32 x i8> %a) {
;CHECK-LABEL: v32i8_to_v4i64:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <32 x i8> %a to <4 x i64>
  ret <4 x i64> %1
}

define <4 x i64> @v4f64_to_v4i64(<4 x double> %a) {
;CHECK-LABEL: v4f64_to_v4i64:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <4 x double> %a to <4 x i64>
  ret <4 x i64> %1
}

define <4 x i64> @v8f32_to_v4i64(<8 x float> %a) {
;CHECK-LABEL: v8f32_to_v4i64:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <8 x float> %a to <4 x i64>
  ret <4 x i64> %1
}

define <8 x i32> @v4i64_to_v8i32(<4 x i64> %a) {
;CHECK-LABEL: v4i64_to_v8i32:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <4 x i64> %a to <8 x i32>
  ret <8 x i32> %1
}

define <8 x i32> @v16i16_to_v8i32(<16 x i16> %a) {
;CHECK-LABEL: v16i16_to_v8i32:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <16 x i16> %a to <8 x i32>
  ret <8 x i32> %1
}

define <8 x i32> @v32i8_to_v8i32(<32 x i8> %a) {
;CHECK-LABEL: v32i8_to_v8i32:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <32 x i8> %a to <8 x i32>
  ret <8 x i32> %1
}

define <8 x i32> @v4f64_to_v8i32(<4 x double> %a) {
;CHECK-LABEL: v4f64_to_v8i32:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <4 x double> %a to <8 x i32>
  ret <8 x i32> %1
}

define <8 x i32> @v8f32_to_v8i32(<8 x float> %a) {
;CHECK-LABEL: v8f32_to_v8i32:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <8 x float> %a to <8 x i32>
  ret <8 x i32> %1
}

define <16 x i16> @v4i64_to_v16i16(<4 x i64> %a) {
;CHECK-LABEL: v4i64_to_v16i16:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <4 x i64> %a to <16 x i16>
  ret <16 x i16> %1
}

define <16 x i16> @v8i32_to_v16i16(<8 x i32> %a) {
;CHECK-LABEL: v8i32_to_v16i16:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <8 x i32> %a to <16 x i16>
  ret <16 x i16> %1
}

define <16 x i16> @v32i8_to_v16i16(<32 x i8> %a) {
;CHECK-LABEL: v32i8_to_v16i16:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <32 x i8> %a to <16 x i16>
  ret <16 x i16> %1
}

define <16 x i16> @v4f64_to_v16i16(<4 x double> %a) {
;CHECK-LABEL: v4f64_to_v16i16:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <4 x double> %a to <16 x i16>
  ret <16 x i16> %1
}

define <16 x i16> @v8f32_to_v16i16(<8 x float> %a) {
;CHECK-LABEL: v8f32_to_v16i16:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <8 x float> %a to <16 x i16>
  ret <16 x i16> %1
}

define <32 x i8> @v16i16_to_v32i8(<16 x i16> %a) {
;CHECK-LABEL: v16i16_to_v32i8:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <16 x i16> %a to <32 x i8>
  ret <32 x i8> %1
}

define <32 x i8> @v4i64_to_v32i8(<4 x i64> %a) {
;CHECK-LABEL: v4i64_to_v32i8:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <4 x i64> %a to <32 x i8>
  ret <32 x i8> %1
}

define <32 x i8> @v8i32_to_v32i8(<8 x i32> %a) {
;CHECK-LABEL: v8i32_to_v32i8:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <8 x i32> %a to <32 x i8>
  ret <32 x i8> %1
}

define <32 x i8> @v4f64_to_v32i8(<4 x double> %a) {
;CHECK-LABEL: v4f64_to_v32i8:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <4 x double> %a to <32 x i8>
  ret <32 x i8> %1
}

define <32 x i8> @v8f32_to_v32i8(<8 x float> %a) {
;CHECK-LABEL: v8f32_to_v32i8:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <8 x float> %a to <32 x i8>
  ret <32 x i8> %1
}

define <8 x float> @v32i8_to_v8f32(<32 x i8> %a) {
;CHECK-LABEL: v32i8_to_v8f32:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <32 x i8> %a to <8 x float>
  ret <8 x float> %1
}

define <8 x float> @v16i16_to_v8f32(<16 x i16> %a) {
;CHECK-LABEL: v16i16_to_v8f32:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <16 x i16> %a to <8 x float>
  ret <8 x float> %1
}

define <8 x float> @v4i64_to_v8f32(<4 x i64> %a) {
;CHECK-LABEL: v4i64_to_v8f32:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <4 x i64> %a to <8 x float>
  ret <8 x float> %1
}

define <8 x float> @v8i32_to_v8f32(<8 x i32> %a) {
;CHECK-LABEL: v8i32_to_v8f32:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <8 x i32> %a to <8 x float>
  ret <8 x float> %1
}

define <8 x float> @v4f64_to_v8f32(<4 x double> %a) {
;CHECK-LABEL: v4f64_to_v8f32:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <4 x double> %a to <8 x float>
  ret <8 x float> %1
}

define <4 x double> @v8f32_to_v4f64(<8 x float> %a) {
;CHECK-LABEL: v8f32_to_v4f64:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <8 x float> %a to <4 x double>
  ret <4 x double> %1
}

define <4 x double> @v32i8_to_v4f64(<32 x i8> %a) {
;CHECK-LABEL: v32i8_to_v4f64:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <32 x i8> %a to <4 x double>
  ret <4 x double> %1
}

define <4 x double> @v16i16_to_v4f64(<16 x i16> %a) {
;CHECK-LABEL: v16i16_to_v4f64:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <16 x i16> %a to <4 x double>
  ret <4 x double> %1
}

define <4 x double> @v4i64_to_v4f64(<4 x i64> %a) {
;CHECK-LABEL: v4i64_to_v4f64:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <4 x i64> %a to <4 x double>
  ret <4 x double> %1
}

define <4 x double> @v8i32_to_v4f64(<8 x i32> %a) {
;CHECK-LABEL: v8i32_to_v4f64:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <8 x i32> %a to <4 x double>
  ret <4 x double> %1
}
