; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+avx512bw -fast-isel -fast-isel-abort=1 -asm-verbose=0 | FileCheck %s
;
; Bitcasts between 512-bit vector types are no-ops since no instruction is
; needed for the conversion.

define <8 x i64> @v16i32_to_v8i64(<16 x i32> %a) {
;CHECK-LABEL: v16i32_to_v8i64:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <16 x i32> %a to <8 x i64>
  ret <8 x i64> %1
}

define <8 x i64> @v32i16_to_v8i64(<32 x i16> %a) {
;CHECK-LABEL: v32i16_to_v8i64:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <32 x i16> %a to <8 x i64>
  ret <8 x i64> %1
}

define <8 x i64> @v64i8_to_v8i64(<64 x i8> %a) {
;CHECK-LABEL: v64i8_to_v8i64:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <64 x i8> %a to <8 x i64>
  ret <8 x i64> %1
}

define <8 x i64> @v8f64_to_v8i64(<8 x double> %a) {
;CHECK-LABEL: v8f64_to_v8i64:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <8 x double> %a to <8 x i64>
  ret <8 x i64> %1
}

define <8 x i64> @v16f32_to_v8i64(<16 x float> %a) {
;CHECK-LABEL: v16f32_to_v8i64:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <16 x float> %a to <8 x i64>
  ret <8 x i64> %1
}

define <16 x i32> @v8i64_to_v16i32(<8 x i64> %a) {
;CHECK-LABEL: v8i64_to_v16i32:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <8 x i64> %a to <16 x i32>
  ret <16 x i32> %1
}

define <16 x i32> @v32i16_to_v16i32(<32 x i16> %a) {
;CHECK-LABEL: v32i16_to_v16i32:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <32 x i16> %a to <16 x i32>
  ret <16 x i32> %1
}

define <16 x i32> @v64i8_to_v16i32(<64 x i8> %a) {
;CHECK-LABEL: v64i8_to_v16i32:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <64 x i8> %a to <16 x i32>
  ret <16 x i32> %1
}

define <16 x i32> @v8f64_to_v16i32(<8 x double> %a) {
;CHECK-LABEL: v8f64_to_v16i32:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <8 x double> %a to <16 x i32>
  ret <16 x i32> %1
}

define <16 x i32> @v16f32_to_v16i32(<16 x float> %a) {
;CHECK-LABEL: v16f32_to_v16i32:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <16 x float> %a to <16 x i32>
  ret <16 x i32> %1
}

define <32 x i16> @v8i64_to_v32i16(<8 x i64> %a) {
;CHECK-LABEL: v8i64_to_v32i16:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <8 x i64> %a to <32 x i16>
  ret <32 x i16> %1
}

define <32 x i16> @v16i32_to_v32i16(<16 x i32> %a) {
;CHECK-LABEL: v16i32_to_v32i16:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <16 x i32> %a to <32 x i16>
  ret <32 x i16> %1
}

define <32 x i16> @v64i8_to_v32i16(<64 x i8> %a) {
;CHECK-LABEL: v64i8_to_v32i16:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <64 x i8> %a to <32 x i16>
  ret <32 x i16> %1
}

define <32 x i16> @v8f64_to_v32i16(<8 x double> %a) {
;CHECK-LABEL: v8f64_to_v32i16:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <8 x double> %a to <32 x i16>
  ret <32 x i16> %1
}

define <32 x i16> @v16f32_to_v32i16(<16 x float> %a) {
;CHECK-LABEL: v16f32_to_v32i16:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <16 x float> %a to <32 x i16>
  ret <32 x i16> %1
}

define <64 x i8> @v32i16_to_v64i8(<32 x i16> %a) {
;CHECK-LABEL: v32i16_to_v64i8:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <32 x i16> %a to <64 x i8>
  ret <64 x i8> %1
}

define <64 x i8> @v8i64_to_v64i8(<8 x i64> %a) {
;CHECK-LABEL: v8i64_to_v64i8:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <8 x i64> %a to <64 x i8>
  ret <64 x i8> %1
}

define <64 x i8> @v16i32_to_v64i8(<16 x i32> %a) {
;CHECK-LABEL: v16i32_to_v64i8:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <16 x i32> %a to <64 x i8>
  ret <64 x i8> %1
}

define <64 x i8> @v8f64_to_v64i8(<8 x double> %a) {
;CHECK-LABEL: v8f64_to_v64i8:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <8 x double> %a to <64 x i8>
  ret <64 x i8> %1
}

define <64 x i8> @v16f32_to_v64i8(<16 x float> %a) {
;CHECK-LABEL: v16f32_to_v64i8:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <16 x float> %a to <64 x i8>
  ret <64 x i8> %1
}

define <16 x float> @v64i8_to_v16f32(<64 x i8> %a) {
;CHECK-LABEL: v64i8_to_v16f32:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <64 x i8> %a to <16 x float>
  ret <16 x float> %1
}

define <16 x float> @v32i16_to_v16f32(<32 x i16> %a) {
;CHECK-LABEL: v32i16_to_v16f32:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <32 x i16> %a to <16 x float>
  ret <16 x float> %1
}

define <16 x float> @v8i64_to_v16f32(<8 x i64> %a) {
;CHECK-LABEL: v8i64_to_v16f32:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <8 x i64> %a to <16 x float>
  ret <16 x float> %1
}

define <16 x float> @v16i32_to_v16f32(<16 x i32> %a) {
;CHECK-LABEL: v16i32_to_v16f32:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <16 x i32> %a to <16 x float>
  ret <16 x float> %1
}

define <16 x float> @v8f64_to_v16f32(<8 x double> %a) {
;CHECK-LABEL: v8f64_to_v16f32:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <8 x double> %a to <16 x float>
  ret <16 x float> %1
}

define <8 x double> @v16f32_to_v8f64(<16 x float> %a) {
;CHECK-LABEL: v16f32_to_v8f64:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <16 x float> %a to <8 x double>
  ret <8 x double> %1
}

define <8 x double> @v64i8_to_v8f64(<64 x i8> %a) {
;CHECK-LABEL: v64i8_to_v8f64:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <64 x i8> %a to <8 x double>
  ret <8 x double> %1
}

define <8 x double> @v32i16_to_v8f64(<32 x i16> %a) {
;CHECK-LABEL: v32i16_to_v8f64:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <32 x i16> %a to <8 x double>
  ret <8 x double> %1
}

define <8 x double> @v8i64_to_v8f64(<8 x i64> %a) {
;CHECK-LABEL: v8i64_to_v8f64:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <8 x i64> %a to <8 x double>
  ret <8 x double> %1
}

define <8 x double> @v16i32_to_v8f64(<16 x i32> %a) {
;CHECK-LABEL: v16i32_to_v8f64:
;CHECK-NEXT: .cfi_startproc
;CHECK-NEXT: ret
  %1 = bitcast <16 x i32> %a to <8 x double>
  ret <8 x double> %1
}
