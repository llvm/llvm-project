; RUN: llc < %s -verify-machineinstrs -mcpu=mvp -mattr=+simd128 \
; RUN:   | FileCheck %s --check-prefix=NO-RELAXED
; RUN: llc < %s -verify-machineinstrs -mcpu=mvp -mattr=+simd128,+relaxed-simd \
; RUN:   | FileCheck %s --check-prefix=RELAXED

; Test that generic vector floating-point-to-integer conversions use relaxed
; SIMD instructions when relaxed SIMD is enabled.

target triple = "wasm32-unknown-unknown"

define <4 x i32> @fptosi_v4f32_v4i32(<4 x float> %a) {
; NO-RELAXED-LABEL: fptosi_v4f32_v4i32:
; NO-RELAXED:       i32x4.trunc_sat_f32x4_s
;
; RELAXED-LABEL:    fptosi_v4f32_v4i32:
; RELAXED:          i32x4.relaxed_trunc_f32x4_s
  %r = fptosi <4 x float> %a to <4 x i32>
  ret <4 x i32> %r
}

define <4 x i32> @fptoui_v4f32_v4i32(<4 x float> %a) {
; NO-RELAXED-LABEL: fptoui_v4f32_v4i32:
; NO-RELAXED:       i32x4.trunc_sat_f32x4_u
;
; RELAXED-LABEL:    fptoui_v4f32_v4i32:
; RELAXED:          i32x4.relaxed_trunc_f32x4_u
  %r = fptoui <4 x float> %a to <4 x i32>
  ret <4 x i32> %r
}
