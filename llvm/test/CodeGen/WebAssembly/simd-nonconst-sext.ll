; RUN: llc < %s -asm-verbose=false -verify-machineinstrs -mattr=+simd128 | FileCheck %s

; A regression test for a bug in the lowering of SIGN_EXTEND_INREG
; with SIMD and without sign-ext where ISel would crash if the index
; of the vector extract was not a constant.

target triple = "wasm32"

; CHECK-LABEL: foo:
; CHECK-NEXT: .functype foo () -> (f32)
; CHECK: i32x4.load16x4_u
; CHECK: f32.convert_i32_s
define float @foo() {
  %1 = load <4 x i16>, <4 x i16>* undef, align 8
  %2 = load i32, i32* undef, align 4
  %vecext = extractelement <4 x i16> %1, i32 %2
  %conv = sitofp i16 %vecext to float
  ret float %conv
}
