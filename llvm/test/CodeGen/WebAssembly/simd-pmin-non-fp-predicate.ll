; NOTE: Regression test for #211459 - WebAssembly isel must not crash when
; relaxed pmin/pmax PatFrag predicates evaluate isKnownNeverNaN on operands.
; RUN: llc -O2 < %s -mtriple=wasm32-unknown-unknown -mattr=+simd128,+relaxed-simd -o /dev/null
; RUN: llc -O3 < %s -mtriple=wasm32-unknown-unknown -mattr=+simd128,+relaxed-simd -o /dev/null

target triple = "wasm32-unknown-unknown"

; Exercises relaxed_pmin predicate path (fcmp+select without nnan/nsz flags).
define <4 x float> @pmin_v4f32_ult(<4 x float> %x, <4 x float> %y) {
; CHECK-LABEL: pmin_v4f32_ult:
; CHECK: f32x4.pmin
  %c = fcmp olt <4 x float> %y, %x
  %a = select <4 x i1> %c, <4 x float> %y, <4 x float> %x
  ret <4 x float> %a
}

; Exercises relaxed_pmax predicate path.
define <4 x float> @pmax_v4f32_ogt(<4 x float> %x, <4 x float> %y) {
; CHECK-LABEL: pmax_v4f32_ogt:
; CHECK: f32x4.pmax
  %c = fcmp ogt <4 x float> %x, %y
  %a = select <4 x i1> %c, <4 x float> %y, <4 x float> %x
  ret <4 x float> %a
}

; Integer v128 min/max path (v4i32 vselect with fp comparison).
define <4 x i32> @pmin_int_v4f32(<4 x i32> %x, <4 x i32> %y) {
; CHECK-LABEL: pmin_int_v4f32:
; CHECK: f32x4.relaxed_min
  %fx = bitcast <4 x i32> %x to <4 x float>
  %fy = bitcast <4 x i32> %y to <4 x float>
  %c = fcmp olt <4 x float> %fy, %fx
  %a = select <4 x i1> %c, <4 x i32> %x, <4 x i32> %y
  ret <4 x i32> %a
}

; fminnum lowering path through HasNoSignedZerosOrNaNs.
define <4 x float> @fminnum_v4f32(<4 x float> %a, <4 x float> %b) {
; CHECK-LABEL: fminnum_v4f32:
; CHECK: f32x4.relaxed_min
  %r = call <4 x float> @llvm.minnum.v4f32(<4 x float> %a, <4 x float> %b)
  ret <4 x float> %r
}

declare <4 x float> @llvm.minnum.v4f32(<4 x float>, <4 x float>)
