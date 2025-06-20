; RUN: llc -mtriple=aarch64-linux-gnu -aarch64-neon-syntax=apple -mattr=+fullfp16 -o - %s | FileCheck %s

; This test file verifies that fdiv(sitofp(x), C)
; where C is a constant power of two,
; is optimized to scvtf(X, shift_amount).
; This typically involves an implicit fdiv -> fmul_reciprocal transformation.

; --- Scalar Tests ---

; Scalar f32 (from i32)
define float @test_f32_div_const(i32 %in) {
; CHECK-LABEL: test_f32_div_const:
; CHECK:       // %bb.0: // %entry
; CHECK-NEXT:  scvtf s0, w0, #4
; CHECK-NEXT:  ret
entry:
  %vcvt.i = sitofp i32 %in to float
  %div.i = fdiv float %vcvt.i, 16.0
  ret float %div.i
}

; Scalar f64 (from i64)
define double @test_f64_div_const(i64 %in) {
; CHECK-LABEL: test_f64_div_const:
; CHECK:       // %bb.0: // %entry
; CHECK-NEXT:  scvtf d0, x0, #4
; CHECK-NEXT:  ret
entry:
  %vcvt.i = sitofp i64 %in to double
  %div.i = fdiv double %vcvt.i, 16.0
  ret double %div.i
}

; --- Vector Tests ---

; Vector v2f32 (from v2i32)
define <2 x float> @testv_v2f32_div_const(<2 x i32> %in) {
; CHECK-LABEL: testv_v2f32_div_const:
; CHECK:       // %bb.0: // %entry
; CHECK-NEXT:  scvtf.2s v0, v0, #4
; CHECK-NEXT:  ret
entry:
  %vcvt.i = sitofp <2 x i32> %in to <2 x float>
  %div.i = fdiv <2 x float> %vcvt.i, <float 16.0, float 16.0>
  ret <2 x float> %div.i
}

; Vector v4f32 (from v4i32)
define <4 x float> @testv_v4f32_div_const(<4 x i32> %in) {
; CHECK-LABEL: testv_v4f32_div_const:
; CHECK:       // %bb.0: // %entry
; CHECK-NEXT:  scvtf.4s v0, v0, #4
; CHECK-NEXT:  ret
entry:
  %vcvt.i = sitofp <4 x i32> %in to <4 x float>
  %div.i = fdiv <4 x float> %vcvt.i, <float 16.0, float 16.0, float 16.0, float 16.0>
  ret <4 x float> %div.i
}

; Vector v2f64 (from v2i64)
define <2 x double> @testv_v2f64_div_const(<2 x i64> %in) {
; CHECK-LABEL: testv_v2f64_div_const:
; CHECK:       // %bb.0: // %entry
; CHECK-NEXT:  scvtf.2d v0, v0, #4
; CHECK-NEXT:  ret
entry:
  %vcvt.i = sitofp <2 x i64> %in to <2 x double>
  %div.i = fdiv <2 x double> %vcvt.i, <double 16.0, double 16.0>
  ret <2 x double> %div.i
}

; --- f16 Tests (assuming fullfp16 is enabled) ---

; Vector v4f16 (from v4i16)
define <4 x half> @testv_v4f16_div_const(<4 x i16> %in) {
; CHECK-LABEL: testv_v4f16_div_const:
; CHECK:       // %bb.0: // %entry
; CHECK-NEXT:  scvtf.4h v0, v0, #4
; CHECK-NEXT:  ret
entry:
  %vcvt.i = sitofp <4 x i16> %in to <4 x half>
  %div.i = fdiv <4 x half> %vcvt.i, <half 16.0, half 16.0, half 16.0, half 16.0> ; 16.0 in half-precision
  ret <4 x half> %div.i
}

; Vector v8f16 (from v8i16)
define <8 x half> @testv_v8f16_div_const(<8 x i16> %in) {
; CHECK-LABEL: testv_v8f16_div_const:
; CHECK:       // %bb.0: // %entry
; CHECK-NEXT:  scvtf.8h v0, v0, #4
; CHECK-NEXT:  ret
entry:
  %vcvt.i = sitofp <8 x i16> %in to <8 x half>
  %div.i = fdiv <8 x half> %vcvt.i, <half 16.0, half 16.0, half 16.0, half 16.0, half 16.0, half 16.0, half 16.0, half 16.0> ; 16.0 in half-precision
  ret <8 x half> %div.i
}