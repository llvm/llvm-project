; RUN: llc -mtriple=aarch64-linux-gnu -aarch64-neon-syntax=apple -mattr=+fullfp16 -o - %s | FileCheck %s

; This test file verifies the optimization for fmul(sitofp(x), C)
; where C is a constant reciprocal of a power of two,
; converting it to scvtf(X, 2^N).

; --- Scalar Tests ---

; Scalar f32 (from i32)
define float @test_f32_div(i32 %in) {
; CHECK-LABEL: test_f32_div:
; CHECK:       // %bb.0: // %entry
; CHECK-NEXT:  scvtf s0, w0, #4
; CHECK-NEXT:  ret
entry:
  %vcvt.i = sitofp i32 %in to float
  %div.i = fdiv float %vcvt.i, 16.0
  ret float %div.i
}

; Scalar f64 (from i64)
define double @test_f64_div(i64 %in) {
; CHECK-LABEL: test_f64_div:
; CHECK:       // %bb.0: // %entry
; CHECK-NEXT:  scvtf d0, x0, #4
; CHECK-NEXT:  ret
entry:
  %vcvt.i = sitofp i64 %in to double
  %div.i = fdiv double %vcvt.i, 16.0
  ret double %div.i
}

; --- Multi-Element Vector F32 Tests ---

; Vector v2f32 (from v2i32)
define <2 x float> @testv_v2f32_div(<2 x i32> %in) {
; CHECK-LABEL: testv_v2f32_div:
; CHECK:       // %bb.0: // %entry
; CHECK-NEXT:  scvtf d0, d0, #4
; CHECK-NEXT:  ret
entry:
  %vcvt.i = sitofp <2 x i32> %in to <2 x float>
  %div.i = fdiv <2 x float> %vcvt.i, <float 16.0, float 16.0>
  ret <2 x float> %div.i
}

; Vector v4f32 (from v4i32)
define <4 x float> @testv_v4f32_div(<4 x i32> %in) {
; CHECK-LABEL: testv_v4f32_div:
; CHECK:       // %bb.0: // %entry
; CHECK-NEXT:  scvtf q0, q0, #4
; CHECK-NEXT:  ret
entry:
  %vcvt.i = sitofp <4 x i32> %in to <4 x float>
  %div.i = fdiv <4 x float> %vcvt.i, <float 16.0, float 16.0, float 16.0, float 16.0>
  ret <4 x float> %div.i
}

; --- Multi-Element Vector F64 Tests ---

; Vector v2f64 (from v2i64)
define <2 x double> @testv_v2f64_div(<2 x i64> %in) {
; CHECK-LABEL: testv_v2f64_div:
; CHECK:       // %bb.0: // %entry
; CHECK-NEXT:  scvtf q0, q0, #4
; CHECK-NEXT:  ret
entry:
  %vcvt.i = sitofp <2 x i64> %in to <2 x double>
  %div.i = fdiv <2 x double> %vcvt.i, <double 16.0, double 16.0>
  ret <2 x double> %div.i
}