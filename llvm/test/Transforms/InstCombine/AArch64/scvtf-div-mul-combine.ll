; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+fullfp16 -aarch64-neon-syntax=apple -verify-machineinstrs -o - %s | FileCheck %s

; Scalar fdiv by 16.0 (f32)
define float @tests_f32_div(i32 %in) {
; CHECK-LABEL: tests_f32_div:
; CHECK:       // %bb.0: // %entry
; CHECK-NEXT:    scvtf   s0, w0, #4
; CHECK-NEXT:    ret
entry:
  %vcvt.i = sitofp i32 %in to float
  %div.i = fdiv float %vcvt.i, 16.0
  ret float %div.i
}

; Scalar fmul by (2^-4) (f32)
define float @testsmul_f32_mul(i32 %in) local_unnamed_addr #0 {
; CHECK-LABEL: testsmul_f32_mul:
; CHECK:       // %bb.0:
; CHECK-NEXT:    scvtf   s0, w0, #4
; CHECK-NEXT:    ret
  %vcvt.i = sitofp i32 %in to float
  %div.i = fmul float %vcvt.i, 6.250000e-02 ; 0.0625 is 2^-4
  ret float %div.i
}

; Vector fdiv by 16.0 (v2f32)
define <2 x float> @testv_v2f32_div(<2 x i32> %in) {
; CHECK-LABEL: testv_v2f32_div:
; CHECK:       // %bb.0: // %entry
; CHECK-NEXT:    scvtf.2s        v0, v0, #4
; CHECK-NEXT:    ret
entry:
  %vcvt.i = sitofp <2 x i32> %in to <2 x float>
  %div.i = fdiv <2 x float> %vcvt.i, <float 16.0, float 16.0>
  ret <2 x float> %div.i
}

; Vector fmul by 2^-4 (v2f32)
define <2 x float> @testvmul_v2f32_mul(<2 x i32> %in) local_unnamed_addr #0 {
; CHECK-LABEL: testvmul_v2f32_mul:
; CHECK:       // %bb.0:
; CHECK-NEXT:    scvtf.2s        v0, v0, #4
; CHECK-NEXT:    ret
  %vcvt.i = sitofp <2 x i32> %in to <2 x float>
  %div.i = fmul <2 x float> %vcvt.i, splat (float 6.250000e-02) ; 0.0625 is 2^-4
  ret <2 x float> %div.i
}

; Scalar fdiv by 16.0 (f64)
define double @tests_f64_div(i64 %in) {
; CHECK-LABEL: tests_f64_div:
; CHECK:       // %bb.0: // %entry
; CHECK-NEXT:    scvtf   d0, x0, #4
; CHECK-NEXT:    ret
entry:
  %vcvt.i = sitofp i64 %in to double
  %div.i = fdiv double %vcvt.i, 1.600000e+01 ; 16.0 in double-precision
  ret double %div.i
}

; Scalar fmul by (2^-4) (f64)
define double @testsmul_f64_mul(i64 %in) local_unnamed_addr #0 {
; CHECK-LABEL: testsmul_f64_mul:
; CHECK:       // %bb.0:
; CHECK-NEXT:    scvtf   d0, x0, #4
; CHECK-NEXT:    ret
  %vcvt.i = sitofp i64 %in to double
  %div.i = fmul double %vcvt.i, 6.250000e-02 ; 0.0625 is 2^-4 in double-precision
  ret double %div.i
}

; Vector fdiv by 16.0 (v2f64)
define <2 x double> @testv_v2f64_div(<2 x i64> %in) {
; CHECK-LABEL: testv_v2f64_div:
; CHECK:       // %bb.0: // %entry
; CHECK-NEXT:    scvtf.2d        v0, v0, #4
; CHECK-NEXT:    ret
entry:
  %vcvt.i = sitofp <2 x i64> %in to <2 x double>
  %div.i = fdiv <2 x double> %vcvt.i, <double 1.600000e+01, double 1.600000e+01>
  ret <2 x double> %div.i
}

; Vector fmul by 2^-4 (v2f64)
define <2 x double> @testvmul_v2f64_mul(<2 x i64> %in) local_unnamed_addr #0 {
; CHECK-LABEL: testvmul_v2f64_mul:
; CHECK:       // %bb.0:
; CHECK-NEXT:    scvtf.2d        v0, v0, #4
; CHECK-NEXT:    ret
  %vcvt.i = sitofp <2 x i64> %in to <2 x double>
  %div.i = fmul <2 x double> %vcvt.i, splat (double 6.250000e-02) ; 0.0625 is 2^-4 in double-precision
  ret <2 x double> %div.i
}