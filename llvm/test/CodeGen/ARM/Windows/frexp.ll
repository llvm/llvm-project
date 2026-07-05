; RUN: llc -mtriple thumbv7--windows-itanium < %s | FileCheck %s

; CHECK-LABEL: test_frexp_f16_i32:
; CHECK: bl __gnu_h2f_ieee
; CHECK: vcvt.f64.f32 d0, s0
; CHECK: bl frexp
; CHECK: vcvt.f32.f64 s0, d0
; CHECK: bl __gnu_f2h_ieee
define { half, i32 } @test_frexp_f16_i32(half %a) {
  %result = call { half, i32 } @llvm.frexp.f16.i32(half %a)
  ret { half, i32 } %result
}

; CHECK-LABEL: test_frexp_f32_i32:
; CHECK: vcvt.f64.f32
; CHECK: bl frexp
; CHECK: vcvt.f32.f64	s0, d0
define { float, i32 } @test_frexp_f32_i32(float %a) {
  %result = call { float, i32 } @llvm.frexp.f32.i32(float %a)
  ret { float, i32 } %result
}

; CHECK-LABEL: test_frexp_f64_i32:
; CHECK: bl frexp
define { double, i32 } @test_frexp_f64_i32(double %a) {
  %result = call { double, i32 } @llvm.frexp.f64.i32(double %a)
  ret { double, i32 } %result
}
