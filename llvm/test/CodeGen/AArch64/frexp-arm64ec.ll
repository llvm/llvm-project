; RUN: llc -mtriple=arm64ec-windows-msvc < %s | FileCheck -check-prefix=ARM64EC %s

; Separate from llvm-frexp.ll test because this errors on half cases

; ARM64EC-LABEL: test_frexp_f16_i32
; ARM64EC: fcvt d0, h0
; ARM64EC: bl "#frexp"
; ARM64EC: fcvt h0, d0
define { half, i32 } @test_frexp_f16_i32(half %a) {
  %result = call { half, i32 } @llvm.frexp.f16.i32(half %a)
  ret { half, i32 } %result
}

; ARM64EC-LABEL: test_frexp_f32_i32
; ARM64EC: fcvt d0, s0
; ARM64EC: bl "#frexp"
; ARM64EC: fcvt s0, d0
define { float, i32 } @test_frexp_f32_i32(float %a) {
  %result = call { float, i32 } @llvm.frexp.f32.i32(float %a)
  ret { float, i32 } %result
}

; ARM64EC-LABEL: test_frexp_f64_i32
; ARM64EC: bl "#frexp"
define { double, i32 } @test_frexp_f64_i32(double %a) {
  %result = call { double, i32 } @llvm.frexp.f64.i32(double %a)
  ret { double, i32 } %result
}
