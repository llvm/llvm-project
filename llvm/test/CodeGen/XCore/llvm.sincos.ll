; RUN: llc -mtriple=xcore-unknown-unknown < %s | FileCheck -check-prefixes=CHECK,OTHER %s
; RUN: llc -mtriple=xcore-none-linux < %s | FileCheck -check-prefixes=CHECK,OTHER %s
; RUN: llc -mtriple=xcore-none-linux-gnu < %s | FileCheck -check-prefixes=CHECK,GNU %s

; CHECK-LABEL: test_sincos_f16:
; CHECK: bl __extendhfsf2

; OTHER: bl cosf
; OTHER: bl sinf

; GNU: bl sincosf

; CHECK: bl __truncsfhf2
; CHECK: bl __truncsfhf2
define { half, half } @test_sincos_f16(half %a) nounwind {
  %result = call { half, half } @llvm.sincos.f16(half %a)
  ret { half, half } %result
}

; CHECK-LABEL: test_sincos_v2f16:
; CHECK: bl __extendhfsf2
; CHECK: bl __extendhfsf2

; OTHER: bl cosf
; OTHER: bl cosf
; OTHER: bl sinf
; OTHER: bl sinf

; GNU: bl sincosf
; GNU: bl sincosf

; CHECK: bl __truncsfhf2
; CHECK: bl __truncsfhf2
define { <2 x half>, <2 x half> } @test_sincos_v2f16(<2 x half> %a) nounwind {
  %result = call { <2 x half>, <2 x half> } @llvm.sincos.v2f16(<2 x half> %a)
  ret { <2 x half>, <2 x half> } %result
}

; CHECK-LABEL: test_sincos_f32:
; OTHER: bl sinf
; OTHER: bl cosf

; GNU: bl sincosf
define { float, float } @test_sincos_f32(float %a) nounwind {
  %result = call { float, float } @llvm.sincos.f32(float %a)
  ret { float, float } %result
}

; CHECK-LABEL: test_sincos_v2f32:
; OTHER: bl sinf
; OTHER: bl sinf
; OTHER: bl cosf
; OTHER: bl cosf

; GNU: bl sincosf
; GNU: bl sincosf
define { <2 x float>, <2 x float> } @test_sincos_v2f32(<2 x float> %a) nounwind {
  %result = call { <2 x float>, <2 x float> } @llvm.sincos.v2f32(<2 x float> %a)
  ret { <2 x float>, <2 x float> } %result
}

; CHECK-LABEL: test_sincos_f64:
; OTHER: bl sin
; OTHER: bl cos

; GNU: bl sincos
define { double, double } @test_sincos_f64(double %a) nounwind {
  %result = call { double, double } @llvm.sincos.f64(double %a)
  ret { double, double } %result
}

; CHECK-LABEL: test_sincos_v2f64:
; OTHER: bl sin
; OTHER: bl sin
; OTHER: bl cos
; OTHER: bl cos

; GNU: bl sincos
; GNU: bl sincos
define { <2 x double>, <2 x double> } @test_sincos_v2f64(<2 x double> %a) nounwind {
  %result = call { <2 x double>, <2 x double> } @llvm.sincos.v2f64(<2 x double> %a)
  ret { <2 x double>, <2 x double> } %result
}

; CHECK-LABEL: test_sincos_f128:
; OTHER: bl sinl
; OTHER: bl cosl

; GNU: bl sincosl
define { fp128, fp128 } @test_sincos_f128(fp128 %a) nounwind {
  %result = call { fp128, fp128 } @llvm.sincos.f128(fp128 %a)
  ret { fp128, fp128 } %result
}

; CHECK-LABEL: test_sincos_v2f128:
; OTHER: bl sinl
; OTHER: bl cosl
; OTHER: bl cosl
; OTHER: bl sinl

; GNU: bl sincosl
; GNU: bl sincosl
define { <2 x fp128>, <2 x fp128> } @test_sincos_v2f128(<2 x fp128> %a) nounwind {
  %result = call { <2 x fp128>, <2 x fp128> } @llvm.sincos.v2f128(<2 x fp128> %a)
  ret { <2 x fp128>, <2 x fp128> } %result
}
