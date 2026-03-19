; RUN: llc -mtriple=xcore-unknown-unknown < %s | FileCheck %s

; CHECK-LABEL: exp10_f16:
; CHECK: bl __extendhfsf2
; CHECK: bl exp10f
; CHECK: bl __truncsfhf2
define half @exp10_f16(half %x) #0 {
  %r = call half @llvm.exp10.f16(half %x)
  ret half %r
}

; CHECK-LABEL: exp10_v2f16:
; CHECK: bl __extendhfsf2
; CHECK: bl exp10f
; CHECK: bl __truncsfhf2
; CHECK: bl __extendhfsf2
; CHECK: bl exp10f
; CHECK: bl __truncsfhf2
define <2 x half> @exp10_v2f16(<2 x half> %x) #0 {
  %r = call <2 x half> @llvm.exp10.v2f16(<2 x half> %x)
  ret <2 x half> %r
}

; CHECK-LABEL: exp10_f32:
; CHECK: bl exp10f
define float @exp10_f32(float %x) #0 {
  %r = call float @llvm.exp10.f32(float %x)
  ret float %r
}

; CHECK-LABEL: exp10_v2f32:
; CHECK: bl exp10f
; CHECK: bl exp10f
define <2 x float> @exp10_v2f32(<2 x float> %x) #0 {
  %r = call <2 x float> @llvm.exp10.v2f32(<2 x float> %x)
  ret <2 x float> %r
}

; CHECK-LABEL: exp10_f64:
; CHECK: bl exp10
define double @exp10_f64(double %x) #0 {
  %r = call double @llvm.exp10.f64(double %x)
  ret double %r
}

; CHECK-LABEL: exp10_v2f64:
; CHECK: bl exp10
; CHECK: bl exp10
define <2 x double> @exp10_v2f64(<2 x double> %x) #0 {
  %r = call <2 x double> @llvm.exp10.v2f64(<2 x double> %x)
  ret <2 x double> %r
}

; CHECK-LABEL: exp10_f128:
; CHECK: bl exp10l
define fp128 @exp10_f128(fp128 %x) #0 {
  %r = call fp128 @llvm.exp10.f128(fp128 %x)
  ret fp128 %r
}

; CHECK-LABEL: exp10_v2f128:
; CHECK: bl exp10l
; CHECK: bl exp10l
define <2 x fp128> @exp10_v2f128(<2 x fp128> %x) #0 {
  %r = call <2 x fp128> @llvm.exp10.v2f128(<2 x fp128> %x)
  ret <2 x fp128> %r
}

attributes #0 = { nounwind }
