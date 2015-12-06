; RUN: llc  -march=x86 -mtriple=i386-linux-gnu  < %s | FileCheck %s

declare float @fmaxf(float, float)
declare double @fmax(double, double)
declare x86_fp80 @fmaxl(x86_fp80, x86_fp80)
declare float @llvm.maxnum.f32(float, float)
declare double @llvm.maxnum.f64(double, double)
declare x86_fp80 @llvm.maxnum.f80(x86_fp80, x86_fp80)

declare <2 x float> @llvm.maxnum.v2f32(<2 x float>, <2 x float>)
declare <4 x float> @llvm.maxnum.v4f32(<4 x float>, <4 x float>)
declare <2 x double> @llvm.maxnum.v2f64(<2 x double>, <2 x double>)
declare <8 x double> @llvm.maxnum.v8f64(<8 x double>, <8 x double>)


; CHECK-LABEL: @test_fmaxf
; CHECK: calll fmaxf
define float @test_fmaxf(float %x, float %y) {
  %z = call float @fmaxf(float %x, float %y) readnone
  ret float %z
}

; CHECK-LABEL: @test_fmax
; CHECK: calll fmax
define double @test_fmax(double %x, double %y) {
  %z = call double @fmax(double %x, double %y) readnone
  ret double %z
}

; CHECK-LABEL: @test_fmaxl
; CHECK: calll fmaxl
define x86_fp80 @test_fmaxl(x86_fp80 %x, x86_fp80 %y) {
  %z = call x86_fp80 @fmaxl(x86_fp80 %x, x86_fp80 %y) readnone
  ret x86_fp80 %z
}

; CHECK-LABEL: @test_intrinsic_fmaxf
; CHECK: calll fmaxf
define float @test_intrinsic_fmaxf(float %x, float %y) {
  %z = call float @llvm.maxnum.f32(float %x, float %y) readnone
  ret float %z
}

; CHECK-LABEL: @test_intrinsic_fmax
; CHECK: calll fmax
define double @test_intrinsic_fmax(double %x, double %y) {
  %z = call double @llvm.maxnum.f64(double %x, double %y) readnone
  ret double %z
}

; CHECK-LABEL: @test_intrinsic_fmaxl
; CHECK: calll fmaxl
define x86_fp80 @test_intrinsic_fmaxl(x86_fp80 %x, x86_fp80 %y) {
  %z = call x86_fp80 @llvm.maxnum.f80(x86_fp80 %x, x86_fp80 %y) readnone
  ret x86_fp80 %z
}

; CHECK-LABEL: @test_intrinsic_fmax_v2f32
; CHECK: calll fmaxf
; CHECK: calll fmaxf
define <2 x float> @test_intrinsic_fmax_v2f32(<2 x float> %x, <2 x float> %y) {
  %z = call <2 x float> @llvm.maxnum.v2f32(<2 x float> %x, <2 x float> %y) readnone
  ret <2 x float> %z
}

; CHECK-LABEL: @test_intrinsic_fmax_v4f32
; CHECK: calll fmaxf
; CHECK: calll fmaxf
; CHECK: calll fmaxf
; CHECK: calll fmaxf
define <4 x float> @test_intrinsic_fmax_v4f32(<4 x float> %x, <4 x float> %y) {
  %z = call <4 x float> @llvm.maxnum.v4f32(<4 x float> %x, <4 x float> %y) readnone
  ret <4 x float> %z
}

; CHECK-LABEL: @test_intrinsic_fmax_v2f64
; CHECK: calll fmax
; CHECK: calll fmax
define <2 x double> @test_intrinsic_fmax_v2f64(<2 x double> %x, <2 x double> %y) {
  %z = call <2 x double> @llvm.maxnum.v2f64(<2 x double> %x, <2 x double> %y) readnone
  ret <2 x double> %z
}

; CHECK-LABEL: @test_intrinsic_fmax_v8f64
; CHECK: calll fmax
; CHECK: calll fmax
; CHECK: calll fmax
; CHECK: calll fmax
; CHECK: calll fmax
; CHECK: calll fmax
; CHECK: calll fmax
; CHECK: calll fmax
define <8 x double> @test_intrinsic_fmax_v8f64(<8 x double> %x, <8 x double> %y) {
  %z = call <8 x double> @llvm.maxnum.v8f64(<8 x double> %x, <8 x double> %y) readnone
  ret <8 x double> %z
}

