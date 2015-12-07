; RUN: llc -mtriple=x86_64-unknown-unknown -mattr=sse2 < %s | FileCheck %s
; RUN: llc -mtriple=x86_64-unknown-unknown -mattr=avx < %s | FileCheck %s

declare float @fminf(float, float)
declare double @fmin(double, double)
declare x86_fp80 @fminl(x86_fp80, x86_fp80)
declare float @llvm.minnum.f32(float, float)
declare double @llvm.minnum.f64(double, double)
declare x86_fp80 @llvm.minnum.f80(x86_fp80, x86_fp80)

declare <2 x float> @llvm.minnum.v2f32(<2 x float>, <2 x float>)
declare <4 x float> @llvm.minnum.v4f32(<4 x float>, <4 x float>)
declare <2 x double> @llvm.minnum.v2f64(<2 x double>, <2 x double>)
declare <4 x double> @llvm.minnum.v4f64(<4 x double>, <4 x double>)
declare <8 x double> @llvm.minnum.v8f64(<8 x double>, <8 x double>)

; CHECK-LABEL: @test_fminf
; CHECK: jmp fminf
define float @test_fminf(float %x, float %y) {
  %z = call float @fminf(float %x, float %y) readnone
  ret float %z
}

; CHECK-LABEL: @test_fmin
; CHECK: jmp fmin
define double @test_fmin(double %x, double %y) {
  %z = call double @fmin(double %x, double %y) readnone
  ret double %z
}

; CHECK-LABEL: @test_fminl
; CHECK: callq fminl
define x86_fp80 @test_fminl(x86_fp80 %x, x86_fp80 %y) {
  %z = call x86_fp80 @fminl(x86_fp80 %x, x86_fp80 %y) readnone
  ret x86_fp80 %z
}

; CHECK-LABEL: @test_intrinsic_fminf
; CHECK: jmp fminf
define float @test_intrinsic_fminf(float %x, float %y) {
  %z = call float @llvm.minnum.f32(float %x, float %y) readnone
  ret float %z
}

; CHECK-LABEL: @test_intrinsic_fmin
; CHECK: jmp fmin
define double @test_intrinsic_fmin(double %x, double %y) {
  %z = call double @llvm.minnum.f64(double %x, double %y) readnone
  ret double %z
}

; CHECK-LABEL: @test_intrinsic_fminl
; CHECK: callq fminl
define x86_fp80 @test_intrinsic_fminl(x86_fp80 %x, x86_fp80 %y) {
  %z = call x86_fp80 @llvm.minnum.f80(x86_fp80 %x, x86_fp80 %y) readnone
  ret x86_fp80 %z
}

; CHECK-LABEL: @test_intrinsic_fmin_v2f32
; CHECK: callq fminf
; CHECK: callq fminf
define <2 x float> @test_intrinsic_fmin_v2f32(<2 x float> %x, <2 x float> %y) {
  %z = call <2 x float> @llvm.minnum.v2f32(<2 x float> %x, <2 x float> %y) readnone
  ret <2 x float> %z
}

; CHECK-LABEL: @test_intrinsic_fmin_v4f32
; CHECK: callq fminf
; CHECK: callq fminf
; CHECK: callq fminf
; CHECK: callq fminf
define <4 x float> @test_intrinsic_fmin_v4f32(<4 x float> %x, <4 x float> %y) {
  %z = call <4 x float> @llvm.minnum.v4f32(<4 x float> %x, <4 x float> %y) readnone
  ret <4 x float> %z
}

; CHECK-LABEL: @test_intrinsic_fmin_v2f64
; CHECK: callq fmin
; CHECK: callq fmin
define <2 x double> @test_intrinsic_fmin_v2f64(<2 x double> %x, <2 x double> %y) {
  %z = call <2 x double> @llvm.minnum.v2f64(<2 x double> %x, <2 x double> %y) readnone
  ret <2 x double> %z
}

; CHECK-LABEL: @test_intrinsic_fmin_v4f64
; CHECK: callq fmin
; CHECK: callq fmin
; CHECK: callq fmin
; CHECK: callq fmin
define <4 x double> @test_intrinsic_fmin_v4f64(<4 x double> %x, <4 x double> %y) {
  %z = call <4 x double> @llvm.minnum.v4f64(<4 x double> %x, <4 x double> %y) readnone
  ret <4 x double> %z
}

; CHECK-LABEL: @test_intrinsic_fmin_v8f64
; CHECK: callq fmin
; CHECK: callq fmin
; CHECK: callq fmin
; CHECK: callq fmin
; CHECK: callq fmin
; CHECK: callq fmin
; CHECK: callq fmin
; CHECK: callq fmin
define <8 x double> @test_intrinsic_fmin_v8f64(<8 x double> %x, <8 x double> %y) {
  %z = call <8 x double> @llvm.minnum.v8f64(<8 x double> %x, <8 x double> %y) readnone
  ret <8 x double> %z
}
