; RUN: opt < %s -instcombine -S -mtriple "i386-pc-linux" | FileCheck -check-prefix=DO-SIMPLIFY %s
; RUN: opt < %s -instcombine -S -mtriple "i386-pc-win32" | FileCheck -check-prefix=DONT-SIMPLIFY %s
; RUN: opt < %s -instcombine -S -mtriple "x86_64-pc-win32" | FileCheck -check-prefix=C89-SIMPLIFY %s
; RUN: opt < %s -instcombine -S -mtriple "i386-pc-mingw32" | FileCheck -check-prefix=DO-SIMPLIFY %s
; RUN: opt < %s -instcombine -S -mtriple "x86_64-pc-mingw32" | FileCheck -check-prefix=DO-SIMPLIFY %s
; RUN: opt < %s -instcombine -S -mtriple "sparc-sun-solaris" | FileCheck -check-prefix=DO-SIMPLIFY %s

; DO-SIMPLIFY: call float @llvm.floor.f32(
; DO-SIMPLIFY: call float @llvm.ceil.f32(
; DO-SIMPLIFY: call float @llvm.round.f32(
; DO-SIMPLIFY: call float @llvm.nearbyint.f32(
; DO-SIMPLIFY: call float @llvm.trunc.f32(
; DO-SIMPLIFY: call float @llvm.fabs.f32(
; DO-SIMPLIFY: call fast float @llvm.fabs.f32(

; C89-SIMPLIFY: call float @llvm.floor.f32(
; C89-SIMPLIFY: call float @llvm.ceil.f32(
; C89-SIMPLIFY: call double @round(
; C89-SIMPLIFY: call double @nearbyint(

; DONT-SIMPLIFY: call float @llvm.floor.f32(
; DONT-SIMPLIFY: call float @llvm.ceil.f32(
; DONT-SIMPLIFY: call double @round(
; DONT-SIMPLIFY: call double @nearbyint(
; DONT-SIMPLIFY: call double @trunc(

; This is replaced with the intrinsic, which does the right thing on
; all platforms.
; DONT-SIMPLIFY: call float @llvm.fabs.f32(

declare double @floor(double)
declare double @ceil(double)
declare double @round(double)
declare double @nearbyint(double)
declare double @trunc(double)
declare double @fabs(double)
declare double @llvm.fabs.f64(double)

define float @test_floor(float %C) {
  %D = fpext float %C to double
  ; --> floorf
  %E = call double @floor(double %D)
  %F = fptrunc double %E to float
  ret float %F
}

define float @test_ceil(float %C) {
  %D = fpext float %C to double
  ; --> ceilf
  %E = call double @ceil(double %D)
  %F = fptrunc double %E to float
  ret float %F
}

define float @test_round(float %C) {
  %D = fpext float %C to double
  ; --> roundf
  %E = call double @round(double %D)
  %F = fptrunc double %E to float
  ret float %F
}

define float @test_nearbyint(float %C) {
  %D = fpext float %C to double
  ; --> nearbyintf
  %E = call double @nearbyint(double %D)
  %F = fptrunc double %E to float
  ret float %F
}

define float @test_trunc(float %C) {
  %D = fpext float %C to double
  ; --> truncf
  %E = call double @trunc(double %D)
  %F = fptrunc double %E to float
  ret float %F
}

define float @test_fabs(float %C) {
  %D = fpext float %C to double
  ; --> fabsf
  %E = call double @fabs(double %D)
  %F = fptrunc double %E to float
  ret float %F
}

; Make sure fast math flags are preserved
define float @test_fabs_fast(float %C) {
  %D = fpext float %C to double
  ; --> fabsf
  %E = call fast double @fabs(double %D)
  %F = fptrunc double %E to float
  ret float %F
}
