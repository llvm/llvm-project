; RUN: opt < %s -O3 -S -mtriple=x86_64-apple-macosx10.8.0 -mcpu=corei7-avx | FileCheck %s --check-prefix=SLP
; RUN: opt < %s -O3 -disable-slp-vectorization -S -mtriple=x86_64-apple-macosx10.8.0 -mcpu=corei7-avx | FileCheck %s --check-prefix=NOSLP

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

; Make sure we can disable slp vectorization in opt.

; SLP-LABEL: test1
; SLP: store <2 x double>

; NOSLP-LABEL: test1
; NOSLP-NOT: store <2 x double>


define void @test1(double* %a, double* %b, double* %c) {
entry:
  %i0 = load double* %a, align 8
  %i1 = load double* %b, align 8
  %mul = fmul double %i0, %i1
  %arrayidx3 = getelementptr inbounds double* %a, i64 1
  %i3 = load double* %arrayidx3, align 8
  %arrayidx4 = getelementptr inbounds double* %b, i64 1
  %i4 = load double* %arrayidx4, align 8
  %mul5 = fmul double %i3, %i4
  store double %mul, double* %c, align 8
  %arrayidx5 = getelementptr inbounds double* %c, i64 1
  store double %mul5, double* %arrayidx5, align 8
  ret void
}
