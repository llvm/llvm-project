; RUN: opt < %s -basicaa -slp-vectorizer -dce -S -mtriple=x86_64-apple-macosx10.8.0 -mcpu=corei7-avx | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

; int foo(double * restrict A, double * restrict B, double G) {
;   A[0] = (B[10] ? G : 1);
;   A[1] = (B[11] ? G : 1);
; }

;CHECK-LABEL: @foo(
;CHECK: load <2 x double>
;CHECK: fcmp une <2 x double>
;CHECK: select <2 x i1>
;CHECK: store <2 x double>
;CHECK: ret i32 undef
define i32 @foo(double* noalias nocapture %A, double* noalias nocapture %B, double %G) {
entry:
  %arrayidx = getelementptr inbounds double, double* %B, i64 10
  %0 = load double, double* %arrayidx, align 8
  %tobool = fcmp une double %0, 0.000000e+00
  %cond = select i1 %tobool, double %G, double 1.000000e+00
  store double %cond, double* %A, align 8
  %arrayidx2 = getelementptr inbounds double, double* %B, i64 11
  %1 = load double, double* %arrayidx2, align 8
  %tobool3 = fcmp une double %1, 0.000000e+00
  %cond7 = select i1 %tobool3, double %G, double 1.000000e+00
  %arrayidx8 = getelementptr inbounds double, double* %A, i64 1
  store double %cond7, double* %arrayidx8, align 8
  ret i32 undef
}

