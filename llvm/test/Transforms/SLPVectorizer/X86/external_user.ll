; RUN: opt < %s -basicaa -slp-vectorizer -dce -S -mtriple=x86_64-apple-macosx10.8.0 -mcpu=corei7-avx | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

; double foo(double * restrict b,  double * restrict a, int n, int m) {
;   double r=a[1];
;   double g=a[0];
;   double x;
;   for (int i=0; i < 100; i++) {
;     r += 10;
;     g += 10;
;     r *= 4;
;     g *= 4;
;     x = g; <----- external user!
;     r += 4;
;     g += 4;
;   }
;   b[0] = g;
;   b[1] = r;
;
;   return x; <-- must extract here!
; }

;CHECK: ext_user
;CHECK: phi <2 x double>
;CHECK: fadd <2 x double>
;CHECK: fmul <2 x double>
;CHECK: br
;CHECK: store <2 x double>
;CHECK: extractelement <2 x double>
;CHECK: ret double

define double @ext_user(double* noalias nocapture %B, double* noalias nocapture %A, i32 %n, i32 %m) {
entry:
  %arrayidx = getelementptr inbounds double* %A, i64 1
  %0 = load double* %arrayidx, align 8
  %1 = load double* %A, align 8
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i.020 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %G.019 = phi double [ %1, %entry ], [ %add5, %for.body ]
  %R.018 = phi double [ %0, %entry ], [ %add4, %for.body ]
  %add = fadd double %R.018, 1.000000e+01
  %add2 = fadd double %G.019, 1.000000e+01
  %mul = fmul double %add, 4.000000e+00
  %mul3 = fmul double %add2, 4.000000e+00
  %add4 = fadd double %mul, 4.000000e+00
  %add5 = fadd double %mul3, 4.000000e+00
  %inc = add nsw i32 %i.020, 1
  %exitcond = icmp eq i32 %inc, 100
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  store double %add5, double* %B, align 8
  %arrayidx7 = getelementptr inbounds double* %B, i64 1
  store double %add4, double* %arrayidx7, align 8
  ret double %mul3
}

; A need-to-gather entry cannot be an external use of the scalar element.
; Instead the insertelement instructions of the need-to-gather entry are the
; external users.
; This test would assert because we would keep the scalar fpext and fadd alive.
; PR18129

; CHECK-LABEL: needtogather
define i32 @needtogather(double *noalias %a, i32 *noalias %b,  float * noalias %c,
                i32 * noalias %d) {
entry:
  %0 = load i32* %d, align 4
  %conv = sitofp i32 %0 to float
  %1 = load float* %c
  %sub = fsub float 0.000000e+00, %1
  %mul = fmul float %sub, 0.000000e+00
  %add = fadd float %conv, %mul
  %conv1 = fpext float %add to double
  %sub3 = fsub float 1.000000e+00, %1
  %mul4 = fmul float %sub3, 0.000000e+00
  %add5 = fadd float %conv, %mul4
  %conv6 = fpext float %add5 to double
  %tobool = fcmp une float %add, 0.000000e+00
  br i1 %tobool, label %if.then, label %if.end

if.then:
  br label %if.end

if.end:
  %storemerge = phi double [ %conv6, %if.then ], [ %conv1, %entry ]
  %e.0 = phi double [ %conv1, %if.then ], [ %conv6, %entry ]
  store double %storemerge, double* %a, align 8
  %conv7 = fptosi double %e.0 to i32
  store i32 %conv7, i32* %b, align 4
  ret i32 undef
}
