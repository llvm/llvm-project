; RUN: opt %loadPolly -polly-print-scops -disable-output < %s | FileCheck %s
;
; CHECK: Function: f_no_fast_math
; CHECK: Reduction Type: NONE
; CHECK: Function: f_fast_math
; CHECK: Reduction Type: +
;
; void f(float *sum) {
;   for (int i = 0; i < 100; i++)
;     *sum += 3.41 * i;
; }
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-n32-S64"

define void @f_no_fast_math(ptr %sum) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %exitcond = icmp ne i32 %i.0, 100
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %conv = sitofp i32 %i.0 to float
  %pi = fptrunc double 3.41 to float
  %mul = fmul float %conv, %pi
  %tmp = load float, ptr %sum, align 4
  %add = fadd float %tmp, %mul
  store float %add, ptr %sum, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

define void @f_fast_math(ptr %sum) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %exitcond = icmp ne i32 %i.0, 100
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %conv = sitofp i32 %i.0 to float
  %pi = fptrunc double 3.41 to float
  %mul = fmul fast float %conv, %pi
  %tmp = load float, ptr %sum, align 4
  %add = fadd fast float %tmp, %mul
  store float %add, ptr %sum, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
