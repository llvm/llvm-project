; RUN: opt %loadPolly -polly-print-scops -disable-output < %s | FileCheck %s
; RUN: opt %loadPolly -polly-codegen -disable-output < %s
;
; void or(float *A, long n, long m) {
;   for (long i = 0; i < 100; i++) {
;     if (i < n || i < m || i > p)
;       A[i] += i;
;   }
; }
;
; CHECK: Function: or
; CHECK:   Stmt_if_then
; CHECK:     Domain :=
; CHECK:       [n, m, p] -> { Stmt_if_then[i0] : 0 <= i0 <= 99 and (i0 > p or i0 < m or i0 < n) };
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; Function Attrs: nounwind uwtable
define void @or(ptr nocapture %A, i64 %n, i64 %m, i64 %p) #0 {
entry:
  br label %for.body

for.body:                                         ; preds = %for.inc, %entry
  %i.03 = phi i64 [ 0, %entry ], [ %inc, %for.inc ]
  %cmp1 = icmp slt i64 %i.03, %n
  %cmp2 = icmp slt i64 %i.03, %m
  %cmp3 = icmp sgt i64 %i.03, %p
  %or.tmp = or i1 %cmp1, %cmp2
  %or.cond = or i1 %or.tmp, %cmp3
  br i1 %or.cond, label %if.then, label %for.inc

if.then:                                          ; preds = %for.body
  %conv = sitofp i64 %i.03 to float
  %arrayidx = getelementptr inbounds float, ptr %A, i64 %i.03
  %0 = load float, ptr %arrayidx, align 4
  %add = fadd float %conv, %0
  store float %add, ptr %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %if.then, %for.body
  %inc = add nuw nsw i64 %i.03, 1
  %exitcond = icmp eq i64 %inc, 100
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.inc
  ret void
}
