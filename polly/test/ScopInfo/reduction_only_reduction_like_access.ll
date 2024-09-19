; RUN: opt %loadNPMPolly '-passes=print<polly-function-scops>' -disable-output < %s 2>&1 | FileCheck %s
;
; CHECK: Reduction Type: +
;
; void f(int *sum) {
;   for (int i = 0; i < 100; i++)
;     sum[i] = sum[99-i] + i;
; }
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-n32-S64"

define void @f(ptr %sum) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %exitcond = icmp ne i32 %i.0, 100
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %sub = sub nsw i32 99, %i.0
  %arrayidx = getelementptr inbounds i32, ptr %sum, i32 %sub
  %tmp = load i32, ptr %arrayidx, align 4
  %add = add nsw i32 %tmp, %i.0
  %arrayidx1 = getelementptr inbounds i32, ptr %sum, i32 %i.0
  store i32 %add, ptr %arrayidx1, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
