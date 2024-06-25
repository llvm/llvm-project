; RUN: opt %loadNPMPolly '-passes=print<polly-function-scops>' -disable-output < %s 2>&1 | FileCheck %s
;
; CHECK: Reduction Type: +
;
; void f(int *sum) {
;   for (int i = 0; i <= 100; i++)
;     sum += 3;
; }
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-n32-S64"

define void @f(ptr %sum) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.cond, %entry
  %i1.0 = phi i32 [ 0, %entry ], [ %inc, %for.cond ]
  %sum.reload = load i32, ptr %sum
  %add = add nsw i32 %sum.reload, 3
  %inc = add nsw i32 %i1.0, 1
  store i32 %add, ptr %sum
  %cmp = icmp slt i32 %i1.0, 100
  br i1 %cmp, label %for.cond, label %for.end

for.end:                                          ; preds = %for.cond
  ret void
}
