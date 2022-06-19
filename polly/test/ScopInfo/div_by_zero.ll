; RUN: opt %loadPolly -polly-print-scops -disable-output < %s | FileCheck %s
;
;    void f(int *A, int N) {
;      for (int i = 0; i < N; i++)
;        A[i / 0]++;
;    }
;
; CHECK-NOT: Statement
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(i32* %A, i32 %N) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc1, %for.inc ]
  %cmp = icmp slt i32 %i.0, %N
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %div = sdiv i32 %i.0, 0
  %idxprom = sext i32 %div to i64
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %idxprom
  %tmp = load i32, i32* %arrayidx, align 4
  %inc = add nsw i32 %tmp, 1
  store i32 %inc, i32* %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %inc1 = add nuw nsw i32 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
