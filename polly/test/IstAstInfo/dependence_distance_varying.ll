; RUN: opt %loadPolly -polly-print-ast -polly-ast-detect-parallel -disable-output < %s | FileCheck %s
; RUN: opt %loadPolly -print-polyhedral-info -polly-check-parallel -disable-output < %s | FileCheck %s -check-prefix=PINFO
;
;         void f(int *A, int N) {
; CHECK:    #pragma minimal dependence distance: -(N % 2) + 2
; PINFO:    for.cond: Loop is not parallel.
;           for (int i = 0; i < N; i++)
;             A[i] = A[N - i] + 1;
;         }
;
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-n32-S64"

define void @f(ptr %A, i32 %N) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %cmp = icmp slt i32 %i.0, %N
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %sub = sub nsw i32 %N, %i.0
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %sub
  %tmp = load i32, ptr %arrayidx, align 4
  %add = add nsw i32 %tmp, 1
  %arrayidx1 = getelementptr inbounds i32, ptr %A, i32 %i.0
  store i32 %add, ptr %arrayidx1, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
