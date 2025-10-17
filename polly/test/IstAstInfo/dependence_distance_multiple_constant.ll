; RUN: opt %loadPolly -basic-aa -polly-stmt-granularity=bb -polly-print-ast -polly-ast-detect-parallel -disable-output < %s | FileCheck %s
;
;        void f(int *restrict A, int *restrict B, int N) {
; CHECK:   #pragma minimal dependence distance: 5
;          for (int i = 0; i < N; i++) {
;            A[i + 7] = A[i] + 1;
;            B[i + 5] = B[i] + 1;
;          }
;        }
;
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-n32-S64"

define void @f(ptr noalias %A, ptr noalias %B, i32 %N) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %cmp = icmp slt i32 %i.0, %N
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %i.0
  %tmp = load i32, ptr %arrayidx, align 4
  %add = add nsw i32 %tmp, 1
  %add1 = add nsw i32 %i.0, 7
  %arrayidx2 = getelementptr inbounds i32, ptr %A, i32 %add1
  store i32 %add, ptr %arrayidx2, align 4
  %arrayidx3 = getelementptr inbounds i32, ptr %B, i32 %i.0
  %tmp1 = load i32, ptr %arrayidx3, align 4
  %add4 = add nsw i32 %tmp1, 1
  %add5 = add nsw i32 %i.0, 5
  %arrayidx6 = getelementptr inbounds i32, ptr %B, i32 %add5
  store i32 %add4, ptr %arrayidx6, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
