; RUN: opt %loadPolly -polly-canonicalize -polly-ast -polly-ast-detect-parallel -analyze < %s | FileCheck %s
; RUN: opt %loadPolly -polyhedral-info -polly-check-parallel -analyze < %s | FileCheck %s -check-prefix=PINFO
;
;        void f(int *restrict A, int *restrict sum) {
; CHECK:   #pragma minimal dependence distance: 1
; PINFO:    for.cond: Loop is not parallel.
;          for (int j = 0; j < 1024; j++)
; CHECK:      #pragma minimal dependence distance: 1
; PINFO-NEXT: for.cond1: Loop is not parallel.
;             for (int i = j; i < 1024; i++)
;               A[i - 3] = A[j] * 2 + A[j] + 2;
;        }
;
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-n32-S64"

define void @f(i32* noalias %A, i32* noalias %sum) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc7, %entry
  %j.0 = phi i32 [ 0, %entry ], [ %inc8, %for.inc7 ]
  %exitcond1 = icmp ne i32 %j.0, 1024
  br i1 %exitcond1, label %for.body, label %for.end9

for.body:                                         ; preds = %for.cond
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc, %for.body
  %i.0 = phi i32 [ %j.0, %for.body ], [ %inc, %for.inc ]
  %exitcond = icmp ne i32 %i.0, 1024
  br i1 %exitcond, label %for.body3, label %for.end

for.body3:                                        ; preds = %for.cond1
  %arrayidx = getelementptr inbounds i32, i32* %A, i32 %j.0
  %tmp = load i32, i32* %arrayidx, align 4
  %add = mul nsw i32 %tmp, 3
  %add5 = add nsw i32 %add, 2
  %sub = add nsw i32 %i.0, -3
  %arrayidx6 = getelementptr inbounds i32, i32* %A, i32 %sub
  store i32 %add5, i32* %arrayidx6, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body3
  %inc = add nsw i32 %i.0, 1
  br label %for.cond1

for.end:                                          ; preds = %for.cond1
  br label %for.inc7

for.inc7:                                         ; preds = %for.end
  %inc8 = add nsw i32 %j.0, 1
  br label %for.cond

for.end9:                                         ; preds = %for.cond
  ret void
}
