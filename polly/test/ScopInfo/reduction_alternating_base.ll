; RUN: opt %loadNPMPolly '-passes=print<polly-function-scops>' -disable-output < %s 2>&1 | FileCheck %s
;
;
;    void f(int *A) {
;      for (int i = 0; i < 1024; i++)
;        A[i % 2] += i;
;    }
;
; Verify that we detect the reduction on A
;
; CHECK: ReadAccess := [Reduction Type: +] [Scalar: 0]
; CHECK: MustWriteAccess :=  [Reduction Type: +] [Scalar: 0]
;
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-n32-S64"

define void @f(ptr %A) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %exitcond = icmp ne i32 %i.0, 1024
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %rem = srem i32 %i.0, 2
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %rem
  %tmp = load i32, ptr %arrayidx, align 4
  %add = add nsw i32 %tmp, %i.0
  store i32 %add, ptr %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
