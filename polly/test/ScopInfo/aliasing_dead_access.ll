; RUN: opt %loadPolly -polly-print-scops -disable-output < %s | FileCheck %s
;
; Check that we do not create a SCoP if there is no statement executed.
;
; CHECK-NOT: Context
;
;    void jd(int *A, int *B) {
;      for (int i = 0; i < 1024; i++)
;        for (int j = i; j < 0; j++)
;          A[i] = B[i];
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @jd(i32* %A, i32* %B) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc6, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc6 ], [ 0, %entry ]
  %exitcond = icmp ne i64 %indvars.iv, 1024
  br i1 %exitcond, label %for.body, label %for.end8

for.body:                                         ; preds = %for.cond
  %tmp = trunc i64 %indvars.iv to i32
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc, %for.body
  %j.0 = phi i32 [ %tmp, %for.body ], [ %inc, %for.inc ]
  %cmp2 = icmp slt i32 %j.0, 0
  br i1 %cmp2, label %for.body3, label %for.end

for.body3:                                        ; preds = %for.cond1
  %arrayidx = getelementptr inbounds i32, i32* %B, i64 %indvars.iv
  %tmp1 = load i32, i32* %arrayidx, align 4
  %arrayidx5 = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  store i32 %tmp1, i32* %arrayidx5, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body3
  %inc = add nsw i32 %j.0, 1
  br label %for.cond1

for.end:                                          ; preds = %for.cond1
  br label %for.inc6

for.inc6:                                         ; preds = %for.end
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond

for.end8:                                         ; preds = %for.cond
  ret void
}
