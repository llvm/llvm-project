; RUN: opt %loadNPMPolly '-passes=print<polly-detect>' -disable-output < %s 2>&1 | FileCheck %s
;
; CHECK: Valid Region for Scop:
;
;    void jd(int *A, int *B) {
;      for (int i = 0; i < 1024; i++)
;        A[i] = B[0] + B[1023];
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @jd(ptr %A, ptr %B) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %exitcond = icmp ne i64 %indvars.iv, 1024
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %tmp = load i32, ptr %B, align 4
  %arrayidx1 = getelementptr inbounds i32, ptr %B, i64 1023
  %tmp1 = load i32, ptr %arrayidx1, align 4
  %add = add nsw i32 %tmp, %tmp1
  %arrayidx2 = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  store i32 %add, ptr %arrayidx2, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
