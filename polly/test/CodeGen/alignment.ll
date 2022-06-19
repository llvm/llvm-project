; RUN: opt %loadPolly -polly-codegen -S < %s | FileCheck %s
;
; Check that the special alignment information is kept
;
; CHECK: align 8
; CHECK: align 8
;
;    void jd(int *A) {
;      for (int i = 0; i < 1024; i += 2)
;        A[i] = i;
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @jd(i32* %A) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %cmp = icmp slt i64 %indvars.iv, 1024
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %tmp = trunc i64 %indvars.iv to i32
  store i32 %tmp, i32* %arrayidx, align 8
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 2
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
