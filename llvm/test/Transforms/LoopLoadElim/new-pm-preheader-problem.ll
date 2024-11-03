; RUN: opt -passes=loop-load-elim -S < %s | FileCheck %s
; Make sure it doesn't crash in new pass manager due to missing preheader.
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

define void @test(ptr noalias nocapture %A, ptr noalias nocapture readonly %B, i64 %N, i1 %C) {
entry:
  br i1 %C, label %for.body, label %for.end

; CHECK: test
for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  %load = load i32, ptr %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, ptr %B, i64 %indvars.iv
  %load_1 = load i32, ptr %arrayidx2, align 4
  %add = add i32 %load_1, %load
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %arrayidx_next = getelementptr inbounds i32, ptr %A, i64 %indvars.iv.next
  store i32 %add, ptr %arrayidx_next, align 4
  %exitcond = icmp eq i64 %indvars.iv.next, %N
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}
