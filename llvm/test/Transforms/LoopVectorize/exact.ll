; RUN: opt < %s -passes=loop-vectorize -force-vector-width=4 -S | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

; CHECK-LABEL: @lshr_exact(
; CHECK: lshr exact <4 x i32>
define void @lshr_exact(ptr %x) {
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %x, i64 %iv
  %0 = load i32, ptr %arrayidx, align 4
  %conv1 = lshr exact i32 %0, 1
  store i32 %conv1, ptr %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond = icmp eq i64 %iv.next, 256
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}
