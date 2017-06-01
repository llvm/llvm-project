; RUN: opt -basicaa -loop-versioning -S < %s | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

; Version this loop with overlap checks between a, c and b, c.

define void @f(i32* %a, i32* %b, i32* %c) {
entry:
  br label %for.body

; CHECK: for.body.lver.check:
; CHECK:   icmp
; CHECK:   icmp
; CHECK:   icmp
; CHECK:   icmp
; CHECK-NOT: icmp
; CHECK:   br i1 %memcheck.conflict, label %for.body.ph.lver.orig, label %for.body.ph

; CHECK: for.body.ph.lver.orig:
; CHECK: for.body.lver.orig:
; CHECK:   br i1 %exitcond.lver.orig, label %for.end, label %for.body.lver.orig
; CHECK: for.body.ph:
; CHECK: for.body:
; CHECK:   br i1 %exitcond, label %for.end, label %for.body
; CHECK: for.end:

for.body:                                         ; preds = %for.body, %entry
  %ind = phi i64 [ 0, %entry ], [ %add, %for.body ]

  %arrayidxA = getelementptr inbounds i32, i32* %a, i64 %ind
  %loadA = load i32, i32* %arrayidxA, align 4

  %arrayidxB = getelementptr inbounds i32, i32* %b, i64 %ind
  %loadB = load i32, i32* %arrayidxB, align 4

  %mulC = mul i32 %loadA, %loadB

  %arrayidxC = getelementptr inbounds i32, i32* %c, i64 %ind
  store i32 %mulC, i32* %arrayidxC, align 4

  %add = add nuw nsw i64 %ind, 1
  %exitcond = icmp eq i64 %add, 20
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}
