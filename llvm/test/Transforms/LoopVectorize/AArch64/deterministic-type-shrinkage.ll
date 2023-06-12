; RUN: opt -S < %s -passes=loop-vectorize,instcombine 2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"
target triple = "aarch64"

;; See https://llvm.org/bugs/show_bug.cgi?id=25490
;; Due to the data structures used, the LLVM IR was not determinisic.
;; This test comes from the PR.

;; CHECK-LABEL: @test(
; CHECK: load <16 x i8>
; CHECK-NEXT: getelementptr
; CHECK-NEXT: load <16 x i8>
; CHECK-NEXT: zext <16 x i8>
; CHECK-NEXT: zext <16 x i8>
define void @test(i32 %n, ptr nocapture %a, ptr nocapture %b, ptr nocapture readonly %c) {
entry:
  %cmp.28 = icmp eq i32 %n, 0
  br i1 %cmp.28, label %for.cond.cleanup, label %for.body.preheader

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.cond.cleanup.loopexit:                        ; preds = %for.body
  br label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond.cleanup.loopexit, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i8, ptr %c, i64 %indvars.iv
  %0 = load i8, ptr %arrayidx, align 1
  %conv = zext i8 %0 to i32
  %arrayidx2 = getelementptr inbounds i8, ptr %a, i64 %indvars.iv
  %1 = load i8, ptr %arrayidx2, align 1
  %conv3 = zext i8 %1 to i32
  %mul = mul nuw nsw i32 %conv3, %conv
  %shr.26 = lshr i32 %mul, 8
  %conv4 = trunc i32 %shr.26 to i8
  store i8 %conv4, ptr %arrayidx2, align 1
  %arrayidx8 = getelementptr inbounds i8, ptr %b, i64 %indvars.iv
  %2 = load i8, ptr %arrayidx8, align 1
  %conv9 = zext i8 %2 to i32
  %mul10 = mul nuw nsw i32 %conv9, %conv
  %shr11.27 = lshr i32 %mul10, 8
  %conv12 = trunc i32 %shr11.27 to i8
  store i8 %conv12, ptr %arrayidx8, align 1
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.cond.cleanup.loopexit, label %for.body
}
