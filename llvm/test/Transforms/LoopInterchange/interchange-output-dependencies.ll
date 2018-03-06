; REQUIRES: asserts
; RUN: opt < %s -basicaa -loop-interchange -verify-dom-info -S -debug 2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@A = common global [100 x [100 x i32]] zeroinitializer

;; Test to make sure we can handle output dependencies.
;;
;;  for (int i = 0; i < 2; ++i)
;;    for(int j = 0; j < 3; ++j) {
;;      A[j][i] = i;
;;      A[j][i+1] = j;
;;    }

; CHECK: Not interchanging loops. Cannot prove legality.

@A10 = local_unnamed_addr global [3 x [3 x i32]] zeroinitializer, align 16

define void @interchange_10() {
entry:
  br label %for.cond1.preheader

for.cond.loopexit:                                ; preds = %for.body4
  %exitcond28 = icmp ne i64 %indvars.iv.next27, 2
  br i1 %exitcond28, label %for.cond1.preheader, label %for.cond.cleanup

for.cond1.preheader:                              ; preds = %for.cond.loopexit, %entry
  %indvars.iv26 = phi i64 [ 0, %entry ], [ %indvars.iv.next27, %for.cond.loopexit ]
  %indvars.iv.next27 = add nuw nsw i64 %indvars.iv26, 1
  br label %for.body4

for.cond.cleanup:                                 ; preds = %for.cond.loopexit
  ret void

for.body4:                                        ; preds = %for.body4, %for.cond1.preheader
  %indvars.iv = phi i64 [ 0, %for.cond1.preheader ], [ %indvars.iv.next, %for.body4 ]
  %arrayidx6 = getelementptr inbounds [3 x [3 x i32]], [3 x [3 x i32]]* @A10, i64 0, i64 %indvars.iv, i64 %indvars.iv26
  %tmp = trunc i64 %indvars.iv26 to i32
  store i32 %tmp, i32* %arrayidx6, align 4
  %arrayidx10 = getelementptr inbounds [3 x [3 x i32]], [3 x [3 x i32]]* @A10, i64 0, i64 %indvars.iv, i64 %indvars.iv.next27
  %tmp1 = trunc i64 %indvars.iv to i32
  store i32 %tmp1, i32* %arrayidx10, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp ne i64 %indvars.iv.next, 3
  br i1 %exitcond, label %for.body4, label %for.cond.loopexit
}
