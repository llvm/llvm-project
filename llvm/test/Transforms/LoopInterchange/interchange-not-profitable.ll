; RUN: opt < %s -basicaa -loop-interchange -S | FileCheck %s
;; We test the complete .ll for adjustment in outer loop header/latch and inner loop header/latch.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@A = common global [100 x [100 x i32]] zeroinitializer
@B = common global [100 x i32] zeroinitializer

;; Loops should not be interchanged in this case as it is not profitable.
;;  for(int i=0;i<100;i++)
;;    for(int j=0;j<100;j++)
;;      A[i][j] = A[i][j]+k;

define void @interchange_03(i32 %k) {
entry:
  br label %for.cond1.preheader

for.cond1.preheader:
  %indvars.iv21 = phi i64 [ 0, %entry ], [ %indvars.iv.next22, %for.inc10 ]
  br label %for.body3

for.body3:
  %indvars.iv = phi i64 [ 0, %for.cond1.preheader ], [ %indvars.iv.next, %for.body3 ]
  %arrayidx5 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @A, i64 0, i64 %indvars.iv21, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx5
  %add = add nsw i32 %0, %k
  store i32 %add, i32* %arrayidx5
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 100
  br i1 %exitcond, label %for.inc10, label %for.body3

for.inc10:
  %indvars.iv.next22 = add nuw nsw i64 %indvars.iv21, 1
  %exitcond23 = icmp eq i64 %indvars.iv.next22, 100
  br i1 %exitcond23, label %for.end12, label %for.cond1.preheader

for.end12:
  ret void
}

; CHECK-LABEL: @interchange_03
; CHECK: entry:
; CHECK:   br label %for.cond1.preheader.preheader
; CHECK: for.cond1.preheader.preheader:                    ; preds = %entry
; CHECK:   br label %for.cond1.preheader
; CHECK: for.cond1.preheader:                              ; preds = %for.cond1.preheader.preheader, %for.inc10
; CHECK:   %indvars.iv21 = phi i64 [ %indvars.iv.next22, %for.inc10 ], [ 0, %for.cond1.preheader.preheader ]
; CHECK:  br label %for.body3.preheader
; CHECK: for.body3.preheader:                              ; preds = %for.cond1.preheader
; CHECK:   br label %for.body3
; CHECK: for.body3:                                        ; preds = %for.body3.preheader, %for.body3
; CHECK:   %indvars.iv = phi i64 [ %indvars.iv.next, %for.body3 ], [ 0, %for.body3.preheader ]
; CHECK:   %arrayidx5 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @A, i64 0, i64 %indvars.iv21, i64 %indvars.iv
; CHECK:   %0 = load i32, i32* %arrayidx5
; CHECK:   %add = add nsw i32 %0, %k
; CHECK:   store i32 %add, i32* %arrayidx5
; CHECK:   %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
; CHECK:   %exitcond = icmp eq i64 %indvars.iv.next, 100
; CHECK:   br i1 %exitcond, label %for.inc10, label %for.body3
; CHECK: for.inc10:                                        ; preds = %for.body3
; CHECK:   %indvars.iv.next22 = add nuw nsw i64 %indvars.iv21, 1
; CHECK:   %exitcond23 = icmp eq i64 %indvars.iv.next22, 100
; CHECK:   br i1 %exitcond23, label %for.end12, label %for.cond1.preheader
; CHECK: for.end12:                                        ; preds = %for.inc10
; CHECK:   ret void
