; RUN: opt < %s -basicaa -loop-interchange -S | FileCheck %s
;; We test the complete .ll for adjustment in outer loop header/latch and inner loop header/latch.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@A = common global [100 x [100 x i32]] zeroinitializer
@B = common global [100 x i32] zeroinitializer

;; Loops should not be interchanged in this case as it is not legal due to dependency.
;;  for(int j=0;j<99;j++)
;;   for(int i=0;i<99;i++)
;;       A[j][i+1] = A[j+1][i]+k;

define void @interchange_04(i32 %k){
entry:
  br label %for.cond1.preheader

for.cond1.preheader:
  %indvars.iv23 = phi i64 [ 0, %entry ], [ %indvars.iv.next24, %for.inc12 ]
  %indvars.iv.next24 = add nuw nsw i64 %indvars.iv23, 1
  br label %for.body3

for.body3:
  %indvars.iv = phi i64 [ 0, %for.cond1.preheader ], [ %indvars.iv.next, %for.body3 ]
  %arrayidx5 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @A, i64 0, i64 %indvars.iv.next24, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx5
  %add6 = add nsw i32 %0, %k
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %arrayidx11 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @A, i64 0, i64 %indvars.iv23, i64 %indvars.iv.next
  store i32 %add6, i32* %arrayidx11
  %exitcond = icmp eq i64 %indvars.iv.next, 99
  br i1 %exitcond, label %for.inc12, label %for.body3

for.inc12:
  %exitcond25 = icmp eq i64 %indvars.iv.next24, 99
  br i1 %exitcond25, label %for.end14, label %for.cond1.preheader

for.end14:
  ret void
}

; CHECK-LABEL: @interchange_04
; CHECK: entry:
; CHECK:   br label %for.cond1.preheader
; CHECK: for.cond1.preheader:                              ; preds = %for.inc12, %entry
; CHECK:   %indvars.iv23 = phi i64 [ 0, %entry ], [ %indvars.iv.next24, %for.inc12 ]
; CHECK:   %indvars.iv.next24 = add nuw nsw i64 %indvars.iv23, 1
; CHECK:   br label %for.body3
; CHECK: for.body3:                                        ; preds = %for.body3, %for.cond1.preheader
; CHECK:   %indvars.iv = phi i64 [ 0, %for.cond1.preheader ], [ %indvars.iv.next, %for.body3 ]
; CHECK:   %arrayidx5 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @A, i64 0, i64 %indvars.iv.next24, i64 %indvars.iv
; CHECK:   %0 = load i32, i32* %arrayidx5
; CHECK:   %add6 = add nsw i32 %0, %k
; CHECK:   %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
; CHECK:   %arrayidx11 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @A, i64 0, i64 %indvars.iv23, i64 %indvars.iv.next
; CHECK:   store i32 %add6, i32* %arrayidx11
; CHECK:   %exitcond = icmp eq i64 %indvars.iv.next, 99
; CHECK:   br i1 %exitcond, label %for.inc12, label %for.body3
; CHECK: for.inc12:                                        ; preds = %for.body3
; CHECK:   %exitcond25 = icmp eq i64 %indvars.iv.next24, 99
; CHECK:   br i1 %exitcond25, label %for.end14, label %for.cond1.preheader
; CHECK: for.end14:                                        ; preds = %for.inc12
; CHECK:   ret void
