; RUN: opt < %s -basicaa -loop-interchange -S | FileCheck %s
;; We test the complete .ll for adjustment in outer loop header/latch and inner loop header/latch.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@A = common global [100 x [100 x i32]] zeroinitializer
@B = common global [100 x i32] zeroinitializer

;; for(int i=0;i<100;i++)
;;   for(int j=100;j>=0;j--)
;;     A[j][i] = A[j][i]+k;

define void @interchange_02(i32 %k) {
entry:
  br label %for.cond1.preheader

for.cond1.preheader:
  %indvars.iv19 = phi i64 [ 0, %entry ], [ %indvars.iv.next20, %for.inc10 ]
  br label %for.body3

for.body3:
  %indvars.iv = phi i64 [ 100, %for.cond1.preheader ], [ %indvars.iv.next, %for.body3 ]
  %arrayidx5 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @A, i64 0, i64 %indvars.iv, i64 %indvars.iv19
  %0 = load i32, i32* %arrayidx5
  %add = add nsw i32 %0, %k
  store i32 %add, i32* %arrayidx5
  %indvars.iv.next = add nsw i64 %indvars.iv, -1
  %cmp2 = icmp sgt i64 %indvars.iv, 0
  br i1 %cmp2, label %for.body3, label %for.inc10

for.inc10:
  %indvars.iv.next20 = add nuw nsw i64 %indvars.iv19, 1
  %exitcond = icmp eq i64 %indvars.iv.next20, 100
  br i1 %exitcond, label %for.end11, label %for.cond1.preheader

for.end11:
  ret void
}

; CHECK-LABEL: @interchange_02
; CHECK: entry:
; CHECK:   br label %for.body3.preheader
; CHECK: for.cond1.preheader.preheader:
; CHECK:   br label %for.cond1.preheader
; CHECK: for.cond1.preheader:
; CHECK:   %indvars.iv19 = phi i64 [ %indvars.iv.next20, %for.inc10 ], [ 0, %for.cond1.preheader.preheader ]
; CHECK:   br label %for.body3.split1
; CHECK: for.body3.preheader:
; CHECK:   br label %for.body3
; CHECK: for.body3:
; CHECK:   %indvars.iv = phi i64 [ %indvars.iv.next, %for.body3.split ], [ 100, %for.body3.preheader ]
; CHECK:   br label %for.cond1.preheader.preheader
; CHECK: for.body3.split1:                                 ; preds = %for.cond1.preheader
; CHECK:   %arrayidx5 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @A, i64 0, i64 %indvars.iv, i64 %indvars.iv19
; CHECK:   %0 = load i32, i32* %arrayidx5
; CHECK:   %add = add nsw i32 %0, %k
; CHECK:   store i32 %add, i32* %arrayidx5
; CHECK:   br label %for.inc10
; CHECK: for.body3.split:
; CHECK:   %indvars.iv.next = add nsw i64 %indvars.iv, -1
; CHECK:   %cmp2 = icmp sgt i64 %indvars.iv, 0
; CHECK:   br i1 %cmp2, label %for.body3, label %for.end11
; CHECK: for.inc10:
; CHECK:   %indvars.iv.next20 = add nuw nsw i64 %indvars.iv19, 1
; CHECK:   %exitcond = icmp eq i64 %indvars.iv.next20, 100
; CHECK:   br i1 %exitcond, label %for.body3.split, label %for.cond1.preheader
; CHECK: for.end11:
; CHECK:   ret void
