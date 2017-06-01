; RUN: opt < %s -basicaa -loop-interchange -S | FileCheck %s
;; We test the complete .ll for adjustment in outer loop header/latch and inner loop header/latch.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@A = common global [100 x [100 x i32]] zeroinitializer
@B = common global [100 x i32] zeroinitializer
@C = common global [100 x [100 x i32]] zeroinitializer
@D = common global [100 x [100 x [100 x i32]]] zeroinitializer

declare void @foo(...)

;;--------------------------------------Test case 01------------------------------------
;;  for(int i=0;i<N;i++)
;;    for(int j=1;j<N;j++)
;;      A[j][i] = A[j][i]+k;

define void @interchange_01(i32 %k, i32 %N) {
entry:
  %cmp21 = icmp sgt i32 %N, 0
  br i1 %cmp21, label %for.cond1.preheader.lr.ph, label %for.end12

for.cond1.preheader.lr.ph:
  %cmp219 = icmp sgt i32 %N, 1
  %0 = add i32 %N, -1
  br label %for.cond1.preheader

for.cond1.preheader: 
  %indvars.iv23 = phi i64 [ 0, %for.cond1.preheader.lr.ph ], [ %indvars.iv.next24, %for.inc10 ]
  br i1 %cmp219, label %for.body3, label %for.inc10

for.body3:
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body3 ], [ 1, %for.cond1.preheader ]
  %arrayidx5 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @A, i64 0, i64 %indvars.iv, i64 %indvars.iv23
  %1 = load i32, i32* %arrayidx5
  %add = add nsw i32 %1, %k
  store i32 %add, i32* %arrayidx5
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %0
  br i1 %exitcond, label %for.inc10, label %for.body3

for.inc10:
  %indvars.iv.next24 = add nuw nsw i64 %indvars.iv23, 1
  %lftr.wideiv25 = trunc i64 %indvars.iv23 to i32
  %exitcond26 = icmp eq i32 %lftr.wideiv25, %0
  br i1 %exitcond26, label %for.end12, label %for.cond1.preheader

for.end12:
  ret void
}

; CHECK-LABEL: @interchange_01
; CHECK: entry:
; CHECK:   %cmp21 = icmp sgt i32 %N, 0
; CHECK:   br i1 %cmp21, label %for.body3.preheader, label %for.end12
; CHECK: for.cond1.preheader.lr.ph:                        
; CHECK:   br label %for.cond1.preheader
; CHECK: for.cond1.preheader:                              
; CHECK:   %indvars.iv23 = phi i64 [ 0, %for.cond1.preheader.lr.ph ], [ %indvars.iv.next24, %for.inc10 ]
; CHECK:   br i1 %cmp219, label %for.body3.split1, label %for.end12.loopexit
; CHECK: for.body3.preheader:                              
; CHECK:   %cmp219 = icmp sgt i32 %N, 1
; CHECK:   %0 = add i32 %N, -1
; CHECK:   br label %for.body3
; CHECK: for.body3:                                        
; CHECK:   %indvars.iv = phi i64 [ %indvars.iv.next, %for.body3.split ], [ 1, %for.body3.preheader ]
; CHECK:   br label %for.cond1.preheader.lr.ph
; CHECK: for.body3.split1:                                 
; CHECK:   %arrayidx5 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @A, i64 0, i64 %indvars.iv, i64 %indvars.iv23
; CHECK:   %1 = load i32, i32* %arrayidx5
; CHECK:   %add = add nsw i32 %1, %k
; CHECK:   store i32 %add, i32* %arrayidx5
; CHECK:   br label %for.inc10.loopexit
; CHECK: for.body3.split:                                  
; CHECK:   %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
; CHECK:   %lftr.wideiv = trunc i64 %indvars.iv to i32
; CHECK:   %exitcond = icmp eq i32 %lftr.wideiv, %0
; CHECK:   br i1 %exitcond, label %for.end12.loopexit, label %for.body3
; CHECK: for.inc10.loopexit:                               
; CHECK:   br label %for.inc10
; CHECK: for.inc10:                                        
; CHECK:   %indvars.iv.next24 = add nuw nsw i64 %indvars.iv23, 1
; CHECK:   %lftr.wideiv25 = trunc i64 %indvars.iv23 to i32
; CHECK:   %exitcond26 = icmp eq i32 %lftr.wideiv25, %0
; CHECK:   br i1 %exitcond26, label %for.body3.split, label %for.cond1.preheader
; CHECK: for.end12.loopexit:                               
; CHECK:   br label %for.end12
; CHECK: for.end12:                                        
; CHECK:   ret void

;;--------------------------------------Test case 02-------------------------------------

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

;;--------------------------------------Test case 03-------------------------------------
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


;;--------------------------------------Test case 04-------------------------------------
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



;;--------------------------------------Test case 05-------------------------------------
;; Loops not tightly nested are not interchanged
;;  for(int j=0;j<N;j++) {
;;    B[j] = j+k;
;;    for(int i=0;i<N;i++)
;;      A[j][i] = A[j][i]+B[j];
;;  }

define void @interchange_05(i32 %k, i32 %N){
entry:
  %cmp30 = icmp sgt i32 %N, 0
  br i1 %cmp30, label %for.body.lr.ph, label %for.end17

for.body.lr.ph:
  %0 = add i32 %N, -1
  %1 = zext i32 %k to i64
  br label %for.body

for.body:
  %indvars.iv32 = phi i64 [ 0, %for.body.lr.ph ], [ %indvars.iv.next33, %for.inc15 ]
  %2 = add nsw i64 %indvars.iv32, %1
  %arrayidx = getelementptr inbounds [100 x i32], [100 x i32]* @B, i64 0, i64 %indvars.iv32
  %3 = trunc i64 %2 to i32
  store i32 %3, i32* %arrayidx
  br label %for.body3

for.body3:
  %indvars.iv = phi i64 [ 0, %for.body ], [ %indvars.iv.next, %for.body3 ]
  %arrayidx7 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @A, i64 0, i64 %indvars.iv32, i64 %indvars.iv
  %4 = load i32, i32* %arrayidx7
  %add10 = add nsw i32 %3, %4
  store i32 %add10, i32* %arrayidx7
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %0
  br i1 %exitcond, label %for.inc15, label %for.body3

for.inc15:
  %indvars.iv.next33 = add nuw nsw i64 %indvars.iv32, 1
  %lftr.wideiv35 = trunc i64 %indvars.iv32 to i32
  %exitcond36 = icmp eq i32 %lftr.wideiv35, %0
  br i1 %exitcond36, label %for.end17, label %for.body

for.end17:
  ret void
}

; CHECK-LABEL: @interchange_05
; CHECK: entry:
; CHECK: %cmp30 = icmp sgt i32 %N, 0
; CHECK: br i1 %cmp30, label %for.body.lr.ph, label %for.end17
; CHECK: for.body.lr.ph:
; CHECK: %0 = add i32 %N, -1
; CHECK: %1 = zext i32 %k to i64
; CHECK: br label %for.body
; CHECK: for.body:
; CHECK: %indvars.iv32 = phi i64 [ 0, %for.body.lr.ph ], [ %indvars.iv.next33, %for.inc15 ]
; CHECK: %2 = add nsw i64 %indvars.iv32, %1
; CHECK: %arrayidx = getelementptr inbounds [100 x i32], [100 x i32]* @B, i64 0, i64 %indvars.iv32
; CHECK: %3 = trunc i64 %2 to i32
; CHECK: store i32 %3, i32* %arrayidx
; CHECK: br label %for.body3.preheader
; CHECK: for.body3.preheader:
; CHECK: br label %for.body3
; CHECK: for.body3:
; CHECK: %indvars.iv = phi i64 [ %indvars.iv.next, %for.body3 ], [ 0, %for.body3.preheader ]
; CHECK: %arrayidx7 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @A, i64 0, i64 %indvars.iv32, i64 %indvars.iv
; CHECK: %4 = load i32, i32* %arrayidx7
; CHECK: %add10 = add nsw i32 %3, %4
; CHECK: store i32 %add10, i32* %arrayidx7
; CHECK: %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
; CHECK: %lftr.wideiv = trunc i64 %indvars.iv to i32
; CHECK: %exitcond = icmp eq i32 %lftr.wideiv, %0
; CHECK: br i1 %exitcond, label %for.inc15, label %for.body3
; CHECK: for.inc15:
; CHECK: %indvars.iv.next33 = add nuw nsw i64 %indvars.iv32, 1
; CHECK: %lftr.wideiv35 = trunc i64 %indvars.iv32 to i32
; CHECK: %exitcond36 = icmp eq i32 %lftr.wideiv35, %0
; CHECK: br i1 %exitcond36, label %for.end17.loopexit, label %for.body
; CHECK: for.end17.loopexit:
; CHECK: br label %for.end17
; CHECK: for.end17:
; CHECK: ret void


;;--------------------------------------Test case 06-------------------------------------
;; Loops not tightly nested are not interchanged
;;  for(int j=0;j<N;j++) {
;;    foo();
;;    for(int i=2;i<N;i++)
;;      A[j][i] = A[j][i]+k;
;;  }

define void @interchange_06(i32 %k, i32 %N) {
entry:
  %cmp22 = icmp sgt i32 %N, 0
  br i1 %cmp22, label %for.body.lr.ph, label %for.end12

for.body.lr.ph:
  %0 = add i32 %N, -1
  br label %for.body

for.body:
  %indvars.iv24 = phi i64 [ 0, %for.body.lr.ph ], [ %indvars.iv.next25, %for.inc10 ]
  tail call void (...) @foo()
  br label %for.body3

for.body3:
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body3 ], [ 2, %for.body ]
  %arrayidx5 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @A, i64 0, i64 %indvars.iv24, i64 %indvars.iv
  %1 = load i32, i32* %arrayidx5
  %add = add nsw i32 %1, %k
  store i32 %add, i32* %arrayidx5
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %0
  br i1 %exitcond, label %for.inc10, label %for.body3

for.inc10:
  %indvars.iv.next25 = add nuw nsw i64 %indvars.iv24, 1
  %lftr.wideiv26 = trunc i64 %indvars.iv24 to i32
  %exitcond27 = icmp eq i32 %lftr.wideiv26, %0
  br i1 %exitcond27, label %for.end12, label %for.body

for.end12:
  ret void
}
;; Here we are checking if the inner phi is not split then we have not interchanged.
; CHECK-LABEL: @interchange_06
; CHECK:  phi i64 [ %indvars.iv.next, %for.body3 ], [ 2, %for.body3.preheader ]
; CHECK-NEXT: getelementptr
; CHECK-NEXT: %1 = load

;;--------------------------------------Test case 07-------------------------------------
;; FIXME:
;; Test for interchange when we have an lcssa phi. This should ideally be interchanged but it is currently not supported.
;;     for(gi=1;gi<N;gi++)
;;       for(gj=1;gj<M;gj++)
;;         A[gj][gi] = A[gj - 1][gi] + C[gj][gi];

@gi = common global i32 0
@gj = common global i32 0

define void @interchange_07(i32 %N, i32 %M){
entry:
  store i32 1, i32* @gi
  %cmp21 = icmp sgt i32 %N, 1
  br i1 %cmp21, label %for.cond1.preheader.lr.ph, label %for.end16

for.cond1.preheader.lr.ph: 
  %cmp218 = icmp sgt i32 %M, 1
  %gi.promoted = load i32, i32* @gi
  %0 = add i32 %M, -1
  %1 = sext i32 %gi.promoted to i64
  %2 = sext i32 %N to i64
  %3 = add i32 %gi.promoted, 1
  %4 = icmp slt i32 %3, %N
  %smax = select i1 %4, i32 %N, i32 %3
  br label %for.cond1.preheader

for.cond1.preheader: 
  %indvars.iv25 = phi i64 [ %1, %for.cond1.preheader.lr.ph ], [ %indvars.iv.next26, %for.inc14 ]
  br i1 %cmp218, label %for.body3, label %for.inc14

for.body3: 
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body3 ], [ 1, %for.cond1.preheader ]
  %5 = add nsw i64 %indvars.iv, -1
  %arrayidx5 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @A, i64 0, i64 %5, i64 %indvars.iv25
  %6 = load i32, i32* %arrayidx5
  %arrayidx9 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @C, i64 0, i64 %indvars.iv, i64 %indvars.iv25
  %7 = load i32, i32* %arrayidx9
  %add = add nsw i32 %7, %6
  %arrayidx13 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @A, i64 0, i64 %indvars.iv, i64 %indvars.iv25
  store i32 %add, i32* %arrayidx13
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %0
  br i1 %exitcond, label %for.inc14, label %for.body3

for.inc14: 
  %inc.lcssa23 = phi i32 [ 1, %for.cond1.preheader ], [ %M, %for.body3 ]
  %indvars.iv.next26 = add nsw i64 %indvars.iv25, 1
  %cmp = icmp slt i64 %indvars.iv.next26, %2
  br i1 %cmp, label %for.cond1.preheader, label %for.cond.for.end16_crit_edge

for.cond.for.end16_crit_edge: 
  store i32 %inc.lcssa23, i32* @gj
  store i32 %smax, i32* @gi
  br label %for.end16

for.end16:
  ret void
}

; CHECK-LABEL: @interchange_07
; CHECK: for.body3:                                        ; preds = %for.body3.preheader, %for.body3
; CHECK:   %indvars.iv = phi i64 [ %indvars.iv.next, %for.body3 ], [ 1, %for.body3.preheader ]
; CHECK:   %5 = add nsw i64 %indvars.iv, -1
; CHECK:   %arrayidx5 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @A, i64 0, i64 %5, i64 %indvars.iv25
; CHECK:   %6 = load i32, i32* %arrayidx5
; CHECK:   %arrayidx9 = getelementptr inbounds [100 x [100 x i32]], [100 x [100 x i32]]* @C, i64 0, i64 %indvars.iv, i64 %indvars.iv25

;;------------------------------------------------Test case 08-------------------------------
;; Test for interchange in loop nest greater than 2.
;;  for(int i=0;i<100;i++)
;;    for(int j=0;j<100;j++)
;;      for(int k=0;k<100;k++)
;;        D[i][k][j] = D[i][k][j]+t;

define void @interchange_08(i32 %t){
entry:
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.inc15, %entry
  %i.028 = phi i32 [ 0, %entry ], [ %inc16, %for.inc15 ]
  br label %for.cond4.preheader

for.cond4.preheader:                              ; preds = %for.inc12, %for.cond1.preheader
  %j.027 = phi i32 [ 0, %for.cond1.preheader ], [ %inc13, %for.inc12 ]
  br label %for.body6

for.body6:                                        ; preds = %for.body6, %for.cond4.preheader
  %k.026 = phi i32 [ 0, %for.cond4.preheader ], [ %inc, %for.body6 ]
  %arrayidx8 = getelementptr inbounds [100 x [100 x [100 x i32]]], [100 x [100 x [100 x i32]]]* @D, i32 0, i32 %i.028, i32 %k.026, i32 %j.027
  %0 = load i32, i32* %arrayidx8
  %add = add nsw i32 %0, %t
  store i32 %add, i32* %arrayidx8
  %inc = add nuw nsw i32 %k.026, 1
  %exitcond = icmp eq i32 %inc, 100
  br i1 %exitcond, label %for.inc12, label %for.body6

for.inc12:                                        ; preds = %for.body6
  %inc13 = add nuw nsw i32 %j.027, 1
  %exitcond29 = icmp eq i32 %inc13, 100
  br i1 %exitcond29, label %for.inc15, label %for.cond4.preheader

for.inc15:                                        ; preds = %for.inc12
  %inc16 = add nuw nsw i32 %i.028, 1
  %exitcond30 = icmp eq i32 %inc16, 100
  br i1 %exitcond30, label %for.end17, label %for.cond1.preheader

for.end17:                                        ; preds = %for.inc15
  ret void
}
; CHECK-LABEL: @interchange_08
; CHECK:   entry:
; CHECK:     br label %for.cond1.preheader.preheader
; CHECK:   for.cond1.preheader.preheader:                    ; preds = %entry
; CHECK:     br label %for.cond1.preheader
; CHECK:   for.cond1.preheader:                              ; preds = %for.cond1.preheader.preheader, %for.inc15
; CHECK:     %i.028 = phi i32 [ %inc16, %for.inc15 ], [ 0, %for.cond1.preheader.preheader ]
; CHECK:     br label %for.body6.preheader
; CHECK:   for.cond4.preheader.preheader:                    ; preds = %for.body6
; CHECK:     br label %for.cond4.preheader
; CHECK:   for.cond4.preheader:                              ; preds = %for.cond4.preheader.preheader, %for.inc12
; CHECK:     %j.027 = phi i32 [ %inc13, %for.inc12 ], [ 0, %for.cond4.preheader.preheader ]
; CHECK:     br label %for.body6.split1
; CHECK:   for.body6.preheader:                              ; preds = %for.cond1.preheader
; CHECK:     br label %for.body6
; CHECK:   for.body6:                                        ; preds = %for.body6.preheader, %for.body6.split
; CHECK:     %k.026 = phi i32 [ %inc, %for.body6.split ], [ 0, %for.body6.preheader ]
; CHECK:     br label %for.cond4.preheader.preheader
; CHECK:   for.body6.split1:                                 ; preds = %for.cond4.preheader
; CHECK:     %arrayidx8 = getelementptr inbounds [100 x [100 x [100 x i32]]], [100 x [100 x [100 x i32]]]* @D, i32 0, i32 %i.028, i32 %k.026, i32 %j.027
; CHECK:     %0 = load i32, i32* %arrayidx8
; CHECK:     %add = add nsw i32 %0, %t
; CHECK:     store i32 %add, i32* %arrayidx8
; CHECK:     br label %for.inc12
; CHECK:   for.body6.split:                                  ; preds = %for.inc12
; CHECK:     %inc = add nuw nsw i32 %k.026, 1
; CHECK:     %exitcond = icmp eq i32 %inc, 100
; CHECK:     br i1 %exitcond, label %for.inc15, label %for.body6
; CHECK:   for.inc12:                                        ; preds = %for.body6.split1
; CHECK:     %inc13 = add nuw nsw i32 %j.027, 1
; CHECK:     %exitcond29 = icmp eq i32 %inc13, 100
; CHECK:     br i1 %exitcond29, label %for.body6.split, label %for.cond4.preheader
; CHECK:   for.inc15:                                        ; preds = %for.body6.split
; CHECK:     %inc16 = add nuw nsw i32 %i.028, 1
; CHECK:     %exitcond30 = icmp eq i32 %inc16, 100
; CHECK:     br i1 %exitcond30, label %for.end17, label %for.cond1.preheader
; CHECK:   for.end17:                                        ; preds = %for.inc15
; CHECK:     ret void

;;-----------------------------------Test case 09-------------------------------
;; Test that a flow dependency in outer loop doesn't prevent interchange in
;; loops i and j.
;;
;;  for (int k = 0; k < 100; ++k) {
;;    T[k] = fn1();
;;    for (int i = 0; i < 1000; ++i)
;;      for(int j = 1; j < 1000; ++j)
;;        Arr[j][i] = Arr[j][i]+k;
;;    fn2(T[k]);
;;  }

@T = internal global [100 x double] zeroinitializer, align 4
@Arr = internal global [1000 x [1000 x i32]] zeroinitializer, align 4

define void @interchange_09(i32 %k) {
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.cond.cleanup4
  ret void

for.body:                                         ; preds = %for.cond.cleanup4, %entry
  %indvars.iv45 = phi i64 [ 0, %entry ], [ %indvars.iv.next46, %for.cond.cleanup4 ]
  %call = call double @fn1()
  %arrayidx = getelementptr inbounds [100 x double], [100 x double]* @T, i64 0, i64 %indvars.iv45
  store double %call, double* %arrayidx, align 8
  br label %for.cond6.preheader

for.cond6.preheader:                              ; preds = %for.cond.cleanup8, %for.body
  %indvars.iv42 = phi i64 [ 0, %for.body ], [ %indvars.iv.next43, %for.cond.cleanup8 ]
  br label %for.body9

for.cond.cleanup4:                                ; preds = %for.cond.cleanup8
  %tmp = load double, double* %arrayidx, align 8
  call void @fn2(double %tmp)
  %indvars.iv.next46 = add nuw nsw i64 %indvars.iv45, 1
  %exitcond47 = icmp ne i64 %indvars.iv.next46, 100
  br i1 %exitcond47, label %for.body, label %for.cond.cleanup

for.cond.cleanup8:                                ; preds = %for.body9
  %indvars.iv.next43 = add nuw nsw i64 %indvars.iv42, 1
  %exitcond44 = icmp ne i64 %indvars.iv.next43, 1000
  br i1 %exitcond44, label %for.cond6.preheader, label %for.cond.cleanup4

for.body9:                                        ; preds = %for.body9, %for.cond6.preheader
  %indvars.iv = phi i64 [ 1, %for.cond6.preheader ], [ %indvars.iv.next, %for.body9 ]
  %arrayidx13 = getelementptr inbounds [1000 x [1000 x i32]], [1000 x [1000 x i32]]* @Arr, i64 0, i64 %indvars.iv, i64 %indvars.iv42
  %tmp1 = load i32, i32* %arrayidx13, align 4
  %tmp2 = trunc i64 %indvars.iv45 to i32
  %add = add nsw i32 %tmp1, %tmp2
  store i32 %add, i32* %arrayidx13, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp ne i64 %indvars.iv.next, 1000
  br i1 %exitcond, label %for.body9, label %for.cond.cleanup8
}

declare double @fn1()
declare void @fn2(double)





;; After interchange %indvars.iv (j) should increment as the middle loop.
;; After interchange %indvars.iv42 (i) should increment with the inner most loop.

; CHECK-LABEL: @interchange_09

; CHECK: for.body:
; CHECK:   %indvars.iv45 = phi i64 [ %indvars.iv.next46, %for.cond.cleanup4 ], [ 0, %for.body.preheader ]
; CHECK:   %call = call double @fn1()
; CHECK:   %arrayidx = getelementptr inbounds [100 x double], [100 x double]* @T, i64 0, i64 %indvars.iv45
; CHECK:   store double %call, double* %arrayidx, align 8
; CHECK:   br label %for.body9.preheader

; CHECK: for.cond6.preheader.preheader:
; CHECK:   br label %for.cond6.preheader

; CHECK: for.cond6.preheader:
; CHECK:   %indvars.iv42 = phi i64 [ %indvars.iv.next43, %for.cond.cleanup8 ], [ 0, %for.cond6.preheader.preheader ]
; CHECK:   br label %for.body9.split1

; CHECK: for.body9.preheader:
; CHECK:   br label %for.body9

; CHECK: for.cond.cleanup4:
; CHECK:   %tmp = load double, double* %arrayidx, align 8
; CHECK:   call void @fn2(double %tmp)
; CHECK:   %indvars.iv.next46 = add nuw nsw i64 %indvars.iv45, 1
; CHECK:   %exitcond47 = icmp ne i64 %indvars.iv.next46, 100
; CHECK:   br i1 %exitcond47, label %for.body, label %for.cond.cleanup

; CHECK: for.cond.cleanup8:
; CHECK:   %indvars.iv.next43 = add nuw nsw i64 %indvars.iv42, 1
; CHECK:   %exitcond44 = icmp ne i64 %indvars.iv.next43, 1000
; CHECK:   br i1 %exitcond44, label %for.cond6.preheader, label %for.body9.split

; CHECK: for.body9:
; CHECK:   %indvars.iv = phi i64 [ %indvars.iv.next, %for.body9.split ], [ 1, %for.body9.preheader ]
; CHECK:   br label %for.cond6.preheader.preheader

; CHECK: for.body9.split1:
; CHECK:   %arrayidx13 = getelementptr inbounds [1000 x [1000 x i32]], [1000 x [1000 x i32]]* @Arr, i64 0, i64 %indvars.iv, i64 %indvars.iv42
; CHECK:   store i32 %add, i32* %arrayidx13, align 4
; CHECK:   br label %for.cond.cleanup8

; CHECK: for.body9.split:
; CHECK:   %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
; CHECK:   %exitcond = icmp ne i64 %indvars.iv.next, 1000
; CHECK:   br i1 %exitcond, label %for.body9, label %for.cond.cleanup4


;;-----------------------------------Test case 10-------------------------------
;; Test to make sure we can handle output dependencies.
;;
;;  for (int i = 0; i < 2; ++i)
;;    for(int j = 0; j < 3; ++j) {
;;      A[j][i] = i;
;;      A[j][i+1] = j;
;;    }

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

; CHECK-LABEL: @interchange_10
; CHECK: entry:
; CHECK:   br label %for.body4.preheader

; CHECK: for.cond1.preheader.preheader:
; CHECK:   br label %for.cond1.preheader

; CHECK: for.cond.loopexit:
; CHECK:   %exitcond28 = icmp ne i64 %indvars.iv.next27, 2
; CHECK:   br i1 %exitcond28, label %for.cond1.preheader, label %for.body4.split

; CHECK: for.cond1.preheader:
; CHECK:   %indvars.iv26 = phi i64 [ %indvars.iv.next27, %for.cond.loopexit ], [ 0, %for.cond1.preheader.preheader ]
; CHECK:   %indvars.iv.next27 = add nuw nsw i64 %indvars.iv26, 1
; CHECK:   br label %for.body4.split1

; CHECK: for.body4.preheader:
; CHECK:   br label %for.body4

; CHECK: for.cond.cleanup:
; CHECK:   ret void

; CHECK: for.body4:
; CHECK:   %indvars.iv = phi i64 [ %indvars.iv.next, %for.body4.split ], [ 0, %for.body4.preheader ]
; CHECK:   br label %for.cond1.preheader.preheader

; CHECK: for.body4.split1:
; CHECK:   %arrayidx6 = getelementptr inbounds [3 x [3 x i32]], [3 x [3 x i32]]* @A10, i64 0, i64 %indvars.iv, i64 %indvars.iv26
; CHECK:   %tmp = trunc i64 %indvars.iv26 to i32
; CHECK:   store i32 %tmp, i32* %arrayidx6, align 4
; CHECK:   %arrayidx10 = getelementptr inbounds [3 x [3 x i32]], [3 x [3 x i32]]* @A10, i64 0, i64 %indvars.iv, i64 %indvars.iv.next27
; CHECK:   %tmp1 = trunc i64 %indvars.iv to i32
; CHECK:   store i32 %tmp1, i32* %arrayidx10, align 4
; CHECK:   br label %for.cond.loopexit

; CHECK: for.body4.split:
; CHECK:   %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
; CHECK:   %exitcond = icmp ne i64 %indvars.iv.next, 3
; CHECK:   br i1 %exitcond, label %for.body4, label %for.cond.cleanup
