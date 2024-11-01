; RUN: opt < %s -passes=loop-interchange -cache-line-size=64 -pass-remarks-missed='loop-interchange' -pass-remarks-output=%t -S \
; RUN:     -verify-dom-info -verify-loop-info -verify-loop-lcssa 2>&1 | FileCheck -check-prefix=IR %s
; RUN: FileCheck --input-file=%t %s

; Outer loop only reductions are not supported currently.

@A = common global [500 x [500 x i32]] zeroinitializer

;; global X

;;  for( int i=1;i<N;i++) {
;;    for( int j=1;j<N;j++)
;;      ;
;;    X+=A[j][i];
;;  }

; CHECK: --- !Missed
; CHECK-NEXT: Pass:            loop-interchange
; CHECK-NEXT: Name:            UnsupportedPHI
; CHECK-NEXT: Function:        reduction_01

; IR-LABEL: @reduction_01(
; IR-NOT: split

define i32 @reduction_01(i32 %N) {
entry:
  br label %outer.header

outer.header:                                  ; preds = %for.cond1.for.inc6_crit_edge, %entry
  %indvars.iv18 = phi i64 [ %indvars.iv.next19, %outer.inc ], [ 1, %entry ]
  %add15 = phi i32 [ 0, %entry ], [ %add, %outer.inc ]
  br label %for.body3

for.body3:                                        ; preds = %for.body3, %for.body3.lr.ph
  %indvars.iv = phi i64 [ 1, %outer.header ], [ %indvars.iv.next, %for.body3 ]
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %N
  br i1 %exitcond, label %outer.inc, label %for.body3

outer.inc:                     ; preds = %for.body3
  %arrayidx5 = getelementptr inbounds [500 x [500 x i32]], [500 x [500 x i32]]* @A, i64 0, i64 %indvars.iv, i64 %indvars.iv18
  %0 = load i32, i32* %arrayidx5
  %add = add nsw i32 %add15, %0
  %indvars.iv.next19 = add nuw nsw i64 %indvars.iv18, 1
  %lftr.wideiv20 = trunc i64 %indvars.iv.next19 to i32
  %exitcond21 = icmp eq i32 %lftr.wideiv20, %N
  br i1 %exitcond21, label %for.end8, label %outer.header

for.end8:                                         ; preds = %for.cond1.for.inc6_crit_edge, %entry
  ret i32 %add
}
