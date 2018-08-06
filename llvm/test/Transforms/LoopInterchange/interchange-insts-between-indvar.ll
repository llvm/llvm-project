; RUN: opt < %s -basicaa -loop-interchange -verify-dom-info -verify-loop-info \
; RUN:     -S -pass-remarks=loop-interchange 2>&1 | FileCheck %s

@A10 = local_unnamed_addr global [3 x [3 x i32]] zeroinitializer, align 16

;; Test to make sure we can handle zext instructions introduced by
;; IndVarSimplify.
;;
;;  for (int i = 0; i < 2; ++i)
;;    for(int j = 0; j < n; ++j) {
;;      A[j][i] = i;
;;    }

; CHECK: Loop interchanged with enclosing loop.

@A11 = local_unnamed_addr global [3 x [3 x i32]] zeroinitializer, align 16

define void @interchange_11(i32 %n) {
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
; The store below does not appear in the C snippet above.
; With two stores in the loop there may be WAW dependences, and interchange is illegal.
;  %arrayidx6 = getelementptr inbounds [3 x [3 x i32]], [3 x [3 x i32]]* @A10, i64 0, i64 %indvars.iv, i64 %indvars.iv26
;  %tmp = trunc i64 %indvars.iv26 to i32
;  store i32 %tmp, i32* %arrayidx6, align 4
  %arrayidx10 = getelementptr inbounds [3 x [3 x i32]], [3 x [3 x i32]]* @A10, i64 0, i64 %indvars.iv, i64 %indvars.iv.next27
  %tmp1 = trunc i64 %indvars.iv to i32
  store i32 %tmp1, i32* %arrayidx10, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %n.wide = zext i32 %n to i64
  %exitcond = icmp ne i64 %indvars.iv.next, %n.wide
  br i1 %exitcond, label %for.body4, label %for.cond.loopexit
}
