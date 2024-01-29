; RUN: opt < %s -passes=loop-vectorize -force-vector-width=2 -S | FileCheck %s

@aa = global [256 x [256 x float]] zeroinitializer, align 4
@a = global [32000 x float] zeroinitializer, align 4

;; Given that SCEV of &a[j] is {@a,+,4}<Loop j>, a[j] will be treated as scalar
;; when vectorizing Loop i. If the accessing size of a[j] <= Dist(a[j], a[i]),
;; there is no overlapped and can be vectorized.
;;
;; In this case, accessing size of a[j] is 4 byte(float) and Dist(a[j], a[i])
;; is {4,+,4} which bring the minimum distance as 4.
;;
;; for (int j = 0; j < 256; j++)    // Loop j
;;   for (int i = j+1; i < 256; i++)// Loop i
;;     a[i] -= aa[j][i] * a[j];

; CHECK: vector.body:

define signext i32 @s115() {
entry:
  br label %for.body

for.cond.loopexit.loopexit:                       ; preds = %for.body4
  br label %for.cond.loopexit

for.cond.loopexit:                                ; preds = %for.cond.loopexit.loopexit, %for.body
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond28.not = icmp eq i64 %indvars.iv.next27, 256
  br i1 %exitcond28.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.cond.loopexit
  ret i32 0

for.body:                                         ; preds = %entry, %for.cond.loopexit
  %indvars.iv26 = phi i64 [ 0, %entry ], [ %indvars.iv.next27, %for.cond.loopexit ]
  %indvars.iv = phi i64 [ 1, %entry ], [ %indvars.iv.next, %for.cond.loopexit ]
  %indvars.iv.next27 = add nuw nsw i64 %indvars.iv26, 1
  %cmp221 = icmp ult i64 %indvars.iv26, 255
  br i1 %cmp221, label %for.body4.lr.ph, label %for.cond.loopexit

for.body4.lr.ph:                                  ; preds = %for.body
  %arrayidx8 = getelementptr inbounds [32000 x float], ptr @a, i64 0, i64 %indvars.iv26
  br label %for.body4

for.body4:                                        ; preds = %for.body4.lr.ph, %for.body4
  %indvars.iv24 = phi i64 [ %indvars.iv, %for.body4.lr.ph ], [ %indvars.iv.next25, %for.body4 ]
  %arrayidx6 = getelementptr inbounds [256 x [256 x float]], ptr @aa, i64 0, i64 %indvars.iv26, i64 %indvars.iv24
  %0 = load float, ptr %arrayidx6, align 4
  %1 = load float, ptr %arrayidx8, align 4
  %arrayidx10 = getelementptr inbounds [32000 x float], ptr @a, i64 0, i64 %indvars.iv24
  %2 = load float, ptr %arrayidx10, align 4
  %neg = fneg float %0
  %3 = tail call float @llvm.fmuladd.f32(float %neg, float %1, float %2)
  store float %3, ptr %arrayidx10, align 4
  %indvars.iv.next25 = add nuw nsw i64 %indvars.iv24, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next25, 256
  br i1 %exitcond.not, label %for.cond.loopexit.loopexit, label %for.body4
}
