; RUN: opt -passes=loop-vectorize -force-vector-interleave=1 -force-vector-width=4 -S < %s | FileCheck %s --check-prefix=CHECK
; RUN: opt -passes=loop-vectorize -force-vector-interleave=4 -force-vector-width=4 -S < %s | FileCheck %s --check-prefix=CHECK
; RUN: opt -passes=loop-vectorize -force-vector-interleave=4 -force-vector-width=1 -S < %s | FileCheck %s --check-prefix=CHECK

; This test can theoretically be vectorized without a runtime-check, by
; pattern-matching on the constructs that are introduced by IndVarSimplify.
; We can check two things:
;   %1 = trunc i64 %iv to i32
; This indicates that the %iv is truncated to i32. We can then check the loop
; guard is a signed i32:
;   %cmp.sgt = icmp sgt i32 %n, 0
; and successfully vectorize the case without a runtime-check.
define i32 @select_icmp_const_truncated_iv_widened_exit(ptr %a, i32 %n) {
; CHECK-LABEL: define i32 @select_icmp_const_truncated_iv_widened_exit
; CHECK-NOT:   vector.body:
;
entry:
  %cmp.sgt = icmp sgt i32 %n, 0
  br i1 %cmp.sgt, label %for.body.preheader, label %exit

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
  %iv = phi i64 [ 0, %for.body.preheader ], [ %inc, %for.body ]
  %rdx = phi i32 [ 331, %for.body.preheader ], [ %spec.select, %for.body ]
  %arrayidx = getelementptr inbounds i64, ptr %a, i64 %iv
  %0 = load i64, ptr %arrayidx, align 8
  %cmp = icmp sgt i64 %0, 3
  %1 = trunc i64 %iv to i32
  %spec.select = select i1 %cmp, i32 %1, i32 %rdx
  %inc = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %inc, %wide.trip.count
  br i1 %exitcond.not, label %exit, label %for.body

exit:                                            ; preds = %for.body, %entry
  %rdx.lcssa = phi i32 [ 331, %entry ], [ %spec.select, %for.body ]
  ret i32 %rdx.lcssa
}

; This test can theoretically be vectorized without a runtime-check, by
; pattern-matching on the constructs that are introduced by IndVarSimplify.
; We can check two things:
;   %1 = trunc i64 %iv to i32
; This indicates that the %iv is truncated to i32. We can then check the loop
; exit condition, which compares to a constant that fits within i32:
;   %exitcond.not = icmp eq i64 %inc, 20000
; and successfully vectorize the case without a runtime-check.
define i32 @select_icmp_const_truncated_iv_const_exit(ptr %a) {
; CHECK-LABEL: define i32 @select_icmp_const_truncated_iv_const_exit
; CHECK-NOT:   vector.body:
;
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %iv = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %rdx = phi i32 [ 331, %entry ], [ %spec.select, %for.body ]
  %arrayidx = getelementptr inbounds i64, ptr %a, i64 %iv
  %0 = load i64, ptr %arrayidx, align 8
  %cmp = icmp sgt i64 %0, 3
  %1 = trunc i64 %iv to i32
  %spec.select = select i1 %cmp, i32 %1, i32 %rdx
  %inc = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %inc, 20000
  br i1 %exitcond.not, label %exit, label %for.body

exit:                                           ; preds = %for.body
  ret i32 %spec.select
}

; Without loop guard, the maximum constant trip count that can be vectorized is
; the signed maximum value of reduction type.
define i32 @select_fcmp_max_valid_const_ub(ptr %a) {
; CHECK-LABEL: define i32 @select_fcmp_max_valid_const_ub
; CHECK-NOT:   vector.body:
;
entry:
  br label %for.body

for.body:                                        ; preds = %entry, %for.body
  %iv = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %rdx = phi i32 [ -1, %entry ], [ %spec.select, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %a, i64 %iv
  %0 = load float, ptr %arrayidx, align 4
  %cmp = fcmp fast olt float %0, 0.000000e+00
  %1 = trunc i64 %iv to i32
  %spec.select = select i1 %cmp, i32 %1, i32 %rdx
  %inc = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %inc, 2147483648
  br i1 %exitcond.not, label %exit, label %for.body

exit:                                            ; preds = %for.body
  ret i32 %spec.select
}

; Negative tests

; This test can theoretically be vectorized, but only with a runtime-check.
; The construct that are introduced by IndVarSimplify is:
;   %1 = trunc i64 %iv to i32
; However, the loop guard is an i64:
;   %cmp.sgt = icmp sgt i64 %n, 0
; We cannot guarantee that %iv won't overflow an i32 value (and hence hit the
; sentinel value), and need a runtime-check to vectorize this case.
define i32 @not_vectorized_select_icmp_const_truncated_iv_unwidened_exit(ptr %a, i64 %n) {
; CHECK-LABEL: define i32 @not_vectorized_select_icmp_const_truncated_iv_unwidened_exit
; CHECK-NOT:   vector.body:
;
entry:
  %cmp.sgt = icmp sgt i64 %n, 0
  br i1 %cmp.sgt, label %for.body, label %exit

for.body:                                         ; preds = %entry, %for.body
  %iv = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %rdx = phi i32 [ 331, %entry ], [ %spec.select, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %a, i64 %iv
  %0 = load i32, ptr %arrayidx, align 4
  %cmp = icmp sgt i32 %0, 3
  %1 = trunc i64 %iv to i32
  %spec.select = select i1 %cmp, i32 %1, i32 %rdx
  %inc = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %inc, %n
  br i1 %exitcond.not, label %exit, label %for.body

exit:                                             ; preds = %for.body, %entry
  %rdx.lcssa = phi i32 [ 331, %entry ], [ %spec.select, %for.body ]
  ret i32 %rdx.lcssa
}

; This test can theoretically be vectorized, but only with a runtime-check.
; The construct that are introduced by IndVarSimplify is:
;   %1 = trunc i64 %iv to i32
; However, the loop guard is unsigned:
;   %cmp.not = icmp eq i32 %n, 0
; We cannot guarantee that %iv won't overflow an i32 value (and hence hit the
; sentinel value), and need a runtime-check to vectorize this case.
define i32 @not_vectorized_select_icmp_const_truncated_iv_unsigned_loop_guard(ptr %a, i32 %n) {
; CHECK-LABEL: define i32 @not_vectorized_select_icmp_const_truncated_iv_unsigned_loop_guard
; CHECK-NOT:   vector.body:
;
entry:
  %cmp.not = icmp eq i32 %n, 0
  br i1 %cmp.not, label %exit, label %for.body.preheader

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
  %iv = phi i64 [ 0, %for.body.preheader ], [ %inc, %for.body ]
  %rdx = phi i32 [ 331, %for.body.preheader ], [ %spec.select, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %a, i64 %iv
  %0 = load i32, ptr %arrayidx, align 4
  %cmp1 = icmp sgt i32 %0, 3
  %1 = trunc i64 %iv to i32
  %spec.select = select i1 %cmp1, i32 %1, i32 %rdx
  %inc = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %inc, %wide.trip.count
  br i1 %exitcond.not, label %exit, label %for.body

exit:                                             ; preds = %for.body, %entry
  %rdx.lcssa = phi i32 [ 331, %entry ], [ %spec.select, %for.body ]
  ret i32 %rdx.lcssa
}

; This test cannot be vectorized, even with a runtime check.
; The construct that are introduced by IndVarSimplify is:
;   %1 = trunc i64 %iv to i32
; However, the loop exit condition is a constant that overflows i32:
;   %exitcond.not = icmp eq i64 %inc, 4294967294
; Hence, the i32 will most certainly wrap and hit the sentinel value, and we
; cannot vectorize this case.
define i32 @not_vectorized_select_icmp_truncated_iv_out_of_bound(ptr %a) {
; CHECK-LABEL: define i32 @not_vectorized_select_icmp_truncated_iv_out_of_bound
; CHECK-NOT:   vector.body:
;
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %iv = phi i64 [ 2147483646, %entry ], [ %inc, %for.body ]
  %rdx = phi i32 [ 331, %entry ], [ %spec.select, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %a, i64 %iv
  %0 = load i32, ptr %arrayidx, align 4
  %cmp = icmp sgt i32 %0, 3
  %conv = trunc i64 %iv to i32
  %spec.select = select i1 %cmp, i32 %conv, i32 %rdx
  %inc = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %inc, 4294967294
  br i1 %exitcond.not, label %exit, label %for.body

exit:                                             ; preds = %for.body
  ret i32 %spec.select
}

; Forbidding vectorization of the FindLastIV pattern involving a truncated
; induction variable in the absence of any loop guard.
define i32 @not_vectorized_select_iv_icmp_no_guard(ptr %a, ptr %b, i32 %start, i32 %n) {
; CHECK-LABEL: define i32 @not_vectorized_select_iv_icmp_no_guard
; CHECK-NOT:   vector.body:
;
entry:
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %iv = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %rdx = phi i32 [ %start, %entry ], [ %cond, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %a, i64 %iv
  %0 = load i32, ptr %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, ptr %b, i64 %iv
  %1 = load i32, ptr %arrayidx2, align 4
  %cmp = icmp sgt i32 %0, %1
  %2 = trunc i64 %iv to i32
  %cond = select i1 %cmp, i32 %2, i32 %rdx
  %inc = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %inc, %wide.trip.count
  br i1 %exitcond.not, label %exit, label %for.body

exit:                                             ; preds = %for.body
  ret i32 %cond
}

; Without loop guard, when the constant trip count exceeds the maximum signed
; value of the reduction type, truncation may cause overflow. Therefore,
; vectorizer is unable to guarantee that the induction variable is monotonic
; increasing.
define i32 @not_vectorized_select_fcmp_invalid_const_ub(ptr %a) {
; CHECK-LABEL: define i32 @not_vectorized_select_fcmp_invalid_const_ub
; CHECK-NOT:   vector.body:
;
entry:
  br label %for.body

for.body:                                        ; preds = %entry, %for.body
  %iv = phi i64 [ 0, %entry ], [ %inc, %for.body ]
  %rdx = phi i32 [ -1, %entry ], [ %spec.select, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %a, i64 %iv
  %0 = load float, ptr %arrayidx, align 4
  %cmp = fcmp fast olt float %0, 0.000000e+00
  %1 = trunc i64 %iv to i32
  %spec.select = select i1 %cmp, i32 %1, i32 %rdx
  %inc = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %inc, 2147483649
  br i1 %exitcond.not, label %exit, label %for.body

exit:                                            ; preds = %for.body
  ret i32 %spec.select
}

; Even with loop guard protection, if the destination type of the truncation
; instruction is smaller than the trip count type before extension, overflow
; could still occur.
define i16 @not_vectorized_select_iv_icmp_overflow_unwidened_tripcount(ptr %a, ptr %b, i16 %start, i32 %n) {
; CHECK-LABEL: define i16 @not_vectorized_select_iv_icmp_overflow_unwidened_tripcount
; CHECK-NOT:   vector.body:
;
entry:
  %cmp9 = icmp sgt i32 %n, 0
  br i1 %cmp9, label %for.body.preheader, label %exit

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
  %iv = phi i64 [ 0, %for.body.preheader ], [ %inc, %for.body ]
  %rdx = phi i16 [ %start, %for.body.preheader ], [ %cond, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %a, i64 %iv
  %0 = load i32, ptr %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, ptr %b, i64 %iv
  %1 = load i32, ptr %arrayidx2, align 4
  %cmp3 = icmp sgt i32 %0, %1
  %2 = trunc i64 %iv to i16
  %cond = select i1 %cmp3, i16 %2, i16 %rdx
  %inc = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %inc, %wide.trip.count
  br i1 %exitcond.not, label %exit, label %for.body

exit:                                             ; preds = %for.body, %entry
  %rdx.0.lcssa = phi i16 [ %start, %entry ], [ %cond, %for.body ]
  ret i16 %rdx.0.lcssa
}
