; RUN: opt -passes=loop-vectorize -force-vector-interleave=1 -force-vector-width=4 -S < %s | FileCheck %s --check-prefix=CHECK
; RUN: opt -passes=loop-vectorize -force-vector-interleave=4 -force-vector-width=4 -S < %s | FileCheck %s --check-prefix=CHECK
; RUN: opt -passes=loop-vectorize -force-vector-interleave=4 -force-vector-width=1 -S < %s | FileCheck %s --check-prefix=CHECK

define i64 @select_icmp_const_1(ptr nocapture readonly %a, i64 %n) {
; CHECK-LABEL: define i64 @select_icmp_const_1
; CHECK-NOT:   vector.body:
;
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %iv = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %rdx = phi i64 [ %cond, %for.body ], [ 3, %entry ]
  %arrayidx = getelementptr inbounds i64, ptr %a, i64 %iv
  %0 = load i64, ptr %arrayidx, align 8
  %cmp2 = icmp eq i64 %0, 3
  %cond = select i1 %cmp2, i64 %iv, i64 %rdx
  %inc = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %inc, %n
  br i1 %exitcond.not, label %exit, label %for.body

exit:                                             ; preds = %for.body
  ret i64 %cond
}

define i64 @select_icmp_const_2(ptr nocapture readonly %a, i64 %n) {
; CHECK-LABEL: define i64 @select_icmp_const_2
; CHECK-NOT:   vector.body:
;
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %iv = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %rdx = phi i64 [ %cond, %for.body ], [ 3, %entry ]
  %arrayidx = getelementptr inbounds i64, ptr %a, i64 %iv
  %0 = load i64, ptr %arrayidx, align 8
  %cmp2 = icmp eq i64 %0, 3
  %cond = select i1 %cmp2, i64 %rdx, i64 %iv
  %inc = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %inc, %n
  br i1 %exitcond.not, label %exit, label %for.body

exit:                                             ; preds = %for.body
  ret i64 %cond
}

define i64 @select_icmp_const_3_variable_rdx_start(ptr nocapture readonly %a, i64 %rdx.start, i64 %n) {
; CHECK-LABEL: define i64 @select_icmp_const_3_variable_rdx_start
; CHECK-NOT:   vector.body:
;
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %iv = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %rdx = phi i64 [ %cond, %for.body ], [ %rdx.start, %entry ]
  %arrayidx = getelementptr inbounds i64, ptr %a, i64 %iv
  %0 = load i64, ptr %arrayidx, align 8
  %cmp2 = icmp eq i64 %0, 3
  %cond = select i1 %cmp2, i64 %iv, i64 %rdx
  %inc = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %inc, %n
  br i1 %exitcond.not, label %exit, label %for.body

exit:                                             ; preds = %for.body
  ret i64 %cond
}

define i64 @select_fcmp_const_fast(ptr nocapture readonly %a, i64 %n) {
; CHECK-LABEL: define i64 @select_fcmp_const_fast
; CHECK-NOT:   vector.body:
;
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %iv = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %rdx = phi i64 [ %cond, %for.body ], [ 2, %entry ]
  %arrayidx = getelementptr inbounds float, ptr %a, i64 %iv
  %0 = load float, ptr %arrayidx, align 4
  %cmp2 = fcmp fast ueq float %0, 3.0
  %cond = select i1 %cmp2, i64 %iv, i64 %rdx
  %inc = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %inc, %n
  br i1 %exitcond.not, label %exit, label %for.body

exit:                                             ; preds = %for.body
  ret i64 %cond
}

define i64 @select_fcmp_const(ptr nocapture readonly %a, i64 %n) {
; CHECK-LABEL: define i64 @select_fcmp_const
; CHECK-NOT:   vector.body:
;
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %iv = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %rdx = phi i64 [ %cond, %for.body ], [ 2, %entry ]
  %arrayidx = getelementptr inbounds float, ptr %a, i64 %iv
  %0 = load float, ptr %arrayidx, align 4
  %cmp2 = fcmp ueq float %0, 3.0
  %cond = select i1 %cmp2, i64 %iv, i64 %rdx
  %inc = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %inc, %n
  br i1 %exitcond.not, label %exit, label %for.body

exit:                                             ; preds = %for.body
  ret i64 %cond
}

define i64 @select_icmp(ptr nocapture readonly %a, ptr nocapture readonly %b, i64 %rdx.start, i64 %n) {
; CHECK-LABEL: define i64 @select_icmp
; CHECK-NOT:   vector.body:
;
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %iv = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %rdx = phi i64 [ %cond, %for.body ], [ %rdx.start, %entry ]
  %arrayidx = getelementptr inbounds i64, ptr %a, i64 %iv
  %0 = load i64, ptr %arrayidx, align 8
  %arrayidx1 = getelementptr inbounds i64, ptr %b, i64 %iv
  %1 = load i64, ptr %arrayidx1, align 8
  %cmp2 = icmp sgt i64 %0, %1
  %cond = select i1 %cmp2, i64 %iv, i64 %rdx
  %inc = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %inc, %n
  br i1 %exitcond.not, label %exit, label %for.body

exit:                                             ; preds = %for.body
  ret i64 %cond
}

define i64 @select_fcmp(ptr nocapture readonly %a, ptr nocapture readonly %b, i64 %rdx.start, i64 %n) {
; CHECK-LABEL: define i64 @select_fcmp
; CHECK-NOT:   vector.body:
;
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %iv = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %rdx = phi i64 [ %cond, %for.body ], [ %rdx.start, %entry ]
  %arrayidx = getelementptr inbounds float, ptr %a, i64 %iv
  %0 = load float, ptr %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds float, ptr %b, i64 %iv
  %1 = load float, ptr %arrayidx1, align 4
  %cmp2 = fcmp ogt float %0, %1
  %cond = select i1 %cmp2, i64 %iv, i64 %rdx
  %inc = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %inc, %n
  br i1 %exitcond.not, label %exit, label %for.body

exit:                                             ; preds = %for.body
  ret i64 %cond
}

define i64 @select_icmp_min_valid_iv_start(ptr nocapture readonly %a, ptr nocapture readonly %b, i64 %rdx.start, i64 %n) {
; CHECK-LABEL: define i64 @select_icmp_min_valid_iv_start
; CHECK-NOT:   vector.body:
;
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %iv.j = phi i64 [ %inc3, %for.body ], [ -9223372036854775807, %entry]
  %iv.i = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %rdx = phi i64 [ %cond, %for.body ], [ %rdx.start, %entry ]
  %arrayidx = getelementptr inbounds i64, ptr %a, i64 %iv.i
  %0 = load i64, ptr %arrayidx, align 8
  %arrayidx1 = getelementptr inbounds i64, ptr %b, i64 %iv.i
  %1 = load i64, ptr %arrayidx1, align 8
  %cmp2 = icmp sgt i64 %0, %1
  %cond = select i1 %cmp2, i64 %iv.j, i64 %rdx
  %inc = add nuw nsw i64 %iv.i, 1
  %inc3 = add nsw i64 %iv.j, 1
  %exitcond.not = icmp eq i64 %inc, %n
  br i1 %exitcond.not, label %exit, label %for.body

exit:                                             ; preds = %for.body
  ret i64 %cond
}

; Negative tests

define float @not_vectorized_select_float_induction_icmp(ptr nocapture readonly %a, ptr nocapture readonly %b, float %rdx.start, i64 %n) {
; CHECK-LABEL: @not_vectorized_select_float_induction_icmp
; CHECK-NOT:   vector.body:
;
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %iv = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %fiv = phi float [ %conv3, %for.body ], [ 0.000000e+00, %entry ]
  %rdx = phi float [ %cond, %for.body ], [ %rdx.start, %entry ]
  %arrayidx = getelementptr inbounds i64, ptr %a, i64 %iv
  %0 = load i64, ptr %arrayidx, align 8
  %arrayidx1 = getelementptr inbounds i64, ptr %b, i64 %iv
  %1 = load i64, ptr %arrayidx1, align 8
  %cmp2 = icmp sgt i64 %0, %1
  %cond = select i1 %cmp2, float %fiv, float %rdx
  %conv3 = fadd float %fiv, 1.000000e+00
  %inc = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %inc, %n
  br i1 %exitcond.not, label %exit, label %for.body

exit:                                             ; preds = %for.body
  ret float %cond
}

define i64 @not_vectorized_select_decreasing_induction_icmp(ptr nocapture readonly %a, ptr nocapture readonly %b, i64 %rdx.start, i64 %n) {
; CHECK-LABEL: @not_vectorized_select_decreasing_induction_icmp
; CHECK-NOT:   vector.body:
;
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.0.in10 = phi i64 [ %iv, %for.body ], [ %n, %entry ]
  %rdx = phi i64 [ %cond, %for.body ], [ %rdx.start, %entry ]
  %iv = add nsw i64 %i.0.in10, -1
  %arrayidx = getelementptr inbounds i64, ptr %a, i64 %iv
  %0 = load i64, ptr %arrayidx, align 8
  %arrayidx1 = getelementptr inbounds i64, ptr %b, i64 %iv
  %1 = load i64, ptr %arrayidx1, align 8
  %cmp2 = icmp sgt i64 %0, %1
  %cond = select i1 %cmp2, i64 %iv, i64 %rdx
  %cmp = icmp ugt i64 %i.0.in10, 1
  br i1 %cmp, label %for.body, label %exit

exit:                                             ; preds = %for.body
  ret i64 %cond
}

define i64 @not_vectorized_select_icmp_iv_out_of_bound(ptr nocapture readonly %a, ptr nocapture readonly %b, i64 %rdx.start, i64 %n) {
; CHECK-LABEL: @not_vectorized_select_icmp_iv_out_of_bound
; CHECK-NOT:   vector.body:
;
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %iv.j = phi i64 [ %inc3, %for.body ], [ -9223372036854775808, %entry]
  %iv.i = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %rdx = phi i64 [ %cond, %for.body ], [ %rdx.start, %entry ]
  %arrayidx = getelementptr inbounds i64, ptr %a, i64 %iv.i
  %0 = load i64, ptr %arrayidx, align 8
  %arrayidx1 = getelementptr inbounds i64, ptr %b, i64 %iv.i
  %1 = load i64, ptr %arrayidx1, align 8
  %cmp2 = icmp sgt i64 %0, %1
  %cond = select i1 %cmp2, i64 %iv.j, i64 %rdx
  %inc = add nuw nsw i64 %iv.i, 1
  %inc3 = add nsw i64 %iv.j, 1
  %exitcond.not = icmp eq i64 %inc, %n
  br i1 %exitcond.not, label %exit, label %for.body

exit:                                             ; preds = %for.body
  ret i64 %cond
}

define i64 @not_vectorized_select_icmp_non_const_iv_start_value(ptr nocapture readonly %a, ptr nocapture readonly %b, i64 %ivstart, i64 %rdx.start, i64 %n) {
; CHECK-LABEL: define i64 @not_vectorized_select_icmp_non_const_iv_start_value
; CHECK-NOT:   vector.body:
;
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %iv = phi i64 [ %inc, %for.body ], [ %ivstart, %entry ]
  %rdx = phi i64 [ %cond, %for.body ], [ %rdx.start, %entry ]
  %arrayidx = getelementptr inbounds i64, ptr %a, i64 %iv
  %0 = load i64, ptr %arrayidx, align 8
  %arrayidx1 = getelementptr inbounds i64, ptr %b, i64 %iv
  %1 = load i64, ptr %arrayidx1, align 8
  %cmp2 = icmp sgt i64 %0, %1
  %cond = select i1 %cmp2, i64 %iv, i64 %rdx
  %inc = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %inc, %n
  br i1 %exitcond.not, label %exit, label %for.body

exit:                                             ; preds = %for.body
  ret i64 %cond
}
