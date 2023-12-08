; RUN: opt -passes=loop-vectorize -force-vector-width=4 -force-vector-interleave=1 -S < %s | FileCheck %s --check-prefix=CHECK
; RUN: opt -passes=loop-vectorize -force-vector-width=4 -force-vector-interleave=4 -S < %s | FileCheck %s --check-prefix=CHECK
; RUN: opt -passes=loop-vectorize -force-vector-width=1 -force-vector-interleave=4 -S < %s | FileCheck %s --check-prefix=CHECK

define i64 @smax_idx(ptr nocapture readonly %a, i64 %mm, i64 %ii, ptr nocapture writeonly %res_max, i64 %n) {
; CHECK-LABEL: @smax_idx(
; CHECK-NOT:   vector.body:
;
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %max.09 = phi i64 [ %mm, %entry ], [ %1, %for.body ]
  %idx.011 = phi i64 [ %ii, %entry ], [ %spec.select7, %for.body ]
  %arrayidx = getelementptr inbounds i64, ptr %a, i64 %indvars.iv
  %0 = load i64, ptr %arrayidx
  %1 = tail call i64 @llvm.smax.i64(i64 %max.09, i64 %0)
  %cmp1 = icmp slt i64 %max.09, %0
  %spec.select7 = select i1 %cmp1, i64 %indvars.iv, i64 %idx.011
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %n
  br i1 %exitcond.not, label %exit, label %for.body

exit:
  store i64 %1, ptr %res_max
  ret i64 %spec.select7
}

;
; Check the different order of reduction phis.
;
define i64 @smax_idx_inverted_phi(ptr nocapture readonly %a, i64 %mm, i64 %ii, ptr nocapture writeonly %res_max, i64 %n) {
; CHECK-LABEL: @smax_idx_inverted_phi(
; CHECK-NOT:   vector.body:
;
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %idx.011 = phi i64 [ %ii, %entry ], [ %spec.select7, %for.body ]
  %max.09 = phi i64 [ %mm, %entry ], [ %1, %for.body ]
  %arrayidx = getelementptr inbounds i64, ptr %a, i64 %indvars.iv
  %0 = load i64, ptr %arrayidx
  %1 = tail call i64 @llvm.smax.i64(i64 %max.09, i64 %0)
  %cmp1 = icmp slt i64 %max.09, %0
  %spec.select7 = select i1 %cmp1, i64 %indvars.iv, i64 %idx.011
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %n
  br i1 %exitcond.not, label %exit, label %for.body

exit:
  store i64 %1, ptr %res_max
  ret i64 %spec.select7
}

; Check if it is a min/max with index (MMI) pattern when the
; min/max value is not used outside the loop.
;
; Currently, the vectorizer checks if smax value is used outside
; the loop. However, even if only the index part has external users,
; and smax itself does not have external users, it can still form a
; MMI pattern.
;
define i64 @smax_idx_max_no_exit_user(ptr nocapture readonly %a, i64 %mm, i64 %ii, i64 %n) {
; CHECK-LABEL: @smax_idx_max_no_exit_user(
; CHECK-NOT:   vector.body:
;
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %max.09 = phi i64 [ %mm, %entry ], [ %1, %for.body ]
  %idx.011 = phi i64 [ %ii, %entry ], [ %spec.select7, %for.body ]
  %arrayidx = getelementptr inbounds i64, ptr %a, i64 %indvars.iv
  %0 = load i64, ptr %arrayidx
  %1 = tail call i64 @llvm.smax.i64(i64 %max.09, i64 %0)
  %cmp1 = icmp slt i64 %max.09, %0
  %spec.select7 = select i1 %cmp1, i64 %indvars.iv, i64 %idx.011
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %n
  br i1 %exitcond.not, label %exit, label %for.body

exit:
  ; %1 has no external users
  ret i64 %spec.select7
}

; Check smax implemented by select(cmp()).
;
; Currently, MMI pattern does not support icmp with multiple users.
; TODO: It may be possible to reuse some of the methods in instcombine pass to
; check whether icmp can be duplicated.
;
define i64 @smax_idx_select_cmp(ptr nocapture readonly %a, i64 %mm, i64 %ii, ptr nocapture writeonly %res_max, i64 %n) {
; CHECK-LABEL: @smax_idx_select_cmp(
; CHECK-NOT:   vector.body:
;
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %max.09 = phi i64 [ %mm, %entry ], [ %spec.select, %for.body ]
  %idx.011 = phi i64 [ %ii, %entry ], [ %spec.select7, %for.body ]
  %arrayidx = getelementptr inbounds i64, ptr %a, i64 %indvars.iv
  %0 = load i64, ptr %arrayidx
  %cmp1 = icmp slt i64 %max.09, %0
  %spec.select = select i1 %cmp1, i64 %0, i64 %max.09
  %spec.select7 = select i1 %cmp1, i64 %indvars.iv, i64 %idx.011
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %n
  br i1 %exitcond.not, label %exit, label %for.body

exit:
  store i64 %spec.select, ptr %res_max
  ret i64 %spec.select7
}

;
; Check sge case.
;
define i64 @smax_idx_inverted_pred(ptr nocapture readonly %a, i64 %mm, i64 %ii, ptr nocapture writeonly %res_max, i64 %n) {
; CHECK-LABEL: @smax_idx_inverted_pred(
; CHECK-NOT:   vector.body:
;
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %max.09 = phi i64 [ %mm, %entry ], [ %1, %for.body ]
  %idx.011 = phi i64 [ %ii, %entry ], [ %spec.select7, %for.body ]
  %arrayidx = getelementptr inbounds i64, ptr %a, i64 %indvars.iv
  %0 = load i64, ptr %arrayidx
  %1 = tail call i64 @llvm.smax.i64(i64 %max.09, i64 %0)
  %cmp1 = icmp sge i64 %0, %max.09
  %spec.select7 = select i1 %cmp1, i64 %indvars.iv, i64 %idx.011
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %n
  br i1 %exitcond.not, label %exit, label %for.body

exit:
  store i64 %1, ptr %res_max
  ret i64 %spec.select7
}

;
; In such cases, the last index should be extracted.
;
define i64 @smax_idx_extract_last(ptr nocapture readonly %a, i64 %mm, i64 %ii, ptr nocapture writeonly %res_max, i64 %n) {
; CHECK-LABEL: @smax_idx_extract_last(
; CHECK-NOT:   vector.body:
;
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %max.09 = phi i64 [ %mm, %entry ], [ %1, %for.body ]
  %idx.011 = phi i64 [ %ii, %entry ], [ %spec.select7, %for.body ]
  %arrayidx = getelementptr inbounds i64, ptr %a, i64 %indvars.iv
  %0 = load i64, ptr %arrayidx
  %1 = tail call i64 @llvm.smax.i64(i64 %max.09, i64 %0)
  %cmp1.not = icmp sgt i64 %max.09, %0
  %spec.select7 = select i1 %cmp1.not, i64 %idx.011, i64 %indvars.iv
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %n
  br i1 %exitcond.not, label %exit, label %for.body

exit:
  store i64 %1, ptr %res_max
  ret i64 %spec.select7
}

;
; The operands of smax intrinsic and icmp are not the same to be recognized as MMI.
;
define i64 @smax_idx_not_vec_1(ptr nocapture readonly %a, ptr nocapture readonly %b, i64 %mm, i64 %ii, ptr nocapture writeonly %res_max, i64 %n) {
; CHECK-LABEL: @smax_idx_not_vec_1(
; CHECK-NOT:   vector.body:
;
  entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %max.09 = phi i64 [ %mm, %entry ], [ %2, %for.body ]
  %idx.011 = phi i64 [ %ii, %entry ], [ %spec.select7, %for.body ]
  %arrayidx = getelementptr inbounds i64, ptr %a, i64 %indvars.iv
  %0 = load i64, ptr %arrayidx
  %arrayidx.01 = getelementptr inbounds i64, ptr %b, i64 %indvars.iv
  %1 = load i64, ptr %arrayidx
  %2 = tail call i64 @llvm.smax.i64(i64 %max.09, i64 %0)
  %cmp1 = icmp slt i64 %max.09, %1
  %spec.select7 = select i1 %cmp1, i64 %indvars.iv, i64 %idx.011
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %n
  br i1 %exitcond.not, label %exit, label %for.body

exit:
  store i64 %2, ptr %res_max
  ret i64 %spec.select7
}

;
; It cannot be recognized as MMI when the operand of index select is not an induction variable.
;
define i64 @smax_idx_not_vec_2(ptr nocapture readonly %a, i64 %mm, i64 %ii, ptr nocapture writeonly %res_max, i64 %n) {
; CHECK-LABEL: @smax_idx_not_vec_2(
; CHECK-NOT:   vector.body:
;
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %max.09 = phi i64 [ %mm, %entry ], [ %1, %for.body ]
  %idx.011 = phi i64 [ %ii, %entry ], [ %spec.select7, %for.body ]
  %arrayidx = getelementptr inbounds i64, ptr %a, i64 %indvars.iv
  %0 = load i64, ptr %arrayidx
  %1 = tail call i64 @llvm.smax.i64(i64 %max.09, i64 %0)
  %cmp1 = icmp slt i64 %max.09, %0
  %spec.select7 = select i1 %cmp1, i64 123, i64 %idx.011
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %n
  br i1 %exitcond.not, label %exit, label %for.body

exit:
  store i64 %1, ptr %res_max
  ret i64 %spec.select7
}

declare i64 @llvm.smax.i64(i64, i64)
