; RUN: opt -passes=loop-vectorize -force-vector-interleave=1 -force-vector-width=4 -S < %s | FileCheck %s --check-prefix=CHECK

define i64 @select_icmp_nuw_nsw(ptr %a, ptr %b, i64 %ii, i64 %n) {
; CHECK-LABEL: define i64 @select_icmp_nuw_nsw
; CHECK-NOT:   vector.body:
;
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %iv = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %rdx = phi i64 [ %cond, %for.body ], [ %ii, %entry ]
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

define i64 @select_icmp_nsw(ptr %a, ptr %b, i64 %ii, i64 %n) {
; CHECK-LABEL: define i64 @select_icmp_nsw
; CHECK-NOT:   vector.body:
;
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %iv = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %rdx = phi i64 [ %cond, %for.body ], [ %ii, %entry ]
  %arrayidx = getelementptr inbounds i64, ptr %a, i64 %iv
  %0 = load i64, ptr %arrayidx, align 8
  %arrayidx1 = getelementptr inbounds i64, ptr %b, i64 %iv
  %1 = load i64, ptr %arrayidx1, align 8
  %cmp2 = icmp sgt i64 %0, %1
  %cond = select i1 %cmp2, i64 %iv, i64 %rdx
  %inc = add nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %inc, %n
  br i1 %exitcond.not, label %exit, label %for.body

exit:                                             ; preds = %for.body
  ret i64 %cond
}

define i64 @select_icmp_nuw(ptr %a, ptr %b, i64 %ii, i64 %n) {
; CHECK-LABEL: define i64 @select_icmp_nuw
; CHECK-NOT:   vector.body:
;
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %iv = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %rdx = phi i64 [ %cond, %for.body ], [ %ii, %entry ]
  %arrayidx = getelementptr inbounds i64, ptr %a, i64 %iv
  %0 = load i64, ptr %arrayidx, align 8
  %arrayidx1 = getelementptr inbounds i64, ptr %b, i64 %iv
  %1 = load i64, ptr %arrayidx1, align 8
  %cmp2 = icmp sgt i64 %0, %1
  %cond = select i1 %cmp2, i64 %iv, i64 %rdx
  %inc = add nuw i64 %iv, 1
  %exitcond.not = icmp eq i64 %inc, %n
  br i1 %exitcond.not, label %exit, label %for.body

exit:                                             ; preds = %for.body
  ret i64 %cond
}

define i64 @select_icmp_noflag(ptr %a, ptr %b, i64 %ii, i64 %n) {
; CHECK-LABEL: define i64 @select_icmp_noflag
; CHECK-NOT:   vector.body:
;
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %iv = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %rdx = phi i64 [ %cond, %for.body ], [ %ii, %entry ]
  %arrayidx = getelementptr inbounds i64, ptr %a, i64 %iv
  %0 = load i64, ptr %arrayidx, align 8
  %arrayidx1 = getelementptr inbounds i64, ptr %b, i64 %iv
  %1 = load i64, ptr %arrayidx1, align 8
  %cmp2 = icmp sgt i64 %0, %1
  %cond = select i1 %cmp2, i64 %iv, i64 %rdx
  %inc = add i64 %iv, 1
  %exitcond.not = icmp eq i64 %inc, %n
  br i1 %exitcond.not, label %exit, label %for.body

exit:                                             ; preds = %for.body
  ret i64 %cond
}
