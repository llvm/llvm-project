; REQUIRES: asserts
; RUN: opt < %s -mcpu=neoverse-v2 -passes=loop-vectorize -debug-only=loop-vectorize -disable-output 2>&1 | FileCheck %s

target triple="aarch64--linux-gnu"

; This test shows that comparison and next iteration IV have zero cost if the
; vector loop gets executed exactly once with the given VF.
define i64 @test(ptr %a, ptr %b) #0 {
; CHECK-LABEL: LV: Checking a loop in 'test'
; CHECK: Cost of 1 for VF 8: induction instruction   %i.iv.next = add nuw nsw i64 %i.iv, 1
; CHECK-NEXT: Cost of 0 for VF 8: induction instruction   %i.iv = phi i64 [ 0, %entry ], [ %i.iv.next, %for.body ]
; CHECK-NEXT: Cost of 1 for VF 8: exit condition instruction   %exitcond.not = icmp eq i64 %i.iv.next, 16
; CHECK-NEXT: Cost of 0 for VF 8: EMIT vp<%2> = CANONICAL-INDUCTION ir<0>, vp<%index.next>
; CHECK: Cost for VF 8: 26
; CHECK-NEXT: Cost of 0 for VF 16: induction instruction   %i.iv = phi i64 [ 0, %entry ], [ %i.iv.next, %for.body ]
; CHECK-NEXT: Cost of 0 for VF 16: EMIT vp<%2> = CANONICAL-INDUCTION ir<0>, vp<%index.next>
; CHECK: Cost for VF 16: 48
; CHECK: LV: Selecting VF: 16
entry:
  br label %for.body

exit:                                 ; preds = %for.body
  ret i64 %add

for.body:                                         ; preds = %entry, %for.body
  %i.iv = phi i64 [ 0, %entry ], [ %i.iv.next, %for.body ]
  %sum = phi i64 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds i8, ptr %a, i64 %i.iv
  %0 = load i8, ptr %arrayidx, align 1
  %conv = zext i8 %0 to i64
  %arrayidx2 = getelementptr inbounds i8, ptr %b, i64 %i.iv
  %1 = load i8, ptr %arrayidx2, align 1
  %conv3 = zext i8 %1 to i64
  %mul = mul nuw nsw i64 %conv3, %conv
  %add = add i64 %mul, %sum
  %i.iv.next = add nuw nsw i64 %i.iv, 1
  %exitcond.not = icmp eq i64 %i.iv.next, 16
  br i1 %exitcond.not, label %exit, label %for.body
}

; Same as above, but in the next iteration IV has extra users, and thus, the cost is not zero.
define i64 @test_external_iv_user(ptr %a, ptr %b) #0 {
; CHECK-LABEL: LV: Checking a loop in 'test_external_iv_user'
; CHECK: Cost of 1 for VF 8: induction instruction   %i.iv.next = add nuw nsw i64 %i.iv, 1
; CHECK-NEXT: Cost of 0 for VF 8: induction instruction   %i.iv = phi i64 [ 0, %entry ], [ %i.iv.next, %for.body ]
; CHECK-NEXT: Cost of 1 for VF 8: exit condition instruction   %exitcond.not = icmp eq i64 %i.iv.next, 16
; CHECK-NEXT: Cost of 0 for VF 8: EMIT vp<%2> = CANONICAL-INDUCTION ir<0>, vp<%index.next>
; CHECK: Cost for VF 8: 26
; CHECK-NEXT: Cost of 1 for VF 16: induction instruction   %i.iv.next = add nuw nsw i64 %i.iv, 1
; CHECK-NEXT: Cost of 0 for VF 16: induction instruction   %i.iv = phi i64 [ 0, %entry ], [ %i.iv.next, %for.body ]
; CHECK-NEXT: Cost of 0 for VF 16: EMIT vp<%2> = CANONICAL-INDUCTION ir<0>, vp<%index.next>
; CHECK: Cost for VF 16: 49
; CHECK: LV: Selecting VF: vscale x 2
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.iv = phi i64 [ 0, %entry ], [ %i.iv.next, %for.body ]
  %sum = phi i64 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds nuw i8, ptr %a, i64 %i.iv
  %0 = load i8, ptr %arrayidx, align 1
  %conv = zext i8 %0 to i64
  %i.iv.next = add nuw nsw i64 %i.iv, 1
  %arrayidx2 = getelementptr inbounds nuw i8, ptr %b, i64 %i.iv.next
  %1 = load i8, ptr %arrayidx2, align 1
  %conv3 = zext i8 %1 to i64
  %mul = mul nuw nsw i64 %conv3, %conv
  %add = add i64 %sum, %mul
  %exitcond.not = icmp eq i64 %i.iv.next, 16
  br i1 %exitcond.not, label %exit, label %for.body

exit:                                 ; preds = %for.body
  ret i64 %add
}

; Same as above but with two IVs without extra users. They all have zero cost when VF equals the number of iterations.
define i64 @test_two_ivs(ptr %a, ptr %b, i64 %start) #0 {
; CHECK-LABEL: LV: Checking a loop in 'test_two_ivs'
; CHECK: Cost of 1 for VF 8: induction instruction   %i.iv.next = add nuw nsw i64 %i.iv, 1
; CHECK-NEXT: Cost of 0 for VF 8: induction instruction   %i.iv = phi i64 [ 0, %entry ], [ %i.iv.next, %for.body ]
; CHECK-NEXT: Cost of 1 for VF 8: induction instruction   %j.iv.next = add nuw nsw i64 %j.iv, 1
; CHECK-NEXT: Cost of 0 for VF 8: induction instruction   %j.iv = phi i64 [ %start, %entry ], [ %j.iv.next, %for.body ]
; CHECK-NEXT: Cost of 1 for VF 8: exit condition instruction   %exitcond.not = icmp eq i64 %i.iv.next, 16
; CHECK-NEXT: Cost of 0 for VF 8: EMIT vp<{{.+}}> = CANONICAL-INDUCTION ir<0>, vp<%index.next>
; CHECK: Cost for VF 8: 27
; CHECK-NEXT: Cost of 0 for VF 16: induction instruction   %i.iv = phi i64 [ 0, %entry ], [ %i.iv.next, %for.body ]
; CHECK-NEXT: Cost of 0 for VF 16: induction instruction   %j.iv = phi i64 [ %start, %entry ], [ %j.iv.next, %for.body ]
; CHECK-NEXT: Cost of 0 for VF 16: EMIT vp<{{.+}}> = CANONICAL-INDUCTION ir<0>, vp<%index.next>
; CHECK: Cost for VF 16: 48
; CHECK: LV: Selecting VF: 16
entry:
  br label %for.body

exit:                                 ; preds = %for.body
  ret i64 %add

for.body:                                         ; preds = %entry, %for.body
  %i.iv = phi i64 [ 0, %entry ], [ %i.iv.next, %for.body ]
  %j.iv = phi i64 [ %start, %entry ], [ %j.iv.next, %for.body ]
  %sum = phi i64 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds i8, ptr %a, i64 %i.iv
  %0 = load i8, ptr %arrayidx, align 1
  %conv = zext i8 %0 to i64
  %arrayidx2 = getelementptr inbounds i8, ptr %b, i64 %j.iv
  %1 = load i8, ptr %arrayidx2, align 1
  %conv3 = zext i8 %1 to i64
  %mul = mul nuw nsw i64 %conv3, %conv
  %add = add i64 %mul, %sum
  %i.iv.next = add nuw nsw i64 %i.iv, 1
  %j.iv.next = add nuw nsw i64 %j.iv, 1
  %exitcond.not = icmp eq i64 %i.iv.next, 16
  br i1 %exitcond.not, label %exit, label %for.body
}

define i1 @test_extra_cmp_user(ptr nocapture noundef %dst, ptr nocapture noundef readonly %src) {
; CHECK-LABEL: LV: Checking a loop in 'test_extra_cmp_user'
; CHECK: Cost of 4 for VF 8: induction instruction   %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
; CHECK-NEXT: Cost of 0 for VF 8: induction instruction   %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
; CHECK-NEXT: Cost of 4 for VF 8: exit condition instruction   %exitcond.not = icmp eq i64 %indvars.iv.next, 16
; CHECK-NEXT: Cost of 0 for VF 8: EMIT vp<%3> = CANONICAL-INDUCTION ir<0>, vp<%index.next>
; CHECK: Cost for VF 8: 12
; CHECK-NEXT: Cost of 0 for VF 16: induction instruction   %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
; CHECK-NEXT: Cost of 0 for VF 16: EMIT vp<%3> = CANONICAL-INDUCTION ir<0>, vp<%index.next>
; CHECK: Cost for VF 16: 4
; CHECK: LV: Selecting VF: 16
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds nuw i8, ptr %src, i64 %indvars.iv
  %0 = load i8, ptr %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds nuw i8, ptr %dst, i64 %indvars.iv
  %1 = load i8, ptr %arrayidx2, align 4
  %add = add nsw i8 %1, %0
  store i8 %add, ptr %arrayidx2, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 16
  br i1 %exitcond.not, label %exit, label %for.body

exit:
  ret i1 %exitcond.not
}

attributes #0 = { vscale_range(1, 16) "target-features"="+sve" }
