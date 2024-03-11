; REQUIRES: asserts
; RUN: opt -p loop-vectorize -debug-only=loop-vectorize -S -disable-output < %s 2>&1 | FileCheck %s

target triple = "aarch64-unknown-linux-gnu"

define void @no_outer_loop(ptr nocapture noundef %a, ptr nocapture noundef readonly %b, i64 noundef %off, i64 noundef %n) {
; CHECK-LABEL: LV: Checking a loop in 'no_outer_loop'
; CHECK:      Calculating cost of runtime checks:
; CHECK-NOT:  We expect runtime memory checks to be hoisted out of the outer loop.
; CHECK:      Total cost of runtime checks: 4
; CHECK-NEXT: LV: Minimum required TC for runtime checks to be profitable:16
entry:
  br label %inner.loop

inner.loop:
  %inner.iv = phi i64 [ 0, %entry ], [ %inner.iv.next, %inner.loop ]
  %add.us = add nuw nsw i64 %inner.iv, %off
  %arrayidx.us = getelementptr inbounds i8, ptr %b, i64 %add.us
  %0 = load i8, ptr %arrayidx.us, align 1
  %arrayidx7.us = getelementptr inbounds i8, ptr %a, i64 %add.us
  %1 = load i8, ptr %arrayidx7.us, align 1
  %add9.us = add i8 %1, %0
  store i8 %add9.us, ptr %arrayidx7.us, align 1
  %inner.iv.next = add nuw nsw i64 %inner.iv, 1
  %exitcond.not = icmp eq i64 %inner.iv.next, %n
  br i1 %exitcond.not, label %inner.exit, label %inner.loop

inner.exit:
  ret void
}

define void @outer_no_tc(ptr nocapture noundef %a, ptr nocapture noundef readonly %b, i64 noundef %m, i64 noundef %n) {
; CHECK-LABEL: LV: Checking a loop in 'outer_no_tc'
; CHECK:      Calculating cost of runtime checks:
; CHECK:      We expect runtime memory checks to be hoisted out of the outer loop. Cost reduced from 6 to 3
; CHECK:      Total cost of runtime checks: 3
; CHECK-NEXT: LV: Minimum required TC for runtime checks to be profitable:16
entry:
  br label %outer.loop

outer.loop:
  %outer.iv = phi i64 [ %outer.iv.next, %inner.exit ], [ 0, %entry ]
  %mul.us = mul nsw i64 %outer.iv, %n
  br label %inner.loop

inner.loop:
  %inner.iv = phi i64 [ 0, %outer.loop ], [ %inner.iv.next, %inner.loop ]
  %add.us = add nuw nsw i64 %inner.iv, %mul.us
  %arrayidx.us = getelementptr inbounds i8, ptr %b, i64 %add.us
  %0 = load i8, ptr %arrayidx.us, align 1
  %arrayidx7.us = getelementptr inbounds i8, ptr %a, i64 %add.us
  %1 = load i8, ptr %arrayidx7.us, align 1
  %add9.us = add i8 %1, %0
  store i8 %add9.us, ptr %arrayidx7.us, align 1
  %inner.iv.next = add nuw nsw i64 %inner.iv, 1
  %exitcond.not = icmp eq i64 %inner.iv.next, %n
  br i1 %exitcond.not, label %inner.exit, label %inner.loop

inner.exit:
  %outer.iv.next = add nuw nsw i64 %outer.iv, 1
  %exitcond27.not = icmp eq i64 %outer.iv.next, %m
  br i1 %exitcond27.not, label %outer.exit, label %outer.loop

outer.exit:
  ret void
}


define void @outer_known_tc3(ptr nocapture noundef %a, ptr nocapture noundef readonly %b, i64 noundef %n) {
; CHECK-LABEL: LV: Checking a loop in 'outer_known_tc3'
; CHECK:      Calculating cost of runtime checks:
; CHECK:      We expect runtime memory checks to be hoisted out of the outer loop. Cost reduced from 6 to 2
; CHECK:      Total cost of runtime checks: 2
; CHECK-NEXT: LV: Minimum required TC for runtime checks to be profitable:16
entry:
  br label %outer.loop

outer.loop:
  %outer.iv = phi i64 [ %outer.iv.next, %inner.exit ], [ 0, %entry ]
  %mul.us = mul nsw i64 %outer.iv, %n
  br label %inner.loop

inner.loop:
  %inner.iv = phi i64 [ 0, %outer.loop ], [ %inner.iv.next, %inner.loop ]
  %add.us = add nuw nsw i64 %inner.iv, %mul.us
  %arrayidx.us = getelementptr inbounds i8, ptr %b, i64 %add.us
  %0 = load i8, ptr %arrayidx.us, align 1
  %arrayidx7.us = getelementptr inbounds i8, ptr %a, i64 %add.us
  %1 = load i8, ptr %arrayidx7.us, align 1
  %add9.us = add i8 %1, %0
  store i8 %add9.us, ptr %arrayidx7.us, align 1
  %inner.iv.next = add nuw nsw i64 %inner.iv, 1
  %exitcond.not = icmp eq i64 %inner.iv.next, %n
  br i1 %exitcond.not, label %inner.exit, label %inner.loop

inner.exit:
  %outer.iv.next = add nuw nsw i64 %outer.iv, 1
  %exitcond26.not = icmp eq i64 %outer.iv.next, 3
  br i1 %exitcond26.not, label %outer.exit, label %outer.loop

outer.exit:
  ret void
}


define void @outer_known_tc64(ptr nocapture noundef %a, ptr nocapture noundef readonly %b, i64 noundef %n) {
; CHECK-LABEL: LV: Checking a loop in 'outer_known_tc64'
; CHECK:      Calculating cost of runtime checks:
; CHECK:      We expect runtime memory checks to be hoisted out of the outer loop. Cost reduced from 6 to 1
; CHECK:      Total cost of runtime checks: 1
; CHECK-NEXT: LV: Minimum required TC for runtime checks to be profitable:16
entry:
  br label %outer.loop

outer.loop:
  %outer.iv = phi i64 [ %outer.iv.next, %inner.exit ], [ 0, %entry ]
  %mul.us = mul nsw i64 %outer.iv, %n
  br label %inner.loop

inner.loop:
  %inner.iv = phi i64 [ 0, %outer.loop ], [ %inner.iv.next, %inner.loop ]
  %add.us = add nuw nsw i64 %inner.iv, %mul.us
  %arrayidx.us = getelementptr inbounds i8, ptr %b, i64 %add.us
  %0 = load i8, ptr %arrayidx.us, align 1
  %arrayidx7.us = getelementptr inbounds i8, ptr %a, i64 %add.us
  %1 = load i8, ptr %arrayidx7.us, align 1
  %add9.us = add i8 %1, %0
  store i8 %add9.us, ptr %arrayidx7.us, align 1
  %inner.iv.next = add nuw nsw i64 %inner.iv, 1
  %exitcond.not = icmp eq i64 %inner.iv.next, %n
  br i1 %exitcond.not, label %inner.exit, label %inner.loop

inner.exit:
  %outer.iv.next = add nuw nsw i64 %outer.iv, 1
  %exitcond26.not = icmp eq i64 %outer.iv.next, 64
  br i1 %exitcond26.not, label %outer.exit, label %outer.loop

outer.exit:
  ret void
}


define void @outer_pgo_3(ptr nocapture noundef %a, ptr nocapture noundef readonly %b, i64 noundef %m, i64 noundef %n) {
; CHECK-LABEL: LV: Checking a loop in 'outer_pgo_3'
; CHECK:      Calculating cost of runtime checks:
; CHECK:      We expect runtime memory checks to be hoisted out of the outer loop. Cost reduced from 6 to 2
; CHECK:      Total cost of runtime checks: 2
; CHECK-NEXT: LV: Minimum required TC for runtime checks to be profitable:16
entry:
  br label %outer.loop

outer.loop:
  %outer.iv = phi i64 [ %outer.iv.next, %inner.exit ], [ 0, %entry ]
  %mul.us = mul nsw i64 %outer.iv, %n
  br label %inner.loop

inner.loop:
  %inner.iv = phi i64 [ 0, %outer.loop ], [ %inner.iv.next, %inner.loop ]
  %add.us = add nuw nsw i64 %inner.iv, %mul.us
  %arrayidx.us = getelementptr inbounds i8, ptr %b, i64 %add.us
  %0 = load i8, ptr %arrayidx.us, align 1
  %arrayidx7.us = getelementptr inbounds i8, ptr %a, i64 %add.us
  %1 = load i8, ptr %arrayidx7.us, align 1
  %add9.us = add i8 %1, %0
  store i8 %add9.us, ptr %arrayidx7.us, align 1
  %inner.iv.next = add nuw nsw i64 %inner.iv, 1
  %exitcond.not = icmp eq i64 %inner.iv.next, %n
  br i1 %exitcond.not, label %inner.exit, label %inner.loop

inner.exit:
  %outer.iv.next = add nuw nsw i64 %outer.iv, 1
  %exitcond26.not = icmp eq i64 %outer.iv.next, %m
  br i1 %exitcond26.not, label %outer.exit, label %outer.loop, !prof !0

outer.exit:
  ret void
}


define void @outer_pgo_minus1(ptr nocapture noundef %a, ptr nocapture noundef readonly %b, i64 noundef %m, i64 noundef %n) {
; CHECK-LABEL: LV: Checking a loop in 'outer_pgo_minus1'
; CHECK:      Calculating cost of runtime checks:
; CHECK-NOT:  We expect runtime memory checks to be hoisted out of the outer loop. Cost reduced
; CHECK:      Total cost of runtime checks: 6
; CHECK-NEXT: LV: Minimum required TC for runtime checks to be profitable:16
entry:
  br label %outer.loop

outer.loop:
  %outer.iv = phi i64 [ %outer.iv.next, %inner.exit ], [ 0, %entry ]
  %mul.us = mul nsw i64 %outer.iv, %n
  br label %inner.loop

inner.loop:
  %inner.iv = phi i64 [ 0, %outer.loop ], [ %inner.iv.next, %inner.loop ]
  %add.us = add nuw nsw i64 %inner.iv, %mul.us
  %arrayidx.us = getelementptr inbounds i8, ptr %b, i64 %add.us
  %0 = load i8, ptr %arrayidx.us, align 1
  %arrayidx7.us = getelementptr inbounds i8, ptr %a, i64 %add.us
  %1 = load i8, ptr %arrayidx7.us, align 1
  %add9.us = add i8 %1, %0
  store i8 %add9.us, ptr %arrayidx7.us, align 1
  %inner.iv.next = add nuw nsw i64 %inner.iv, 1
  %exitcond.not = icmp eq i64 %inner.iv.next, %n
  br i1 %exitcond.not, label %inner.exit, label %inner.loop

inner.exit:
  %outer.iv.next = add nuw nsw i64 %outer.iv, 1
  %exitcond26.not = icmp eq i64 %outer.iv.next, %m
  br i1 %exitcond26.not, label %outer.exit, label %outer.loop, !prof !1

outer.exit:
  ret void
}


define void @outer_known_tc3_full_range_checks(ptr nocapture noundef %dst, ptr nocapture noundef readonly %src, i64 noundef %n) {
; CHECK-LABEL: LV: Checking a loop in 'outer_known_tc3_full_range_checks'
; CHECK:      Calculating cost of runtime checks:
; CHECK:      We expect runtime memory checks to be hoisted out of the outer loop. Cost reduced from 6 to 2
; CHECK:      Total cost of runtime checks: 2
; CHECK-NEXT: LV: Minimum required TC for runtime checks to be profitable:4
entry:
  br label %outer.loop

outer.loop:
  %outer.iv = phi i64 [ 0, %entry ], [ %outer.iv.next, %inner.exit ]
  %0 = mul nsw i64 %outer.iv, %n
  br label %inner.loop

inner.loop:
  %iv.inner = phi i64 [ 0, %outer.loop ], [ %iv.inner.next, %inner.loop ]
  %1 = add nuw nsw i64 %iv.inner, %0
  %arrayidx.us = getelementptr inbounds i32, ptr %src, i64 %1
  %2 = load i32, ptr %arrayidx.us, align 4
  %arrayidx8.us = getelementptr inbounds i32, ptr %dst, i64 %1
  %3 = load i32, ptr %arrayidx8.us, align 4
  %add9.us = add nsw i32 %3, %2
  store i32 %add9.us, ptr %arrayidx8.us, align 4
  %iv.inner.next = add nuw nsw i64 %iv.inner, 1
  %inner.exit.cond = icmp eq i64 %iv.inner.next, %n
  br i1 %inner.exit.cond, label %inner.exit, label %inner.loop

inner.exit:
  %outer.iv.next = add nuw nsw i64 %outer.iv, 1
  %outer.exit.cond = icmp eq i64 %outer.iv.next, 3
  br i1 %outer.exit.cond, label %outer.exit, label %outer.loop

outer.exit:
  ret void
}


!0 = !{!"branch_weights", i32 10, i32 20}
!1 = !{!"branch_weights", i32 1, i32 -1}
