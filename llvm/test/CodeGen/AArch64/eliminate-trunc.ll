; RUN: llc -verify-machineinstrs -o - %s -mtriple=aarch64-none-apple-ios7.0 -mcpu=cyclone | FileCheck %s

; Check  trunc i64 operation is translated as a subregister access
; eliminating an i32 induction varible.

; CHECK-NOT: add {{x[0-9]+}}, {{x[0-9]+}}, #1
; CHECK: add {{w[0-9]+}}, {{w[0-9]+}}, #1
; CHECK-NEXT: cmp {{w[0-9]+}}, {{w[0-9]+}}
define void @test1_signed([8 x i8]* nocapture %a, i8* nocapture readonly %box, i8 %limit, i64 %inv) minsize {
entry:
  %conv = zext i8 %limit to i32
  %cmp223 = icmp eq i8 %limit, 0
  br i1 %cmp223, label %for.end15, label %for.body4.lr.ph.us

for.body4.us:
  %indvars.iv = phi i64 [ 0, %for.body4.lr.ph.us ], [ %indvars.iv.next, %for.body4.us ]
  %arrayidx6.us = getelementptr inbounds [8 x i8], [8 x i8]* %a, i64 %indvars.iv, i64 %inv
  %0 = load i8, i8* %arrayidx6.us, align 1
  %idxprom7.us = zext i8 %0 to i64
  %arrayidx8.us = getelementptr inbounds i8, i8* %box, i64 %idxprom7.us
  %1 = load i8, i8* %arrayidx8.us, align 1
  store i8 %1, i8* %arrayidx6.us, align 1
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %2 = trunc i64 %indvars.iv.next to i32
  %cmp2.us = icmp slt i32 %2, %conv
  br i1 %cmp2.us, label %for.body4.us, label %for.cond1.for.inc13_crit_edge.us

for.body4.lr.ph.us:
  %indvars.iv26 = phi i64 [ %indvars.iv.next27, %for.cond1.for.inc13_crit_edge.us ], [ 0, %entry ]
  br label %for.body4.us

for.cond1.for.inc13_crit_edge.us:
  %indvars.iv.next27 = add nuw nsw i64 %indvars.iv26, 1
  %exitcond28 = icmp eq i64 %indvars.iv26, 3
  br i1 %exitcond28, label %for.end15, label %for.body4.lr.ph.us

for.end15:
  ret void
}
