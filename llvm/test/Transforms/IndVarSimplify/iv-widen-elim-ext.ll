; RUN: opt < %s -indvars -S | FileCheck %s --implicit-check-not sext --implicit-check-not zext

target datalayout = "p:64:64:64-n32:64"

; When widening IV and its users, trunc and zext/sext are not needed
; if the original 32-bit user is known to be non-negative, whether
; the IV is considered signed or unsigned.
define void @foo(i32* %A, i32* %B, i32* %C, i32 %N) {
; CHECK-LABEL: @foo(
; CHECK: wide.trip.count = zext
; CHECK: ret void
entry:
  %cmp1 = icmp slt i32 0, %N
  br i1 %cmp1, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %for.inc
  %i.02 = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %for.inc ]
  %idxprom = sext i32 %i.02 to i64
  %arrayidx = getelementptr inbounds i32, i32* %B, i64 %idxprom
  %0 = load i32, i32* %arrayidx, align 4
  %add = add nsw i32 %i.02, 2
  %idxprom1 = zext i32 %add to i64
  %arrayidx2 = getelementptr inbounds i32, i32* %C, i64 %idxprom1
  %1 = load i32, i32* %arrayidx2, align 4
  %add3 = add nsw i32 %0, %1
  %idxprom4 = zext i32 %i.02 to i64
  %arrayidx5 = getelementptr inbounds i32, i32* %A, i64 %idxprom4
  store i32 %add3, i32* %arrayidx5, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %inc = add nsw i32 %i.02, 1
  %cmp = icmp slt i32 %inc, %N
  br i1 %cmp, label %for.body, label %for.cond.for.end_crit_edge

for.cond.for.end_crit_edge:                       ; preds = %for.inc
  br label %for.end

for.end:                                          ; preds = %for.cond.for.end_crit_edge, %entry
  ret void
}

define void @foo1(i32* %A, i32* %B, i32* %C, i32 %N) {
; CHECK-LABEL: @foo1(
; CHECK: wide.trip.count = zext
; CHECK: ret void
entry:
  %cmp1 = icmp slt i32 0, %N
  br i1 %cmp1, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %for.inc
  %i.02 = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %for.inc ]
  %idxprom = zext i32 %i.02 to i64
  %arrayidx = getelementptr inbounds i32, i32* %B, i64 %idxprom
  %0 = load i32, i32* %arrayidx, align 4
  %add = add nsw i32 %i.02, 2
  %idxprom1 = sext i32 %add to i64
  %arrayidx2 = getelementptr inbounds i32, i32* %C, i64 %idxprom1
  %1 = load i32, i32* %arrayidx2, align 4
  %add3 = add nsw i32 %0, %1
  %idxprom4 = sext i32 %i.02 to i64
  %arrayidx5 = getelementptr inbounds i32, i32* %A, i64 %idxprom4
  store i32 %add3, i32* %arrayidx5, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %inc = add nsw i32 %i.02, 1
  %cmp = icmp slt i32 %inc, %N
  br i1 %cmp, label %for.body, label %for.cond.for.end_crit_edge

for.cond.for.end_crit_edge:                       ; preds = %for.inc
  br label %for.end

for.end:                                          ; preds = %for.cond.for.end_crit_edge, %entry
  ret void
}
