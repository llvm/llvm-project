; RUN: opt < %s -S -passes='module(cgscc(function(loop(loop-flatten)))),function(loop-vectorize)' --force-vector-width=4 --force-vector-interleave=3 | FileCheck %s
; RUN: opt < %s -S -passes='loop(loop-flatten),verify' -loop-flatten-version-loops=false \
; RUN:     -loop-flatten-cost-threshold=6 -verify-loop-info -verify-dom-info \
; RUN:     -verify-scev | FileCheck %s --check-prefix=NOVERSION

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Make sure loop-flatten does not create an unversioned widened loop where the
; replacement narrow IV may wrap.
define void @zext_i8(i8 %N, ptr %A) {
; CHECK-LABEL: @zext_i8(
; CHECK:       for.cond1.preheader.us.lver.check:
; CHECK-NEXT:    %flatten.mul = call { i8, i1 } @llvm.umul.with.overflow.i8(i8 %N, i8 %N)
; CHECK-NEXT:    %flatten.tripcount = extractvalue { i8, i1 } %flatten.mul, 0
; CHECK-NEXT:    %flatten.overflow = extractvalue { i8, i1 } %flatten.mul, 1
; CHECK-NEXT:    br i1 %flatten.overflow, label %for.cond1.preheader.us.ph.lver.orig, label %for.cond1.preheader.us.ph
; CHECK-NOT:     flatten.trunciv
; NOVERSION-LABEL: @zext_i8(
; NOVERSION-NOT:   flatten.tripcount
; NOVERSION-NOT:   flatten.trunciv
entry:
  %cmp20.not = icmp eq i8 %N, 0
  br i1 %cmp20.not, label %common.ret, label %for.cond1.preheader.us

for.cond1.preheader.us:
  %i.021.us = phi i8 [ %inc8.us, %for.cond1.for.inc7_crit_edge.us ], [ 0, %entry ]
  br label %for.body3.us

for.body3.us:
  %j.019.us = phi i8 [ 0, %for.cond1.preheader.us ], [ %inc.us, %for.body3.us ]
  %mul.us = mul i8 %i.021.us, %N
  %add.us = add i8 %j.019.us, %mul.us
  %idxprom.us = zext i8 %add.us to i64
  %arrayidx.us = getelementptr [2 x i8], ptr %A, i64 %idxprom.us
  store i16 1, ptr %arrayidx.us, align 2
  %inc.us = add i8 %j.019.us, 1
  %cmp2.us = icmp ult i8 %inc.us, %N
  br i1 %cmp2.us, label %for.body3.us, label %for.cond1.for.inc7_crit_edge.us

for.cond1.for.inc7_crit_edge.us:
  %inc8.us = add i8 %i.021.us, 1
  %cmp.us = icmp ult i8 %inc8.us, %N
  br i1 %cmp.us, label %for.cond1.preheader.us, label %common.ret

common.ret:
  ret void
}

; The narrow linear IV replacement is still valid if the original computation
; would be poison on overflow.
define void @nsw_i32(ptr %A, i32 %N, i32 %M) {
; CHECK-LABEL: @nsw_i32(
; NOVERSION-LABEL: @nsw_i32(
; NOVERSION-NOT:   lver.check
; NOVERSION:       %flatten.tripcount = mul i64
entry:
  %cmp17 = icmp sgt i32 %N, 0
  br i1 %cmp17, label %for.cond1.preheader.lr.ph, label %for.cond.cleanup

for.cond1.preheader.lr.ph:
  %cmp215 = icmp sgt i32 %M, 0
  br i1 %cmp215, label %for.cond1.preheader.us.preheader, label %for.cond.cleanup

for.cond1.preheader.us.preheader:
  br label %for.cond1.preheader.us

for.cond1.preheader.us:
  %i.018.us = phi i32 [ %inc6.us, %for.cond1.for.cond.cleanup3_crit_edge.us ], [ 0, %for.cond1.preheader.us.preheader ]
  %mul.us = mul nsw i32 %i.018.us, %M
  br label %for.body4.us

for.body4.us:
  %j.016.us = phi i32 [ 0, %for.cond1.preheader.us ], [ %inc.us, %for.body4.us ]
  %add.us = add nsw i32 %j.016.us, %mul.us
  %idxprom.us = sext i32 %add.us to i64
  %arrayidx.us = getelementptr inbounds i32, ptr %A, i64 %idxprom.us
  tail call void @f(ptr %arrayidx.us)
  %inc.us = add nuw nsw i32 %j.016.us, 1
  %cmp2.us = icmp slt i32 %inc.us, %M
  br i1 %cmp2.us, label %for.body4.us, label %for.cond1.for.cond.cleanup3_crit_edge.us

for.cond1.for.cond.cleanup3_crit_edge.us:
  %inc6.us = add nuw nsw i32 %i.018.us, 1
  %cmp.us = icmp slt i32 %inc6.us, %N
  br i1 %cmp.us, label %for.cond1.preheader.us, label %for.cond.cleanup.loopexit

for.cond.cleanup.loopexit:
  br label %for.cond.cleanup

for.cond.cleanup:
  ret void
}

declare void @f(ptr)
