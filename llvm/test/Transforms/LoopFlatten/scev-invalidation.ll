; RUN: opt < %s -S -passes='module(cgscc(function(loop(loop-flatten)))),function(loop-vectorize)' --force-vector-width=4 --force-vector-interleave=3 | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Make sure loop-flatten does not preserve stale ScalarEvolution results that
; make loop-vectorize omit the runtime SCEV check for the scalarized zext index.
define void @zext_i8(i8 %N, ptr %A) {
; CHECK-LABEL: @zext_i8(
; CHECK:       for.cond1.preheader.us.preheader:
; CHECK:         %flatten.tripcount = mul i64
; CHECK:         br i1 %min.iters.check, label %scalar.ph, label %vector.scevcheck
; CHECK:       vector.scevcheck:
; CHECK-NEXT:    {{%.*}} = add nsw i64 %flatten.tripcount, -1
; CHECK-NEXT:    {{%.*}} = icmp ugt i64 {{%.*}}, 255
; CHECK-NEXT:    br i1 {{%.*}}, label %scalar.ph, label %vector.ph
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
