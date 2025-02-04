; RUN: opt %loadNPMPolly -passes=polly-codegen -S < %s | FileCheck %s
;
; llvm.org/PR48422
; Use of ScalarEvolution in Codegen not possible because DominatorTree is not updated.
;
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

define dso_local void @func(i1 %b, i1 %p3, ptr %d) local_unnamed_addr {
entry:
  %conv = zext i1 %b to i16
  %add = select i1 %p3, i32 21, i32 20
  br label %for.body.us.us

for.body.us.us:
  %e.062.us.us = phi i16 [ %inc.us.us, %omp.inner.for.cond.simd.if.end.loopexit_crit_edge.us.us ], [ %conv, %entry ]
  %idxprom.us.us = sext i16 %e.062.us.us to i64
  br i1 %b, label %omp.inner.for.body.us.us.us.preheader, label %omp.inner.for.body.us63.us.preheader

omp.inner.for.body.us63.us.preheader:
  %arrayidx25.us.le71.us = getelementptr inbounds [14 x i32], ptr %d, i64 %idxprom.us.us, i64 0
  %0 = load i32, ptr %arrayidx25.us.le71.us, align 4
  br label %omp.inner.for.cond.simd.if.end.loopexit_crit_edge.us.us

omp.inner.for.body.us.us.us.preheader:
  %arrayidx25.us.le.us.us = getelementptr inbounds [14 x i32], ptr %d, i64 %idxprom.us.us, i64 0
  %1 = load i32, ptr %arrayidx25.us.le.us.us, align 4
  %conv27.us.le.us.us = select i1 undef, i16 0, i16 undef
  br label %omp.inner.for.cond.simd.if.end.loopexit_crit_edge.us.us

omp.inner.for.cond.simd.if.end.loopexit_crit_edge.us.us:
  %conv27.lcssa.us.us = phi i16 [ undef, %omp.inner.for.body.us63.us.preheader ], [ %conv27.us.le.us.us, %omp.inner.for.body.us.us.us.preheader ]
  %inc.us.us = add i16 %e.062.us.us, 1
  %conv2.us.us = sext i16 %inc.us.us to i32
  %cmp.us.us = icmp sgt i32 %add, %conv2.us.us
  br i1 %cmp.us.us, label %for.body.us.us, label %for.cond.cleanup.loopexit

for.cond.cleanup.loopexit:
  ret void
}


; CHECK-LABEL: @func(
; CHECK:         polly.stmt.omp.inner.for.body.us.us.us.preheader:
; CHECK:         load i32, ptr %scevgep, align 4, !alias.scope !2, !noalias !5

; CHECK:       !0 = distinct !{!0, !1}
; CHECK:       !1 = !{!"llvm.loop.vectorize.enable", i32 0}
; CHECK:       !2 = !{!3}
; CHECK:       !3 = distinct !{!3, !4, !"polly.alias.scope.MemRef_d"}
; CHECK:       !4 = distinct !{!4, !"polly.alias.scope.domain"}
; CHECK:       !5 = !{}
