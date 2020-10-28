; RUN: opt < %s  -O1  -S -loop-versioning-licm -licm -debug-only=loop-versioning-licm -disable-loop-unrolling 2>&1 | FileCheck %s
; RUN: opt < %s  -S -passes='loop-versioning-licm,licm' --aa-pipeline=default -debug-only=loop-versioning-licm -disable-loop-unrolling 2>&1 | FileCheck %s
; REQUIRES: asserts
;
; Test to confirm loop is a good candidate for LoopVersioningLICM
; It also confirms invariant moved out of loop.
;
; CHECK: Loop: Loop at depth 2 containing: %for.body3.us<header><latch><exiting>
; CHECK-NEXT:     Loop Versioning found to be beneficial
;
; CHECK: for.cond1.for.inc17_crit_edge.us.loopexit6:       ; preds = %for.body3.us
; CHECK-NEXT: %add14.us.lcssa = phi float [ %add14.us, %for.body3.us ]
; CHECK-NEXT: store float %add14.us.lcssa, float* %arrayidx.us, align 4, !alias.scope !0, !noalias !0
; CHECK-NEXT: br label %for.cond1.for.inc17_crit_edge.us
;
define i32 @foo(float* nocapture %var2, float** nocapture readonly %var3, i32 %itr) #0 {
entry:
  %cmp38 = icmp sgt i32 %itr, 1
  br i1 %cmp38, label %for.body3.lr.ph.us, label %for.end19

for.body3.us:                                     ; preds = %for.body3.us, %for.body3.lr.ph.us
  %0 = phi float [ %.pre, %for.body3.lr.ph.us ], [ %add14.us, %for.body3.us ]
  %indvars.iv = phi i64 [ 1, %for.body3.lr.ph.us ], [ %indvars.iv.next, %for.body3.us ]
  %1 = trunc i64 %indvars.iv to i32
  %conv.us = sitofp i32 %1 to float
  %add.us = fadd float %conv.us, %0
  %arrayidx7.us = getelementptr inbounds float, float* %3, i64 %indvars.iv
  store float %add.us, float* %arrayidx7.us, align 4
  %2 = load float, float* %arrayidx.us, align 4
  %add14.us = fadd float %2, %add.us
  store float %add14.us, float* %arrayidx.us, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %itr
  br i1 %exitcond, label %for.cond1.for.inc17_crit_edge.us, label %for.body3.us

for.body3.lr.ph.us:                               ; preds = %entry, %for.cond1.for.inc17_crit_edge.us
  %indvars.iv40 = phi i64 [ %indvars.iv.next41, %for.cond1.for.inc17_crit_edge.us ], [ 1, %entry ]
  %arrayidx.us = getelementptr inbounds float, float* %var2, i64 %indvars.iv40
  %arrayidx6.us = getelementptr inbounds float*, float** %var3, i64 %indvars.iv40
  %3 = load float*, float** %arrayidx6.us, align 8
  %.pre = load float, float* %arrayidx.us, align 4
  br label %for.body3.us

for.cond1.for.inc17_crit_edge.us:                 ; preds = %for.body3.us
  %indvars.iv.next41 = add nuw nsw i64 %indvars.iv40, 1
  %lftr.wideiv42 = trunc i64 %indvars.iv.next41 to i32
  %exitcond43 = icmp eq i32 %lftr.wideiv42, %itr
  br i1 %exitcond43, label %for.end19, label %for.body3.lr.ph.us

for.end19:                                        ; preds = %for.cond1.for.inc17_crit_edge.us, %entry
  ret i32 0
}
