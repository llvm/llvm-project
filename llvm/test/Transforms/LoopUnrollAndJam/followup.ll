; RUN: opt -basic-aa -tbaa -loop-unroll-and-jam -allow-unroll-and-jam -unroll-and-jam-count=4 -unroll-remainder < %s -S | FileCheck %s
;
; Check that followup attributes are set in the new loops.
;
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"

define void @followup(i32 %I, i32 %J, i32* noalias nocapture %A, i32* noalias nocapture readonly %B) {
entry:
  %cmp = icmp ne i32 %J, 0
  %cmp122 = icmp ne i32 %I, 0
  %or.cond = and i1 %cmp, %cmp122
  br i1 %or.cond, label %for.outer.preheader, label %for.end

for.outer.preheader:
  br label %for.outer

for.outer:
  %i.us = phi i32 [ %add8.us, %for.latch ], [ 0, %for.outer.preheader ]
  br label %for.inner

for.inner:
  %j.us = phi i32 [ 0, %for.outer ], [ %inc.us, %for.inner ]
  %sum1.us = phi i32 [ 0, %for.outer ], [ %add.us, %for.inner ]
  %arrayidx.us = getelementptr inbounds i32, i32* %B, i32 %j.us
  %0 = load i32, i32* %arrayidx.us, align 4
  %add.us = add i32 %0, %sum1.us
  %inc.us = add nuw i32 %j.us, 1
  %exitcond = icmp eq i32 %inc.us, %J
  br i1 %exitcond, label %for.latch, label %for.inner

for.latch:
  %add.us.lcssa = phi i32 [ %add.us, %for.inner ]
  %arrayidx6.us = getelementptr inbounds i32, i32* %A, i32 %i.us
  store i32 %add.us.lcssa, i32* %arrayidx6.us, align 4
  %add8.us = add nuw i32 %i.us, 1
  %exitcond25 = icmp eq i32 %add8.us, %I
  br i1 %exitcond25, label %for.end.loopexit, label %for.outer, !llvm.loop !0

for.end.loopexit:
  br label %for.end

for.end:
  ret void
}

!0 = !{!0, !1, !2, !3, !4, !6}
!1 = !{!"llvm.loop.unroll_and_jam.enable"}
!2 = !{!"llvm.loop.unroll_and_jam.followup_outer", !{!"FollowupOuter"}}
!3 = !{!"llvm.loop.unroll_and_jam.followup_inner", !{!"FollowupInner"}}
!4 = !{!"llvm.loop.unroll_and_jam.followup_all", !{!"FollowupAll"}}
!6 = !{!"llvm.loop.unroll_and_jam.followup_remainder_inner", !{!"FollowupRemainderInner"}}


; CHECK: br i1 %exitcond.3, label %for.latch, label %for.inner, !llvm.loop ![[LOOP_INNER:[0-9]+]]
; CHECK: br i1 %niter.ncmp.3, label %for.end.loopexit.unr-lcssa.loopexit, label %for.outer, !llvm.loop ![[LOOP_OUTER:[0-9]+]]
; CHECK: br i1 %exitcond.epil, label %for.latch.epil, label %for.inner.epil, !llvm.loop ![[LOOP_REMAINDER_INNER:[0-9]+]]
; CHECK: br i1 %exitcond.epil.1, label %for.latch.epil.1, label %for.inner.epil.1, !llvm.loop ![[LOOP_REMAINDER_INNER]]
; CHECK: br i1 %exitcond.epil.2, label %for.latch.epil.2, label %for.inner.epil.2, !llvm.loop ![[LOOP_REMAINDER_INNER]]

; CHECK: ![[LOOP_INNER]] = distinct !{![[LOOP_INNER]], ![[FOLLOWUP_ALL:[0-9]+]], ![[FOLLOWUP_INNER:[0-9]+]]}
; CHECK: ![[FOLLOWUP_ALL]] = !{!"FollowupAll"}
; CHECK: ![[FOLLOWUP_INNER]] = !{!"FollowupInner"}
; CHECK: ![[LOOP_OUTER]] = distinct !{![[LOOP_OUTER]], ![[FOLLOWUP_ALL]], ![[FOLLOWUP_OUTER:[0-9]+]]}
; CHECK: ![[FOLLOWUP_OUTER]] = !{!"FollowupOuter"}
; CHECK: ![[LOOP_REMAINDER_INNER]] = distinct !{![[LOOP_REMAINDER_INNER]], ![[FOLLOWUP_ALL]], ![[FOLLOWUP_REMAINDER_INNER:[0-9]+]]}
; CHECK: ![[FOLLOWUP_REMAINDER_INNER]] = !{!"FollowupRemainderInner"}
