; RUN: opt -loop-vectorize -force-vector-width=4 -force-vector-interleave=1 -S < %s | FileCheck %s
;
; Check that the followup loop attributes are applied.
;
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

define void @followup(i32* nocapture %a, i32 %n) {
entry:
  %cmp4 = icmp sgt i32 %n, 0
  br i1 %cmp4, label %for.body, label %for.end

for.body:
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %indvars.iv
  %0 = trunc i64 %indvars.iv to i32
  store i32 %0, i32* %arrayidx, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !0

for.end:
  ret void
}

!0 = distinct !{!0, !3, !4, !5}
!3 = !{!"llvm.loop.vectorize.followup_vectorized", !{!"FollowupVectorized"}}
!4 = !{!"llvm.loop.vectorize.followup_epilogue", !{!"FollowupEpilogue"}}
!5 = !{!"llvm.loop.vectorize.followup_all", !{!"FollowupAll"}}


; CHECK-LABEL @followup(

; CHECK-LABEL: vector.body:
; CHECK: br i1 %13, label %middle.block, label %vector.body, !llvm.loop ![[LOOP_VECTOR:[0-9]+]]
; CHECK-LABEL: for.body:
; CHECK: br i1 %exitcond, label %for.end.loopexit, label %for.body, !llvm.loop ![[LOOP_EPILOGUE:[0-9]+]]

; CHECK: ![[LOOP_VECTOR]] = distinct !{![[LOOP_VECTOR]], ![[FOLLOWUP_ALL:[0-9]+]], ![[FOLLOWUP_VECTORIZED:[0-9]+]]}
; CHECK: ![[FOLLOWUP_ALL]] = !{!"FollowupAll"}
; CHECK: ![[FOLLOWUP_VECTORIZED:[0-9]+]] = !{!"FollowupVectorized"}
; CHECK: ![[LOOP_EPILOGUE]] = distinct !{![[LOOP_EPILOGUE]], ![[FOLLOWUP_ALL]], ![[FOLLOWUP_EPILOGUE:[0-9]+]]}
; CHECK: ![[FOLLOWUP_EPILOGUE]] = !{!"FollowupEpilogue"}
