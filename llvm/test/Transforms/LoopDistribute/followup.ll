; RUN: opt -basic-aa -loop-distribute -S < %s | FileCheck %s
;
; Check that followup loop-attributes are applied to the loops after
; loop distribution.
;
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

define void @f(i32* %a, i32* %b, i32* %c, i32* %d, i32* %e) {
entry:
  br label %for.body

for.body:
  %ind = phi i64 [ 0, %entry ], [ %add, %for.body ]

  %arrayidxA = getelementptr inbounds i32, i32* %a, i64 %ind
  %loadA = load i32, i32* %arrayidxA, align 4

  %arrayidxB = getelementptr inbounds i32, i32* %b, i64 %ind
  %loadB = load i32, i32* %arrayidxB, align 4

  %mulA = mul i32 %loadB, %loadA

  %add = add nuw nsw i64 %ind, 1
  %arrayidxA_plus_4 = getelementptr inbounds i32, i32* %a, i64 %add
  store i32 %mulA, i32* %arrayidxA_plus_4, align 4

  %arrayidxD = getelementptr inbounds i32, i32* %d, i64 %ind
  %loadD = load i32, i32* %arrayidxD, align 4

  %arrayidxE = getelementptr inbounds i32, i32* %e, i64 %ind
  %loadE = load i32, i32* %arrayidxE, align 4

  %mulC = mul i32 %loadD, %loadE

  %arrayidxC = getelementptr inbounds i32, i32* %c, i64 %ind
  store i32 %mulC, i32* %arrayidxC, align 4

  %exitcond = icmp eq i64 %add, 20
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !0

for.end:
  ret void
}

!0 = distinct !{!0, !1, !2, !3, !4, !5}
!1 = !{!"llvm.loop.distribute.enable", i1 true}
!2 = !{!"llvm.loop.distribute.followup_all", !{!"FollowupAll"}}
!3 = !{!"llvm.loop.distribute.followup_coincident", !{!"FollowupCoincident", i1 false}}
!4 = !{!"llvm.loop.distribute.followup_sequential", !{!"FollowupSequential", i32 8}}
!5 = !{!"llvm.loop.distribute.followup_fallback", !{!"FollowupFallback"}}


; CHECK-LABEL: for.body.lver.orig:
; CHECK: br i1 %exitcond.lver.orig, label %for.end.loopexit, label %for.body.lver.orig, !llvm.loop ![[LOOP_ORIG:[0-9]+]]
; CHECK-LABEL: for.body.ldist1:
; CHECK: br i1 %exitcond.ldist1, label %for.body.ph, label %for.body.ldist1, !llvm.loop ![[LOOP_SEQUENTIAL:[0-9]+]]
; CHECK-LABEL: for.body:
; CHECK: br i1 %exitcond, label %for.end.loopexit26, label %for.body, !llvm.loop ![[LOOP_COINCIDENT:[0-9]+]]
; CHECK-LABEL: for.end.loopexit:
; CHECK: br label %for.end
; CHECK-LABEL: for.end.loopexit26:
; CHECK: br label %for.end

; CHECK: ![[LOOP_ORIG]] = distinct !{![[LOOP_ORIG]], ![[FOLLOWUP_ALL:[0-9]+]], ![[FOLLOUP_FALLBACK:[0-9]+]]}
; CHECK: ![[FOLLOWUP_ALL]] = !{!"FollowupAll"}
; CHECK: ![[FOLLOUP_FALLBACK]] = !{!"FollowupFallback"}
; CHECK: ![[LOOP_SEQUENTIAL]] = distinct !{![[LOOP_SEQUENTIAL]], ![[FOLLOWUP_ALL]], ![[FOLLOWUP_SEQUENTIAL:[0-9]+]]}
; CHECK: ![[FOLLOWUP_SEQUENTIAL]] = !{!"FollowupSequential", i32 8}
; CHECK: ![[LOOP_COINCIDENT]] = distinct !{![[LOOP_COINCIDENT]], ![[FOLLOWUP_ALL]], ![[FOLLOWUP_COINCIDENT:[0-9]+]]}
; CHECK: ![[FOLLOWUP_COINCIDENT]] = !{!"FollowupCoincident", i1 false}
