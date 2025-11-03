; RUN: opt < %s -S -passes=loop-unroll -unroll-runtime=true -unroll-count=4 | FileCheck %s

;; Check that the remainder loop is properly assigned a branch weight for its latch branch.
; CHECK-LABEL: @test(
; CHECK-LABEL: entry:
;       CHECK: [[FOR_BODY_PREHEADER:.*]]:
;       CHECK:   br i1 %{{.*}}, label %[[FOR_BODY_EPIL_PREHEADER:.*]], label %[[FOR_BODY_PREHEADER_NEW:.*]], !prof ![[#PROF_UR_GUARD:]]
;       CHECK: [[FOR_BODY_PREHEADER_NEW]]:
;       CHECK:   br label %for.body
;       CHECK: for.body:
;       CHECK:   %add = add
;       CHECK:   %add.1 = add
;       CHECK:   %add.2 = add
;       CHECK:   %add.3 = add
;   CHECK-NOT:   %add.4 = add
;       CHECK:   br i1 %{{.*}}, label %[[FOR_END_LOOPEXIT_UNR_LCSSA:.*]], label %for.body, !prof ![[#PROF_UR_LATCH:]], !llvm.loop ![[#LOOP_UR_LATCH:]]
;       CHECK: [[FOR_END_LOOPEXIT_UNR_LCSSA]]:
;       CHECK:   br i1 %{{.*}}, label %[[FOR_BODY_EPIL_PREHEADER]], label %[[FOR_END_LOOPEXIT:.*]], !prof ![[#PROF_RM_GUARD:]]
;       CHECK: [[FOR_BODY_EPIL_PREHEADER]]:
;       CHECK:   br label %[[FOR_BODY_EPIL:.*]]
;       CHECK: [[FOR_BODY_EPIL]]:
;       CHECK:   br i1 {{.*}}, label %[[FOR_BODY_EPIL]], label %[[FOR_END_LOOPEXIT_EPILOG_LCSSA:.*]], !prof ![[#PROF_RM_LATCH:]], !llvm.loop ![[#LOOP_RM_LATCH:]]

define i3 @test(ptr %a, i3 %n) {
entry:
  %cmp1 = icmp eq i3 %n, 0
  br i1 %cmp1, label %for.end, label %for.body

for.body:
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %sum.02 = phi i3 [ %add, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i3, ptr %a, i64 %indvars.iv
  %0 = load i3, ptr %arrayidx
  %add = add nsw i3 %0, %sum.02
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i3
  %exitcond = icmp eq i3 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body, !prof !0

for.end:
  %sum.0.lcssa = phi i3 [ 0, %entry ], [ %add, %for.body ]
  ret i3 %sum.0.lcssa
}

!0 = !{!"branch_weights", i32 1, i32 9999}

; Original loop probability: p = 9999/(1+9999) = 0.9999
; Original estimated trip count: (1+9999)/1 = 10000
; Unroll count: 4

; Probability of >=3 iterations after first: p^3 = 0.9970003 =~
; 2146839468 / (644180 + 2146839468).
; CHECK: ![[#PROF_UR_GUARD]] = !{!"branch_weights", i32 644180, i32 2146839468}

; Probability of >=4 more iterations: p^4 = 0.99960006 =~
; 2146624784 / (858864 + 2146624784).
; CHECK: ![[#PROF_UR_LATCH]] = !{!"branch_weights", i32 858864, i32 2146624784}

; 10000//4 = 2500
; CHECK: ![[#LOOP_UR_LATCH]] = distinct !{![[#LOOP_UR_LATCH]], ![[#LOOP_UR_TC:]], ![[#DISABLE:]]}
; CHECK: ![[#LOOP_UR_TC]] = !{!"llvm.loop.estimated_trip_count", i32 2500}
;
; CHECK: ![[#DISABLE]] = !{!"llvm.loop.unroll.disable"}

; Probability of 1 to 3 more of 3 more remainder iterations:
; (p-p^4)/(1-p^4) = 0.749962497 =~ 1610532724 / (1610532724 + 536950924).
; CHECK: ![[#PROF_RM_GUARD]] = !{!"branch_weights", i32 1610532724, i32 536950924}

; Frequency of first remainder iter:  r1 =                      1
; Frequency of second remainder iter: r2 = r1*(p-p^3)/(1-p^3) = 0.666633331
; Frequency of third remainder iter:  r3 = r2*(p-p^2)/(1-p^2) = 0.333299999
; Solve for loop probability that produces that frequency: f = 1/(1-p') =>
; p' = 1-1/f = 1-1/(r1+r2+r3) = 0.499983332 =~
; 1073706403 / (1073706403 + 1073777245).
; CHECK: ![[#PROF_RM_LATCH]] = !{!"branch_weights", i32 1073706403, i32 1073777245}

; 10000%4 = 0
; CHECK: ![[#LOOP_RM_LATCH]] = distinct !{![[#LOOP_RM_LATCH]], ![[#LOOP_RM_TC:]], ![[#DISABLE:]]}
; CHECK: ![[#LOOP_RM_TC]] = !{!"llvm.loop.estimated_trip_count", i32 0}
