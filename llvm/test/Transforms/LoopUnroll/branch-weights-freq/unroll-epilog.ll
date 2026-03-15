; Test branch weight metadata, estimated trip count metadata, and block
; frequencies after loop unrolling with an epilogue.
;
; We check various interesting unroll count values relative to the original
; loop's body frequency of 11 (e.g., minimum and boundary values).
;
; For each case, we check:
; - Iteration frequencies
;   - When each is multiplied by the number of original loop bodies that execute
;     within it, they should sum to almost exactly the original loop body
;     frequency, 11.
; - CFGs
;   - We verify which branch weights go with which branches and that we did not
;     overlook any other branch weights (no extra !prof or branch_weights).
;   - We also check the number of original loop bodies (represented by a call to
;     @f) that appear within each unrolled iteration.
; - llvm.loop.estimated_trip_count
;   - For the unrolled and epilogue loops, must be the number of iterations
;     required for the original loop body to reach its original estimated trip
;     count, which is its original frequency, 11, because there is no prior
;     llvm.loop.estimated_trip_count.
;   - Must not be blindly duplicated between the unrolled and epilogue loops.
;   - Must not be blindly computed from any new latch branch weights.

; ------------------------------------------------------------------------------
; Verify that the test code produces the original loop body frequency we expect.
;
; RUN: opt %s -S -passes='print<block-freq>' 2>&1 | \
; RUN:   FileCheck %s -check-prefixes ORIG
;
; ORIG: - do.body: float = 11.0,

; ------------------------------------------------------------------------------
; Define LIT substitutions for checking the unrolled and epilogue loop.
;
; DEFINE: %{ur-bf} = opt %s -S -passes='loop-unroll,print<block-freq>' 2>&1 \
; DEFINE:     -unroll-runtime
; DEFINE: %{fc} = FileCheck %s \
; DEFINE:     -implicit-check-not='llvm.loop.estimated_trip_count' \
; DEFINE:     -implicit-check-not='!prof' \
; DEFINE:     -implicit-check-not='branch_weights' \
; DEFINE:     -implicit-check-not='call void @f' -check-prefixes

; ------------------------------------------------------------------------------
; Check -unroll-count=2.
;
; RUN: %{ur-bf} -unroll-count=2 | %{fc} UR2
;
; Multiply do.body by 2 and add do.body.epil to get the original loop body
; frequency, 11.
; UR2: - do.body: float = 5.2381,
; UR2: - do.body.epil: float = 0.52381,
;
; Unrolled loop guard, body, and latch.
; UR2: br i1 %{{.*}}, label %do.body.epil.preheader, label %entry.new, !prof !0
; UR2-COUNT-2: call void @f
; UR2: br i1 %{{.*}}, label %do.end.unr-lcssa, label %do.body, !prof !1, !llvm.loop !2
;
; Epilogue guard and its sole iteration, so it is completely unrolled with no
; remaining conditional latches.
; UR2: br i1 %{{.*}}, label %do.body.epil.preheader, label %do.end, !prof !5
; UR2: call void @f
;
; Unrolled loop metadata.
; UR2: !0 = !{!"branch_weights", i32 195225786, i32 1952257862}
; UR2: !1 = !{!"branch_weights", i32 372703773, i32 1774779875}
; UR2: !2 = distinct !{!2, !3, !4}
; UR2: !3 = !{!"llvm.loop.estimated_trip_count", i32 5}
; UR2: !4 = !{!"llvm.loop.unroll.disable"}
; UR2: !5 = !{!"branch_weights", i32 1022611260, i32 1124872388}

; ------------------------------------------------------------------------------
; Check -unroll-count=4.
;
; RUN: %{ur-bf} -unroll-count=4 | %{fc} UR4
;
; Multiply do.body by 4 and add do.body.epil* to get the original loop body
; frequency, 11.
; UR4: - do.body: float = 2.3702,
; UR4: - do.body.epil: float = 1.5193,
;
; Unrolled loop guard, body, and latch.
; UR4: br i1 %{{.*}}, label %do.body.epil.preheader, label %entry.new, !prof !0
; UR4-COUNT-4: call void @f
; UR4: br i1 %{{.*}}, label %do.end.unr-lcssa, label %do.body, !prof !1, !llvm.loop !2
;
; Epilogue guard and loop.
; UR4: br i1 %{{.*}}, label %do.body.epil.preheader, label %do.end, !prof !5
; UR4: call void @f
; UR4: br i1 %{{.*}}, label %do.body.epil, label %do.end.epilog-lcssa, !prof !6, !llvm.loop !7
;
; Unrolled loop metadata.
; UR4: !0 = !{!"branch_weights", i32 534047398, i32 1613436250}
; UR4: !1 = !{!"branch_weights", i32 680723421, i32 1466760227}
; UR4: !2 = distinct !{!2, !3, !4}
; UR4: !3 = !{!"llvm.loop.estimated_trip_count", i32 2}
; UR4: !4 = !{!"llvm.loop.unroll.disable"}
; UR4: !5 = !{!"branch_weights", i32 1531603292, i32 615880356}
;
; Epilogue loop metadata.
; UR4: !6 = !{!"branch_weights", i32 1038564635, i32 1108919013}
; UR4: !7 = distinct !{!7, !8, !4}
; UR4: !8 = !{!"llvm.loop.estimated_trip_count", i32 3}

; ------------------------------------------------------------------------------
; Check -unroll-count=10.
;
; RUN: %{ur-bf} -unroll-count=10 | %{fc} UR10
;
; Multiply do.body by 8 and add do.body.epil* to get the original loop body
; frequency, 11.
; UR10: - do.body: float = 0.6902,
; UR10: - do.body.epil: float = 4.098,
;
; Unrolled loop guard, body, and latch.
; UR10: br i1 %{{.*}}, label %do.body.epil.preheader, label %entry.new, !prof !0
; UR10-COUNT-10: call void @f
; UR10: br i1 %{{.*}}, label %do.end.unr-lcssa, label %do.body, !prof !1, !llvm.loop !2
;
; Epilogue guard and loop.
; UR10: br i1 %{{.*}}, label %do.body.epil.preheader, label %do.end, !prof !5
; UR10: call void @f
; UR10: br i1 %{{.*}}, label %do.body.epil, label %do.end.epilog-lcssa, !prof !6, !llvm.loop !7
;
; Unrolled loop metadata.
; UR10: !0 = !{!"branch_weights", i32 1236740947, i32 910742701}
; UR10: !1 = !{!"branch_weights", i32 1319535738, i32 827947910}
; UR10: !2 = distinct !{!2, !3, !4}
; UR10: !3 = !{!"llvm.loop.estimated_trip_count", i32 1}
; UR10: !4 = !{!"llvm.loop.unroll.disable"}
; UR10: !5 = !{!"branch_weights", i32 1829762672, i32 317720976}
;
; Epilogue loop metadata.  Its llvm.loop.estimated_trip_count happens to be the
; same as the unrolled loop's, so there's no new metadata node.
; UR10: !6 = !{!"branch_weights", i32 1656332913, i32 491150735}
; UR10: !7 = distinct !{!7, ![[#LOOP_UR_TC:]], ![[#DISABLE:]]}

; ------------------------------------------------------------------------------
; Check -unroll-count=11.
;
; RUN: %{ur-bf} -unroll-count=11 | %{fc} UR11
;
; Multiply do.body by 11 and add do.body.epil* to get the original loop body
; frequency, 11.
; UR11: - do.body: float = 0.59359,
; UR11: - do.body.epil: float = 4.4705,
;
; Unrolled loop guard, body, and latch.
; UR11: br i1 %{{.*}}, label %do.body.epil.preheader, label %entry.new, !prof !0
; UR11-COUNT-11: call void @f
; UR11: br i1 %{{.*}}, label %do.end.unr-lcssa, label %do.body, !prof !1, !llvm.loop !2

; Epilogue guard and loop.
; UR11: br i1 %{{.*}}, label %do.body.epil.preheader, label %do.end, !prof !5
; UR11: call void @f
; UR11: br i1 %{{.*}}, label %do.body.epil, label %do.end.epilog-lcssa, !prof !6, !llvm.loop !7
;
; Unrolled loop metadata.
; UR11: !0 = !{!"branch_weights", i32 1319535738, i32 827947910}
; UR11: !1 = !{!"branch_weights", i32 1394803730, i32 752679918}
; UR11: !2 = distinct !{!2, !3, !4}
; UR11: !3 = !{!"llvm.loop.estimated_trip_count", i32 1}
; UR11: !4 = !{!"llvm.loop.unroll.disable"}
; UR11: !5 = !{!"branch_weights", i32 1846907894, i32 300575754}
;
; Epilogue loop metadata.
; UR11: !6 = !{!"branch_weights", i32 1693034047, i32 454449601}
; UR11: !7 = distinct !{!7, !8, !4}
; UR11: !8 = !{!"llvm.loop.estimated_trip_count", i32 0}

; ------------------------------------------------------------------------------
; Check -unroll-count=12.
;
; RUN: %{ur-bf} -unroll-count=12 | %{fc} UR12
;
; Multiply do.body by 12 and add do.body.epil* to get the original loop body
; frequency, 11.
; UR12: - do.body: float = 0.5144,
; UR12: - do.body.epil: float = 4.8272,
;
; Unrolled loop guard, body, and latch.
; UR12: br i1 %{{.*}}, label %do.body.epil.preheader, label %entry.new, !prof !0
; UR12-COUNT-12: call void @f
; UR12: br i1 %{{.*}}, label %do.end.unr-lcssa, label %do.body, !prof !1, !llvm.loop !2
;
; Epilogue guard and loop.
; UR12: br i1 %{{.*}}, label %do.body.epil.preheader, label %do.end, !prof !5
; UR12: call void @f
; UR12: br i1 %{{.*}}, label %do.body.epil, label %do.end.epilog-lcssa, !prof !6, !llvm.loop !7
;
; Unrolled loop metadata.
; UR12: !0 = !{!"branch_weights", i32 1394803730, i32 752679918}
; UR12: !1 = !{!"branch_weights", i32 1463229177, i32 684254471}
; UR12: !2 = distinct !{!2, !3, !4}
; UR12: !3 = !{!"llvm.loop.estimated_trip_count", i32 0}
; UR12: !4 = !{!"llvm.loop.unroll.disable"}
; UR12: !5 = !{!"branch_weights", i32 1860963812, i32 286519836}
;
; Epilogue loop metadata.
; UR12: !6 = !{!"branch_weights", i32 1723419551, i32 424064097}
; UR12: !7 = distinct !{!7, !8, !4}
; UR12: !8 = !{!"llvm.loop.estimated_trip_count", i32 11}

declare void @f(i32)

define void @test(i32 %n) {
entry:
  br label %do.body

do.body:
  %i = phi i32 [ 0, %entry ], [ %inc, %do.body ]
  %inc = add i32 %i, 1
  call void @f(i32 %i)
  %c = icmp sge i32 %inc, %n
  br i1 %c, label %do.end, label %do.body, !prof !0

do.end:
  ret void
}

; Loop body frequency is 11.
!0 = !{!"branch_weights", i32 1, i32 10}
