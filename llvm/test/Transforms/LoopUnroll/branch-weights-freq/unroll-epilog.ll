; Test branch weight metadata, estimated trip count metadata, and block
; frequencies after loop unrolling with an epilogue.

; ------------------------------------------------------------------------------
; Define substitutions.
;
; Check original loop body frequency.
; DEFINE: %{bf-fc} = opt %s -S -passes='print<block-freq>' 2>&1 | \
; DEFINE:   FileCheck %s -check-prefixes
;
; Unroll loops and then check block frequency.  The -implicit-check-not options
; make sure that no additional labels or @f calls show up.
; DEFINE: %{ur-bf} = opt %s -S -passes='loop-unroll,print<block-freq>' 2>&1
; DEFINE: %{fc} = FileCheck %s \
; DEFINE:     -implicit-check-not='{{^( *- )?[^ ;]*:}}' \
; DEFINE:     -implicit-check-not='call void @f' -check-prefixes

; ------------------------------------------------------------------------------
; Check various interesting unroll count values relative to the original loop's
; estimated trip count of 11 (e.g., minimum and boundary values).
;
; RUN: %{bf-fc} ALL,ORIG
; RUN: %{ur-bf} -unroll-count=2 -unroll-runtime | %{fc} ALL,UR,UR2
; RUN: %{ur-bf} -unroll-count=4 -unroll-runtime | %{fc} ALL,UR,UR4
; RUN: %{ur-bf} -unroll-count=10 -unroll-runtime | %{fc} ALL,UR,UR10
; RUN: %{ur-bf} -unroll-count=11 -unroll-runtime | %{fc} ALL,UR,UR11
; RUN: %{ur-bf} -unroll-count=12 -unroll-runtime | %{fc} ALL,UR,UR12

; ------------------------------------------------------------------------------
; Check the iteration frequencies, which, when each is multiplied by the number
; of original loop bodies that execute within it, should sum to almost exactly
; the original loop body frequency.
;
; ALL-LABEL: block-frequency-info: test
;
;      ORIG: - [[ENTRY:.*]]:
;      ORIG: - [[DO_BODY:.*]]: float = 11.0,
;      ORIG: - [[DO_END:.*]]:
;
;        UR: - [[ENTRY:.*]]:
;        UR: - [[ENTRY_NEW:.*]]:
;       UR2: - [[DO_BODY:.*]]: float = 5.2381,
;       UR4: - [[DO_BODY:.*]]: float = 2.3702,
;      UR10: - [[DO_BODY:.*]]: float = 0.6902,
;      UR11: - [[DO_BODY:.*]]: float = 0.59359,
;      UR12: - [[DO_BODY:.*]]: float = 0.5144,
;        UR: - [[DO_END_UNR_LCSSA:.*]]:
;        UR: - [[DO_BODY_EPIL_PREHEADER:.*]]:
;       UR2: - [[DO_BODY_EPIL:.*]]: float = 0.52381,
;       UR4: - [[DO_BODY_EPIL:.*]]: float = 1.5193,
;      UR10: - [[DO_BODY_EPIL:.*]]: float = 4.098,
;      UR11: - [[DO_BODY_EPIL:.*]]: float = 4.4705,
;      UR12: - [[DO_BODY_EPIL:.*]]: float = 4.8272,
;       UR4: - [[DO_END_EPILOG_LCSSA:.*]]:
;      UR10: - [[DO_END_EPILOG_LCSSA:.*]]:
;      UR11: - [[DO_END_EPILOG_LCSSA:.*]]:
;      UR12: - [[DO_END_EPILOG_LCSSA:.*]]:
;        UR: - [[DO_END:.*]]:

; ------------------------------------------------------------------------------
; Check the CFGs, including the number of original loop bodies that appear
; within each unrolled iteration.
;
;      UR-LABEL: define void @test(i32 %{{.*}}) {
;            UR: [[ENTRY]]:
;            UR:   br i1 %{{.*}}, label %[[DO_BODY_EPIL_PREHEADER]], label %[[ENTRY_NEW]], !prof ![[#PROF_UR_GUARD:]]{{$}}
;            UR: [[ENTRY_NEW]]:
;            UR:   br label %[[DO_BODY]]
;            UR: [[DO_BODY]]:
;   UR2-COUNT-2:   call void @f
;   UR4-COUNT-4:   call void @f
; UR10-COUNT-10:   call void @f
; UR11-COUNT-11:   call void @f
; UR12-COUNT-12:   call void @f
;            UR:   br i1 %{{.*}}, label %[[DO_END_UNR_LCSSA]], label %[[DO_BODY]], !prof ![[#PROF_UR_LATCH:]], !llvm.loop ![[#LOOP_UR_LATCH:]]{{$}}
;            UR: [[DO_END_UNR_LCSSA]]:
;            UR:   br i1 %{{.*}}, label %[[DO_BODY_EPIL_PREHEADER]], label %[[DO_END:.*]], !prof ![[#PROF_RM_GUARD:]]{{$}}
;            UR: [[DO_BODY_EPIL_PREHEADER]]:
;            UR:   br label %[[DO_BODY_EPIL]]
;            UR: [[DO_BODY_EPIL]]:
;            UR:   call void @f
;           UR4:   br i1 %{{.*}}, label %[[DO_BODY_EPIL]], label %[[DO_END_EPILOG_LCSSA]], !prof ![[#PROF_RM_LATCH:]], !llvm.loop ![[#LOOP_RM_LATCH:]]{{$}}
;          UR10:   br i1 %{{.*}}, label %[[DO_BODY_EPIL]], label %[[DO_END_EPILOG_LCSSA]], !prof ![[#PROF_RM_LATCH:]], !llvm.loop ![[#LOOP_RM_LATCH:]]{{$}}
;          UR11:   br i1 %{{.*}}, label %[[DO_BODY_EPIL]], label %[[DO_END_EPILOG_LCSSA]], !prof ![[#PROF_RM_LATCH:]], !llvm.loop ![[#LOOP_RM_LATCH:]]{{$}}
;          UR12:   br i1 %{{.*}}, label %[[DO_BODY_EPIL]], label %[[DO_END_EPILOG_LCSSA]], !prof ![[#PROF_RM_LATCH:]], !llvm.loop ![[#LOOP_RM_LATCH:]]{{$}}
;           UR4: [[DO_END_EPILOG_LCSSA]]:
;          UR10: [[DO_END_EPILOG_LCSSA]]:
;          UR11: [[DO_END_EPILOG_LCSSA]]:
;          UR12: [[DO_END_EPILOG_LCSSA]]:
;            UR:   br label %[[DO_END]]
;            UR: [[DO_END]]:
;            UR:   ret void

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

!0 = !{!"branch_weights", i32 1, i32 10}

; ------------------------------------------------------------------------------
; Check branch weight metadata and estimated trip count metadata.
;
;  UR2: ![[#PROF_UR_GUARD]] = !{!"branch_weights", i32 195225786, i32 1952257862}
;  UR4: ![[#PROF_UR_GUARD]] = !{!"branch_weights", i32 534047398, i32 1613436250}
; UR10: ![[#PROF_UR_GUARD]] = !{!"branch_weights", i32 1236740947, i32 910742701}
; UR11: ![[#PROF_UR_GUARD]] = !{!"branch_weights", i32 1319535738, i32 827947910}
; UR12: ![[#PROF_UR_GUARD]] = !{!"branch_weights", i32 1394803730, i32 752679918}
;
;  UR2: ![[#PROF_UR_LATCH]] = !{!"branch_weights", i32 372703773, i32 1774779875}
;  UR4: ![[#PROF_UR_LATCH]] = !{!"branch_weights", i32 680723421, i32 1466760227}
; UR10: ![[#PROF_UR_LATCH]] = !{!"branch_weights", i32 1319535738, i32 827947910}
; UR11: ![[#PROF_UR_LATCH]] = !{!"branch_weights", i32 1394803730, i32 752679918}
; UR12: ![[#PROF_UR_LATCH]] = !{!"branch_weights", i32 1463229177, i32 684254471}
;
;  UR2: ![[#LOOP_UR_LATCH]] = distinct !{![[#LOOP_UR_LATCH]], ![[#LOOP_UR_TC:]], ![[#DISABLE:]]}
;  UR4: ![[#LOOP_UR_LATCH]] = distinct !{![[#LOOP_UR_LATCH]], ![[#LOOP_UR_TC:]], ![[#DISABLE:]]}
; UR10: ![[#LOOP_UR_LATCH]] = distinct !{![[#LOOP_UR_LATCH]], ![[#LOOP_UR_TC:]], ![[#DISABLE:]]}
; UR11: ![[#LOOP_UR_LATCH]] = distinct !{![[#LOOP_UR_LATCH]], ![[#LOOP_UR_TC:]], ![[#DISABLE:]]}
; UR12: ![[#LOOP_UR_LATCH]] = distinct !{![[#LOOP_UR_LATCH]], ![[#LOOP_UR_TC:]], ![[#DISABLE:]]}
;
;  UR2: ![[#LOOP_UR_TC]] = !{!"llvm.loop.estimated_trip_count", i32 5}
;  UR4: ![[#LOOP_UR_TC]] = !{!"llvm.loop.estimated_trip_count", i32 2}
; UR10: ![[#LOOP_UR_TC]] = !{!"llvm.loop.estimated_trip_count", i32 1}
; UR11: ![[#LOOP_UR_TC]] = !{!"llvm.loop.estimated_trip_count", i32 1}
; UR12: ![[#LOOP_UR_TC]] = !{!"llvm.loop.estimated_trip_count", i32 0}
;   UR: ![[#DISABLE]] = !{!"llvm.loop.unroll.disable"}
;
;  UR2: ![[#PROF_RM_GUARD]] = !{!"branch_weights", i32 1022611260, i32 1124872388}
;  UR4: ![[#PROF_RM_GUARD]] = !{!"branch_weights", i32 1531603292, i32 615880356}
; UR10: ![[#PROF_RM_GUARD]] = !{!"branch_weights", i32 1829762672, i32 317720976}
; UR11: ![[#PROF_RM_GUARD]] = !{!"branch_weights", i32 1846907894, i32 300575754}
; UR12: ![[#PROF_RM_GUARD]] = !{!"branch_weights", i32 1860963812, i32 286519836}
;
;  UR4: ![[#PROF_RM_LATCH]] = !{!"branch_weights", i32 1038564635, i32 1108919013}
; UR10: ![[#PROF_RM_LATCH]] = !{!"branch_weights", i32 1656332913, i32 491150735}
; UR11: ![[#PROF_RM_LATCH]] = !{!"branch_weights", i32 1693034047, i32 454449601}
; UR12: ![[#PROF_RM_LATCH]] = !{!"branch_weights", i32 1723419551, i32 424064097}

;  UR4: ![[#LOOP_RM_LATCH]] = distinct !{![[#LOOP_RM_LATCH]], ![[#LOOP_RM_TC:]], ![[#DISABLE:]]}
; UR10: ![[#LOOP_RM_LATCH]] = distinct !{![[#LOOP_RM_LATCH]], ![[#LOOP_UR_TC:]], ![[#DISABLE:]]}
; UR11: ![[#LOOP_RM_LATCH]] = distinct !{![[#LOOP_RM_LATCH]], ![[#LOOP_RM_TC:]], ![[#DISABLE:]]}
; UR12: ![[#LOOP_RM_LATCH]] = distinct !{![[#LOOP_RM_LATCH]], ![[#LOOP_RM_TC:]], ![[#DISABLE:]]}
;
;  UR4: ![[#LOOP_RM_TC]] = !{!"llvm.loop.estimated_trip_count", i32 3}
; For UR10, llvm.loop.estimated_trip_count is the same for both loops.
; UR11: ![[#LOOP_RM_TC]] = !{!"llvm.loop.estimated_trip_count", i32 0}
; UR12: ![[#LOOP_RM_TC]] = !{!"llvm.loop.estimated_trip_count", i32 11}
