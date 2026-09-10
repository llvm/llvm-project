; RUN: opt < %s -passes='function(loop-unroll),verify-pgo-flow' \
; RUN:     -disable-output 2>&1 | FileCheck %s
;
; Unrolling A marks the function approxprofile, so B's BFI mismatch is skipped.

; CHECK: *** PGO Flow Verification After verify-pgo-flow ***{{$}}
; CHECK: PGOFlowVerify[ApproxProfileSkip] two_loops: skipping strict InstrProf verification (approxprofile)
; CHECK-NOT: PGOFlowVerify[BlockFrequencyMismatch]

define void @two_loops(i32 %n, i32 %m) !prof !0 {
entry:
  br label %loop_a

loop_a:
  %ia = phi i32 [ 0, %entry ], [ %inca, %loop_a ]
  %inca = add i32 %ia, 1
  %ca = icmp sge i32 %inca, %n
  br i1 %ca, label %loop_b.preheader, label %loop_a, !prof !1, !llvm.loop !10

loop_b.preheader:
  br label %loop_b

loop_b:
  %ib = phi i32 [ 0, %loop_b.preheader ], [ %incb, %loop_b ]
  %incb = add i32 %ib, 1
  %cb = icmp sge i32 %incb, %m
  br i1 %cb, label %exit, label %loop_b, !prof !2, !llvm.loop !11

exit:
  ret void
}

!0 = !{!"function_entry_count", i64 10}
!1 = !{!"branch_weights", i32 1, i32 9}
!2 = !{!"branch_weights", i32 10, i32 1}

!10 = distinct !{!10, !12}
!11 = distinct !{!11, !13}
!12 = !{!"llvm.loop.unroll.count", i32 2}
!13 = !{!"llvm.loop.unroll.disable"}

!llvm.module.flags = !{!20}
!20 = !{i32 1, !"ProfileSummary", !21}
!21 = !{!22, !23, !24, !25, !26, !27, !28, !29}
!22 = !{!"ProfileFormat", !"InstrProf"}
!23 = !{!"TotalCount", i64 20}
!24 = !{!"MaxCount", i64 10}
!25 = !{!"MaxInternalCount", i64 10}
!26 = !{!"MaxFunctionCount", i64 10}
!27 = !{!"NumCounts", i64 4}
!28 = !{!"NumFunctions", i64 1}
!29 = !{!"DetailedSummary", !30}
!30 = !{!31}
!31 = !{i32 10000, i64 10, i32 1}
