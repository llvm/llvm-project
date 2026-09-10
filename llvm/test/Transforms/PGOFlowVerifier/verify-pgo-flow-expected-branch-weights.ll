; RUN: opt < %s -passes=verify-pgo-flow -disable-output 2>&1 | FileCheck %s
;
; !"expected" branch_weights are probabilities (llvm.expect), not InstrProf
; counts. Do not report BlockFrequencyMismatch for well-formed expected MD.

; CHECK: *** PGO Flow Verification After verify-pgo-flow ***{{$}}
; CHECK-NOT: PGOFlowVerify[BlockFrequencyMismatch]

define i32 @expected_br(i32 %x) !prof !0 {
entry:
  %c = icmp sgt i32 %x, 0
  br i1 %c, label %then, label %else, !prof !1

then:
  ret i32 1

else:
  ret i32 0
}

!0 = !{!"function_entry_count", i64 10}
!1 = !{!"branch_weights", !"expected", i32 1, i32 2000}

!llvm.module.flags = !{!10}
!10 = !{i32 1, !"ProfileSummary", !11}
!11 = !{!12, !13, !14, !15, !16, !17, !18, !19}
!12 = !{!"ProfileFormat", !"InstrProf"}
!13 = !{!"TotalCount", i64 10}
!14 = !{!"MaxCount", i64 10}
!15 = !{!"MaxInternalCount", i64 10}
!16 = !{!"MaxFunctionCount", i64 10}
!17 = !{!"NumCounts", i64 2}
!18 = !{!"NumFunctions", i64 1}
!19 = !{!"DetailedSummary", !20}
!20 = !{!21}
!21 = !{i32 10000, i64 10, i32 1}
