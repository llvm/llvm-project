; RUN: opt < %s -passes=verify-pgo-flow -disable-output 2>&1 | FileCheck %s
;
; Two switch cases share a successor. pred_size counts Uses, not unique
; predecessors, so both edges are unknown-ins and conservation holds.

; CHECK: *** PGO Flow Verification After verify-pgo-flow ***{{$}}
; CHECK-NOT: PGOFlowVerify[BlockFrequencyMismatch]

define i32 @switch_multi(i32 %x) !prof !0 {
entry:
  switch i32 %x, label %def [
    i32 0, label %hot
    i32 1, label %hot
  ], !prof !1

hot:
  ret i32 1

def:
  ret i32 0
}

!0 = !{!"function_entry_count", i64 10}
!1 = !{!"branch_weights", i32 1, i32 6, i32 3}

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
