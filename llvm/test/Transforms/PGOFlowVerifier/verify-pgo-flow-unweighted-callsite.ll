; RUN: opt < %s -passes=verify-pgo-flow -disable-output 2>&1 | FileCheck %s
; RUN: opt < %s -passes=verify-pgo-flow -verify-pgo-flow-fatal \
; RUN:     -disable-output 2>&1 | FileCheck %s
;
; Two unweighted calls in one block must not each inherit the block SumIn
; (that would be entry=10 vs caller-sum=20). Treat missing call !prof as
; unknown and skip the callee's entry-count check.

; CHECK: *** PGO Flow Verification After verify-pgo-flow ***{{$}}
; CHECK-NOT: PGOFlowVerify[EntryCountMismatch]

define internal i32 @unweighted_callee(i32 %x) !prof !0 {
entry:
  ret i32 %x
}

define i32 @unweighted_caller(i32 %x) !prof !0 {
entry:
  %a = call i32 @unweighted_callee(i32 %x)
  %b = call i32 @unweighted_callee(i32 %x)
  ret i32 %b
}

!0 = !{!"function_entry_count", i64 10}

!llvm.module.flags = !{!10}
!10 = !{i32 1, !"ProfileSummary", !11}
!11 = !{!12, !13, !14, !15, !16, !17, !18, !19}
!12 = !{!"ProfileFormat", !"InstrProf"}
!13 = !{!"TotalCount", i64 10}
!14 = !{!"MaxCount", i64 10}
!15 = !{!"MaxInternalCount", i64 10}
!16 = !{!"MaxFunctionCount", i64 10}
!17 = !{!"NumCounts", i64 2}
!18 = !{!"NumFunctions", i64 2}
!19 = !{!"DetailedSummary", !20}
!20 = !{!21}
!21 = !{i32 10000, i64 10, i32 1}
