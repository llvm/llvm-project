; RUN: opt < %s -passes=verify-pgo-flow -disable-output 2>&1 | FileCheck %s
;
; Leftover call !prof on an unreachable block is not live flow. Do not add
; it into the callee's caller-sum (would be entry=1 vs caller-sum=10).

; CHECK: *** PGO Flow Verification After verify-pgo-flow ***{{$}}
; CHECK-NOT: PGOFlowVerify[EntryCountMismatch]

define internal i32 @dead_callee(i32 %x) !prof !0 {
entry:
  ret i32 %x
}

define i32 @live_and_dead_caller(i32 %x) !prof !1 {
entry:
  %r = call i32 @dead_callee(i32 %x), !prof !2
  ret i32 %r

dead:
  %d = call i32 @dead_callee(i32 %x), !prof !3
  ret i32 %d
}

define internal i32 @only_dead_callee(i32 %x) !prof !0 {
entry:
  ret i32 %x
}

define i32 @only_dead_caller(i32 %x) !prof !1 {
entry:
  ret i32 %x

dead:
  %d = call i32 @only_dead_callee(i32 %x), !prof !3
  ret i32 %d
}

!0 = !{!"function_entry_count", i64 1}
!1 = !{!"function_entry_count", i64 1}
!2 = !{!"branch_weights", i32 1}
!3 = !{!"branch_weights", i32 10}

!llvm.module.flags = !{!10}
!10 = !{i32 1, !"ProfileSummary", !11}
!11 = !{!12, !13, !14, !15, !16, !17, !18, !19}
!12 = !{!"ProfileFormat", !"InstrProf"}
!13 = !{!"TotalCount", i64 12}
!14 = !{!"MaxCount", i64 10}
!15 = !{!"MaxInternalCount", i64 10}
!16 = !{!"MaxFunctionCount", i64 10}
!17 = !{!"NumCounts", i64 4}
!18 = !{!"NumFunctions", i64 4}
!19 = !{!"DetailedSummary", !20}
!20 = !{!21}
!21 = !{i32 10000, i64 10, i32 1}
