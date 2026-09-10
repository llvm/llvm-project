; RUN: opt < %s -passes=verify-pgo-flow -disable-output 2>&1 | FileCheck %s
;
; Direct callsites must not use extractProfTotalWeight on leftover VP
; (operand 2 is the aggregate over every target) or llvm.expect weights.
; Count-type branch_weights still credit. Unknown sites must not hide a
; definite overcount from a weighted caller.

; CHECK: *** PGO Flow Verification After verify-pgo-flow ***{{$}}
; CHECK: PGOFlowVerify[EntryCountMismatch] mixed_callee: entry=1 vs caller-sum=10
; CHECK-NOT: PGOFlowVerify[EntryCountMismatch] vp_only_callee:
; CHECK-NOT: PGOFlowVerify[EntryCountMismatch] expected_only_callee:

define internal i32 @mixed_callee(i32 %x) !prof !0 {
entry:
  ret i32 %x
}

define i32 @folded_vp_caller(i32 %x) !prof !1 {
entry:
  %r = call i32 @mixed_callee(i32 %x), !prof !3
  ret i32 %r
}

define i32 @expected_caller(i32 %x) !prof !1 {
entry:
  %r = call i32 @mixed_callee(i32 %x), !prof !4
  ret i32 %r
}

define i32 @weighted_caller(i32 %x) !prof !1 {
entry:
  %r = call i32 @mixed_callee(i32 %x), !prof !2
  ret i32 %r
}

define internal i32 @vp_only_callee(i32 %x) !prof !0 {
entry:
  ret i32 %x
}

define i32 @vp_only_caller(i32 %x) !prof !1 {
entry:
  %r = call i32 @vp_only_callee(i32 %x), !prof !3
  ret i32 %r
}

define internal i32 @expected_only_callee(i32 %x) !prof !0 {
entry:
  ret i32 %x
}

define i32 @expected_only_caller(i32 %x) !prof !1 {
entry:
  %r = call i32 @expected_only_callee(i32 %x), !prof !4
  ret i32 %r
}

!0 = !{!"function_entry_count", i64 1}
!1 = !{!"function_entry_count", i64 1}
!2 = !{!"branch_weights", i32 10}
!3 = !{!"VP", i32 0, i64 100, i64 999, i64 100}
!4 = !{!"branch_weights", !"expected", i32 50}

!llvm.module.flags = !{!10}
!10 = !{i32 1, !"ProfileSummary", !11}
!11 = !{!12, !13, !14, !15, !16, !17, !18, !19}
!12 = !{!"ProfileFormat", !"InstrProf"}
!13 = !{!"TotalCount", i64 12}
!14 = !{!"MaxCount", i64 10}
!15 = !{!"MaxInternalCount", i64 10}
!16 = !{!"MaxFunctionCount", i64 10}
!17 = !{!"NumCounts", i64 8}
!18 = !{!"NumFunctions", i64 8}
!19 = !{!"DetailedSummary", !20}
!20 = !{!21}
!21 = !{i32 10000, i64 10, i32 1}
