; RUN: opt < %s -passes=verify-pgo-flow -verify-pgo-flow-funcs=ok_callee \
; RUN:     -disable-output 2>&1 | FileCheck %s --check-prefix=OK
; RUN: opt < %s -passes=verify-pgo-flow -verify-pgo-flow-funcs=bad_callee \
; RUN:     -disable-output 2>&1 | FileCheck %s --check-prefix=BAD
; RUN: opt < %s -passes=verify-pgo-flow \
; RUN:     -verify-pgo-flow-funcs=ok_callee,bad_callee -disable-output 2>&1 \
; RUN:   | FileCheck %s --check-prefix=BOTH
;
; Empty -verify-pgo-flow-funcs checks every function. A non-empty list is an
; allow-list. Callers are still used to sum weights when the callee is listed.

; OK: *** PGO Flow Verification After verify-pgo-flow ***{{$}}
; OK-NOT: PGOFlowVerify[

; BAD: *** PGO Flow Verification After verify-pgo-flow ***{{$}}
; BAD-NOT: PGOFlowVerify[EntryCountMismatch] ok_callee:
; BAD: PGOFlowVerify[EntryCountMismatch] bad_callee: entry=1 vs caller-sum=10
; BAD-NOT: PGOFlowVerify[EntryCountMismatch] ok_callee:

; BOTH: *** PGO Flow Verification After verify-pgo-flow ***{{$}}
; BOTH-NOT: PGOFlowVerify[EntryCountMismatch] ok_callee:
; BOTH: PGOFlowVerify[EntryCountMismatch] bad_callee: entry=1 vs caller-sum=10
; BOTH-NOT: PGOFlowVerify[EntryCountMismatch] ok_callee:

define internal i32 @ok_callee(i32 %x) !prof !0 {
entry:
  ret i32 %x
}

define i32 @ok_caller(i32 %x) !prof !1 {
entry:
  %r = call i32 @ok_callee(i32 %x), !prof !2
  ret i32 %r
}

define internal i32 @bad_callee(i32 %x) !prof !0 {
entry:
  ret i32 %x
}

define i32 @bad_caller(i32 %x) !prof !1 {
entry:
  %r = call i32 @bad_callee(i32 %x), !prof !3
  ret i32 %r
}

!0 = !{!"function_entry_count", i64 1}
!1 = !{!"function_entry_count", i64 1}
!2 = !{!"branch_weights", i32 1}
!3 = !{!"branch_weights", i32 10}

!llvm.module.flags = !{!10}
!10 = !{i32 1, !"ProfileSummary", !11}
!11 = !{!12, !13, !14, !15, !16, !17, !18, !19}
!12 = !{!"ProfileFormat", !"InstrProf"}
!13 = !{!"TotalCount", i64 13}
!14 = !{!"MaxCount", i64 10}
!15 = !{!"MaxInternalCount", i64 10}
!16 = !{!"MaxFunctionCount", i64 1}
!17 = !{!"NumCounts", i64 4}
!18 = !{!"NumFunctions", i64 4}
!19 = !{!"DetailedSummary", !20}
!20 = !{!21}
!21 = !{i32 10000, i64 10, i32 1}
