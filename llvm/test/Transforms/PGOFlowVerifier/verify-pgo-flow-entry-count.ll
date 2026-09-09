; RUN: opt < %s -passes=verify-pgo-flow -disable-output 2>&1 \
; RUN:   | FileCheck %s --check-prefix=DIAG
; RUN: opt < %s -passes='function(verify-pgo-flow)' -disable-output 2>&1 \
; RUN:   | FileCheck %s --check-prefix=FUNC
; RUN: not opt < %s -passes=verify-pgo-flow -verify-pgo-flow-fatal \
; RUN:     -disable-output 2>&1 | FileCheck %s --check-prefix=FATAL
;
; Report only when visible direct-caller weight exceeds entry count.
; Entry-count vs caller-sum is a module walk; function(verify-pgo-flow)
; only checks in-function block flow.

; DIAG: *** PGO Flow Verification After verify-pgo-flow ***{{$}}
; DIAG-NOT: PGOFlowVerify[EntryCountMismatch] ok_callee:
; DIAG-NOT: PGOFlowVerify[EntryCountMismatch] undercount_callee:
; DIAG: PGOFlowVerify[EntryCountMismatch] bad_callee: entry=1 vs caller-sum=10
; DIAG: PGOFlowVerify[EntryCountMismatch] real_alias_callee: entry=1 vs caller-sum=10
; DIAG-NOT: PGOFlowVerify[EntryCountMismatch] ok_callee:
; DIAG-NOT: PGOFlowVerify[EntryCountMismatch] undercount_callee:

; FUNC-NOT: PGOFlowVerify[EntryCountMismatch]

; FATAL: PGOFlowVerify[EntryCountMismatch]

define internal i32 @ok_callee(i32 %x) !prof !0 {
entry:
  ret i32 %x
}

define i32 @ok_caller(i32 %x) !prof !1 {
entry:
  %r = call i32 @ok_callee(i32 %x), !prof !2
  ret i32 %r
}

define internal i32 @undercount_callee(i32 %x) !prof !3 {
entry:
  ret i32 %x
}

define i32 @undercount_caller(i32 %x) !prof !1 {
entry:
  %r = call i32 @undercount_callee(i32 %x), !prof !2
  ret i32 %r
}

define internal i32 @bad_callee(i32 %x) !prof !0 {
entry:
  ret i32 %x
}

define i32 @bad_caller(i32 %x) !prof !1 {
entry:
  %r = call i32 @bad_callee(i32 %x), !prof !4
  ret i32 %r
}

@alias_callee = internal alias i32 (i32), ptr @real_alias_callee

define internal i32 @real_alias_callee(i32 %x) !prof !0 {
entry:
  ret i32 %x
}

define i32 @alias_caller(i32 %x) !prof !1 {
entry:
  %r = call i32 @alias_callee(i32 %x), !prof !4
  ret i32 %r
}

!0 = !{!"function_entry_count", i64 1}
!1 = !{!"function_entry_count", i64 1}
!2 = !{!"branch_weights", i32 1}
!3 = !{!"function_entry_count", i64 10}
!4 = !{!"branch_weights", i32 10}

!llvm.module.flags = !{!10}
!10 = !{i32 1, !"ProfileSummary", !11}
!11 = !{!12, !13, !14, !15, !16, !17, !18, !19}
!12 = !{!"ProfileFormat", !"InstrProf"}
!13 = !{!"TotalCount", i64 14}
!14 = !{!"MaxCount", i64 10}
!15 = !{!"MaxInternalCount", i64 10}
!16 = !{!"MaxFunctionCount", i64 10}
!17 = !{!"NumCounts", i64 8}
!18 = !{!"NumFunctions", i64 8}
!19 = !{!"DetailedSummary", !20}
!20 = !{!21}
!21 = !{i32 10000, i64 10, i32 1}
