; RUN: opt < %s -passes=verify-pgo-flow -disable-output 2>&1 \
; RUN:   | FileCheck %s --check-prefix=DEFAULT
; RUN: opt < %s -passes=verify-pgo-flow \
; RUN:     -verify-pgo-flow-report-entry-count-undercount \
; RUN:     -verify-pgo-flow-report-recursive-entry-count-mismatch \
; RUN:     -disable-output 2>&1 | FileCheck %s --check-prefix=REPORTED
; RUN: opt < %s -passes=verify-pgo-flow -verify-pgo-flow-aggressive \
; RUN:     -disable-output 2>&1 | FileCheck %s --check-prefix=REPORTED
;
; The self-call is count-type !prof 3 vs entry 10. Silent by default;
; reported only with the recursive/undercount opt-ins.
; A dead leftover self-call must not hide a live external overcount.
; A live self-call that overcounts is reported without the recursive opt-in.

; DEFAULT: PGOFlowVerify[EntryCountMismatch] leftover_rec: entry=1 vs caller-sum=10
; DEFAULT: PGOFlowVerify[EntryCountMismatch] rec_over: entry=1 vs caller-sum=10
; DEFAULT-NOT: PGOFlowVerify[EntryCountMismatch] rec:

; REPORTED: PGOFlowVerify[EntryCountMismatch] rec: entry=10 vs caller-sum=3

define i32 @rec(i32 %n) !prof !0 {
entry:
  %cond = icmp sgt i32 %n, 0
  br i1 %cond, label %recurse, label %base, !prof !1

recurse:
  %n1 = sub nsw i32 %n, 1
  %r = call i32 @rec(i32 %n1), !prof !2
  ret i32 %r

base:
  ret i32 0
}

define internal i32 @leftover_rec(i32 %x) !prof !3 {
entry:
  ret i32 %x

dead:
  %d = call i32 @leftover_rec(i32 %x), !prof !4
  ret i32 %d
}

define i32 @leftover_rec_caller(i32 %x) !prof !3 {
entry:
  %r = call i32 @leftover_rec(i32 %x), !prof !4
  ret i32 %r
}

define i32 @rec_over(i32 %n) !prof !3 {
entry:
  %r = call i32 @rec_over(i32 %n), !prof !4
  ret i32 %r
}

!0 = !{!"function_entry_count", i64 10}
!1 = !{!"branch_weights", i32 3, i32 7}
!2 = !{!"branch_weights", i32 3}
!3 = !{!"function_entry_count", i64 1}
!4 = !{!"branch_weights", i32 10}

!llvm.module.flags = !{!10}
!10 = !{i32 1, !"ProfileSummary", !11}
!11 = !{!12, !13, !14, !15, !16, !17, !18, !19}
!12 = !{!"ProfileFormat", !"InstrProf"}
!13 = !{!"TotalCount", i64 10}
!14 = !{!"MaxCount", i64 10}
!15 = !{!"MaxInternalCount", i64 10}
!16 = !{!"MaxFunctionCount", i64 10}
!17 = !{!"NumCounts", i64 3}
!18 = !{!"NumFunctions", i64 1}
!19 = !{!"DetailedSummary", !20}
!20 = !{!21}
!21 = !{i32 10000, i64 10, i32 1}
