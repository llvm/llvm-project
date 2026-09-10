; RUN: opt < %s -passes=verify-pgo-flow -disable-output 2>&1 | FileCheck %s
; RUN: not opt < %s -passes=verify-pgo-flow -verify-pgo-flow-fatal \
; RUN:     -disable-output 2>&1 | FileCheck %s --check-prefix=FATAL
;
; Unreachable blocks can keep stale count-type branch_weights after CFG
; edits. Those weights are not live flow, do not report BlockFrequencyMismatch.
; Unreachable SCCs (every block has pred_size > 0) and leftover weights on a
; zero function_entry_count terminator are the same class of leftover MD.
;
; @order: a 0-weight successor listed after a child that still has leftover
; !prof. Do not apply that MD in raw BB-list order.
; @dead_pred: %live has preds {entry, %dead}, %dead is unreachable. Releasing
; the dead pred must still conservation-check %live.

; CHECK: *** PGO Flow Verification After verify-pgo-flow ***{{$}}
; CHECK-NOT: PGOFlowVerify[BlockFrequencyMismatch] leftover:
; CHECK-NOT: PGOFlowVerify[BlockFrequencyMismatch] leftover_cycle:
; CHECK-NOT: PGOFlowVerify[BlockFrequencyMismatch] zero_entry:
; CHECK-NOT: PGOFlowVerify[BlockFrequencyMismatch] order:
; CHECK-NOT: PGOFlowVerify[BlockFrequencyMismatch] dead_pred:
; CHECK: PGOFlowVerify[BlockFrequencyMismatch] dead_pred_bad: block live: incoming=10 vs outgoing=9

; FATAL: PGOFlowVerify[BlockFrequencyMismatch] dead_pred_bad:

define i32 @leftover(i32 %x) !prof !0 {
entry:
  ret i32 0

dead:
  %c = icmp sgt i32 %x, 0
  br i1 %c, label %then, label %else, !prof !1

then:
  ret i32 1

else:
  %c2 = icmp eq i32 %x, 0
  br i1 %c2, label %join, label %other, !prof !2

other:
  ret i32 3

join:
  ret i32 2
}

define i32 @leftover_cycle(i32 %x) !prof !0 {
entry:
  ret i32 0

dead1:
  %c = icmp sgt i32 %x, 0
  br i1 %c, label %dead2, label %dead1, !prof !1

dead2:
  br label %dead1
}

define void @zero_entry(i1 %c) !prof !3 {
entry:
  br i1 %c, label %a, label %b, !prof !1

a:
  ret void

b:
  ret void
}

define i32 @order(i1 %c) !prof !0 {
entry:
  br i1 %c, label %live, label %zero, !prof !4

child:
  br i1 %c, label %child.a, label %child.b, !prof !1

child.a:
  ret i32 1

child.b:
  ret i32 2

zero:
  br i1 %c, label %child, label %child, !prof !1

live:
  ret i32 0
}

define i32 @dead_pred(i1 %c) !prof !0 {
entry:
  br label %live

dead:
  br i1 %c, label %live, label %live, !prof !1

live:
  br i1 %c, label %a, label %b, !prof !1

a:
  ret i32 0

b:
  ret i32 1
}

define i32 @dead_pred_bad(i1 %c) !prof !0 {
entry:
  br label %live

dead:
  br i1 %c, label %live, label %live, !prof !1

live:
  br i1 %c, label %a, label %b, !prof !6

a:
  ret i32 0

b:
  ret i32 1
}

!0 = !{!"function_entry_count", i64 10}
!1 = !{!"branch_weights", i32 7, i32 3}
!2 = !{!"branch_weights", i32 9, i32 1}
!3 = !{!"function_entry_count", i64 0}
!4 = !{!"branch_weights", i32 10, i32 0}
!6 = !{!"branch_weights", i32 9, i32 0}

!llvm.module.flags = !{!10}
!10 = !{i32 1, !"ProfileSummary", !11}
!11 = !{!12, !13, !14, !15, !16, !17, !18, !19}
!12 = !{!"ProfileFormat", !"InstrProf"}
!13 = !{!"TotalCount", i64 10}
!14 = !{!"MaxCount", i64 10}
!15 = !{!"MaxInternalCount", i64 7}
!16 = !{!"MaxFunctionCount", i64 10}
!17 = !{!"NumCounts", i64 2}
!18 = !{!"NumFunctions", i64 6}
!19 = !{!"DetailedSummary", !20}
!20 = !{!21}
!21 = !{i32 10000, i64 10, i32 1}
