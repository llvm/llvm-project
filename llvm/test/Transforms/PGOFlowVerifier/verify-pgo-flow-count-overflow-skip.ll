; RUN: opt < %s -passes=verify-pgo-flow -disable-output 2>&1 \
; RUN:   | FileCheck %s --check-prefix=DIAG
; RUN: opt < %s -passes=verify-pgo-flow -verify-pgo-flow-fatal \
; RUN:     -verify-pgo-flow-funcs=huge_weight -disable-output 2>&1 \
; RUN:   | FileCheck %s --check-prefix=SKIP-FATAL
;
; Counts that cannot fit in uint32_t on a terminator are not compared
; strictly. A wide i64 function_entry_count is not overflow. Mismatch
; against u32 outs is still reported.
; Overflowed callers must not charge leftover call !prof onto a callee.
; Skip notes must not abort under -verify-pgo-flow-fatal.

; DIAG: *** PGO Flow Verification After verify-pgo-flow ***{{$}}
; DIAG: PGOFlowVerify[BlockFrequencyMismatch] overflow_bad: block entry: incoming=4294967296 vs outgoing=9
; DIAG-NOT: PGOFlowVerify[CountOverflowSkip] overflow_bad:
; DIAG: PGOFlowVerify[CountOverflowSkip] huge_weight: skipping strict InstrProf verification (profile count overflow)
; DIAG-NOT: PGOFlowVerify[BlockFrequencyMismatch] huge_weight:
; DIAG: PGOFlowVerify[BlockFrequencyMismatch] small_bad: block entry: incoming=10 vs outgoing=9
; DIAG: PGOFlowVerify[CountOverflowSkip] huge_loop: skipping strict InstrProf verification (profile count overflow)
; DIAG-NOT: PGOFlowVerify[BlockFrequencyMismatch] huge_loop:
; DIAG-NOT: PGOFlowVerify[BlockFrequencyMismatch] overflow_bad:
; DIAG-NOT: PGOFlowVerify[CountOverflowSkip] small_bad:
; DIAG-NOT: PGOFlowVerify[EntryCountMismatch] overflow_callee:

; SKIP-FATAL: PGOFlowVerify[CountOverflowSkip] huge_weight:
; SKIP-FATAL-NOT: PGOFlowVerify[BlockFrequencyMismatch]

define i32 @overflow_bad(i32 %x) !prof !0 {
entry:
  %c = icmp sgt i32 %x, 0
  br i1 %c, label %then, label %else, !prof !2

then:
  ret i32 1

else:
  ret i32 0
}

define internal i32 @overflow_callee(i32 %x) !prof !4 {
entry:
  ret i32 %x
}

define i32 @huge_weight(i32 %x) !prof !1 {
entry:
  %c = icmp sgt i32 %x, 0
  br i1 %c, label %then, label %else, !prof !3

then:
  %a = call i32 @overflow_callee(i32 %x), !prof !5
  ret i32 %a

else:
  %b = call i32 @overflow_callee(i32 %x), !prof !5
  ret i32 %b
}

define i32 @small_bad(i32 %x) !prof !1 {
entry:
  %c = icmp sgt i32 %x, 0
  br i1 %c, label %then, label %else, !prof !2

then:
  ret i32 1

else:
  ret i32 0
}

; Overflow on a loop header is observed even though the backedge pred is
; still unknown when the header weights are applied.
define void @huge_loop(i1 %c) !prof !1 {
entry:
  br label %header

header:
  br i1 %c, label %latch, label %exit, !prof !3

latch:
  br label %header

exit:
  ret void
}

!0 = !{!"function_entry_count", i64 4294967296}
!1 = !{!"function_entry_count", i64 10}
!2 = !{!"branch_weights", i32 7, i32 2}
!3 = !{!"branch_weights", i64 4294967296, i64 1}
!4 = !{!"function_entry_count", i64 1}
!5 = !{!"branch_weights", i32 10}

!llvm.module.flags = !{!10}
!10 = !{i32 1, !"ProfileSummary", !11}
!11 = !{!12, !13, !14, !15, !16, !17, !18, !19}
!12 = !{!"ProfileFormat", !"InstrProf"}
!13 = !{!"TotalCount", i64 4294967306}
!14 = !{!"MaxCount", i64 4294967296}
!15 = !{!"MaxInternalCount", i64 4294967296}
!16 = !{!"MaxFunctionCount", i64 4294967296}
!17 = !{!"NumCounts", i64 4}
!18 = !{!"NumFunctions", i64 5}
!19 = !{!"DetailedSummary", !20}
!20 = !{!21}
!21 = !{i32 10000, i64 4294967296, i32 1}
