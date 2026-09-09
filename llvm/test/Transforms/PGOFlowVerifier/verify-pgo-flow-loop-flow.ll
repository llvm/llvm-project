; RUN: opt < %s -passes=verify-pgo-flow -disable-output 2>&1 | FileCheck %s
; RUN: not opt < %s -passes=verify-pgo-flow -verify-pgo-flow-fatal \
; RUN:     -disable-output 2>&1 | FileCheck %s --check-prefix=FATAL
;
; Count-type branch_weights on a live loop header must be applied before the
; backedge pred closes. @ok_loop conserves (entry 10 + backedge 90 = 100).
; @bad_loop does not (outgoing 95 vs incoming 100).

; CHECK: *** PGO Flow Verification After verify-pgo-flow ***{{$}}
; CHECK-NOT: PGOFlowVerify[BlockFrequencyMismatch] ok_loop:
; CHECK: PGOFlowVerify[BlockFrequencyMismatch] bad_loop: block header: incoming=100 vs outgoing=95
; CHECK-NOT: PGOFlowVerify[BlockFrequencyMismatch] ok_loop:

; FATAL: PGOFlowVerify[BlockFrequencyMismatch] bad_loop:

define void @ok_loop(i1 %c) !prof !0 {
entry:
  br label %header

header:
  br i1 %c, label %latch, label %exit, !prof !1

latch:
  br label %header

exit:
  ret void
}

define void @bad_loop(i1 %c) !prof !0 {
entry:
  br label %header

header:
  br i1 %c, label %latch, label %exit, !prof !2

latch:
  br label %header

exit:
  ret void
}

!0 = !{!"function_entry_count", i64 10}
!1 = !{!"branch_weights", i32 90, i32 10}
!2 = !{!"branch_weights", i32 90, i32 5}

!llvm.module.flags = !{!10}
!10 = !{i32 1, !"ProfileSummary", !11}
!11 = !{!12, !13, !14, !15, !16, !17, !18, !19}
!12 = !{!"ProfileFormat", !"InstrProf"}
!13 = !{!"TotalCount", i64 10}
!14 = !{!"MaxCount", i64 90}
!15 = !{!"MaxInternalCount", i64 90}
!16 = !{!"MaxFunctionCount", i64 10}
!17 = !{!"NumCounts", i64 4}
!18 = !{!"NumFunctions", i64 2}
!19 = !{!"DetailedSummary", !20}
!20 = !{!21}
!21 = !{i32 10000, i64 90, i32 1}
