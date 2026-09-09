; RUN: opt < %s -passes=verify-pgo-flow -disable-output 2>&1 \
; RUN:   | FileCheck %s --check-prefix=DIAG
; RUN: opt < %s -passes=verify-pgo-flow -verify-pgo-flow-fatal \
; RUN:     -verify-pgo-flow-funcs=approx_bad -disable-output 2>&1 \
; RUN:   | FileCheck %s --check-prefix=SKIP-FATAL
;
; approxprofile means counts may be scaled; do not report a hard mismatch.
; Skip notes must not abort under -verify-pgo-flow-fatal.

; SKIP-FATAL: PGOFlowVerify[ApproxProfileSkip] approx_bad:
; SKIP-FATAL-NOT: PGOFlowVerify[BlockFrequencyMismatch]

; DIAG: *** PGO Flow Verification After verify-pgo-flow ***{{$}}
; DIAG: PGOFlowVerify[ApproxProfileSkip] approx_bad: skipping strict InstrProf verification (approxprofile)
; DIAG-NOT: PGOFlowVerify[BlockFrequencyMismatch] approx_bad:
; DIAG: PGOFlowVerify[BlockFrequencyMismatch] strict_bad: block entry: incoming=10 vs outgoing=9
; DIAG-NOT: PGOFlowVerify[BlockFrequencyMismatch] approx_bad:
; DIAG-NOT: PGOFlowVerify[ApproxProfileSkip] strict_bad:

define i32 @approx_bad(i32 %x) approxprofile !prof !0 {
entry:
  %c = icmp sgt i32 %x, 0
  br i1 %c, label %then, label %else, !prof !2

then:
  ret i32 1

else:
  ret i32 0
}

define i32 @strict_bad(i32 %x) !prof !0 {
entry:
  %c = icmp sgt i32 %x, 0
  br i1 %c, label %then, label %else, !prof !2

then:
  ret i32 1

else:
  ret i32 0
}

!0 = !{!"function_entry_count", i64 10}
!2 = !{!"branch_weights", i32 7, i32 2}

!llvm.module.flags = !{!10}
!10 = !{i32 1, !"ProfileSummary", !11}
!11 = !{!12, !13, !14, !15, !16, !17, !18, !19}
!12 = !{!"ProfileFormat", !"InstrProf"}
!13 = !{!"TotalCount", i64 20}
!14 = !{!"MaxCount", i64 10}
!15 = !{!"MaxInternalCount", i64 7}
!16 = !{!"MaxFunctionCount", i64 10}
!17 = !{!"NumCounts", i64 4}
!18 = !{!"NumFunctions", i64 2}
!19 = !{!"DetailedSummary", !20}
!20 = !{!21}
!21 = !{i32 10000, i64 10, i32 1}
