; RUN: opt < %s -passes=verify-pgo-flow -disable-output 2>&1 \
; RUN:   | FileCheck %s --check-prefix=PASS
; RUN: opt < %s -passes=break-pgo-flow-branch-weights -verify-pgo-flow \
; RUN:     -disable-output 2>&1 | FileCheck %s --check-prefix=HOOK
; RUN: opt < %s -passes='break-pgo-flow-branch-weights,verify-pgo-flow' \
; RUN:     -disable-output 2>&1 | FileCheck %s --check-prefix=PIPE
; RUN: not opt < %s -passes=verify-pgo-flow -verify-pgo-flow-fatal \
; RUN:     -disable-output 2>&1 | FileCheck %s --check-prefix=FATAL
;
; @ok is consistent; @bad is not. The standalone pass reports only @bad.
; break-pgo-flow-branch-weights corrupts weights. HOOK uses the post-pass
; `-verify-pgo-flow` flag; PIPE puts `verify-pgo-flow` in the pipeline
; instead (no `-verify-pgo-flow`). Needs an InstrProf ProfileSummary.

; PASS: *** PGO Flow Verification After verify-pgo-flow ***{{$}}
; PASS-NOT: PGOFlowVerify[BlockFrequencyMismatch] ok:
; PASS: PGOFlowVerify[BlockFrequencyMismatch] bad: block entry: incoming=10 vs outgoing=9
; PASS-NOT: PGOFlowVerify[BlockFrequencyMismatch] ok:

; HOOK: *** PGO Flow Verification After BreakPGOFlowBranchWeightsPass ***{{$}}
; HOOK: PGOFlowVerify[BlockFrequencyMismatch] ok: block entry: incoming=10 vs outgoing=9
; HOOK: *** PGO Flow Verification After BreakPGOFlowBranchWeightsPass ***
; HOOK: PGOFlowVerify[BlockFrequencyMismatch] bad: block entry: incoming=10 vs outgoing=8

; PIPE-NOT: BreakPGOFlowBranchWeightsPass
; PIPE: *** PGO Flow Verification After verify-pgo-flow ***{{$}}
; PIPE: PGOFlowVerify[BlockFrequencyMismatch] ok: block entry: incoming=10 vs outgoing=9
; PIPE: PGOFlowVerify[BlockFrequencyMismatch] bad: block entry: incoming=10 vs outgoing=8
; PIPE-NOT: *** PGO Flow Verification After BreakPGOFlowBranchWeightsPass

; FATAL: PGOFlowVerify[BlockFrequencyMismatch]

define i32 @ok(i32 %x) !prof !0 {
entry:
  %c = icmp sgt i32 %x, 0
  br i1 %c, label %then, label %else, !prof !1

then:
  ret i32 1

else:
  ret i32 0
}

define i32 @bad(i32 %x) !prof !0 {
entry:
  %c = icmp sgt i32 %x, 0
  br i1 %c, label %then, label %else, !prof !2

then:
  ret i32 1

else:
  ret i32 0
}

!0 = !{!"function_entry_count", i64 10}
!1 = !{!"branch_weights", i32 7, i32 3}
!2 = !{!"branch_weights", i32 7, i32 2}

!llvm.module.flags = !{!10}
!10 = !{i32 1, !"ProfileSummary", !11}
!11 = !{!12, !13, !14, !15, !16, !17, !18, !19}
!12 = !{!"ProfileFormat", !"InstrProf"}
!13 = !{!"TotalCount", i64 10}
!14 = !{!"MaxCount", i64 10}
!15 = !{!"MaxInternalCount", i64 7}
!16 = !{!"MaxFunctionCount", i64 10}
!17 = !{!"NumCounts", i64 2}
!18 = !{!"NumFunctions", i64 2}
!19 = !{!"DetailedSummary", !20}
!20 = !{!21}
!21 = !{i32 10000, i64 10, i32 1}
