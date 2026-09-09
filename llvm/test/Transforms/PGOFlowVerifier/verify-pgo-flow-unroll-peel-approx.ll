; RUN: opt < %s -passes='loop-unroll,verify-pgo-flow' -unroll-force-peel-count=2 \
; RUN:     -verify-pgo-flow-funcs=peel_loop \
; RUN:     -verify-pgo-flow-print-diagnostics=false -S \
; RUN:   | FileCheck %s --check-prefix=PEEL-IR
; RUN: opt < %s -passes='loop-unroll,verify-pgo-flow' -unroll-force-peel-count=2 \
; RUN:     -verify-pgo-flow-funcs=peel_loop -disable-output 2>&1 \
; RUN:   | FileCheck %s --check-prefix=PEEL
; RUN: opt < %s -passes='loop-unroll,verify-pgo-flow' -unroll-runtime -unroll-count=2 \
; RUN:     -verify-pgo-flow-funcs=runtime_loop \
; RUN:     -verify-pgo-flow-print-diagnostics=false -S \
; RUN:   | FileCheck %s --check-prefix=RT-IR
; RUN: opt < %s -passes='loop-unroll,verify-pgo-flow' -unroll-runtime -unroll-count=2 \
; RUN:     -verify-pgo-flow-funcs=runtime_loop -disable-output 2>&1 \
; RUN:   | FileCheck %s --check-prefix=RT
; RUN: opt < %s -passes='function(loop-unroll),verify-pgo-flow' \
; RUN:     -unroll-force-peel-count=2 \
; RUN:     -verify-pgo-flow-funcs=peel_callee -disable-output 2>&1 \
; RUN:   | FileCheck %s --check-prefix=CALLEE-PEEL
; RUN: opt < %s -passes='function(loop-unroll),verify-pgo-flow' \
; RUN:     -unroll-runtime -unroll-count=2 \
; RUN:     -verify-pgo-flow-funcs=peel_callee -disable-output 2>&1 \
; RUN:   | FileCheck %s --check-prefix=CALLEE-RT
; RUN: opt < %s -passes=loop-unroll -unroll-force-peel-count=2 -S \
; RUN:   | FileCheck %s --check-prefix=CALL-IR
; RUN: opt < %s -passes='loop-unroll,verify-pgo-flow' -unroll-force-peel-count=2 \
; RUN:     -verify-pgo-flow-funcs=peel_expect_vp -disable-output 2>&1 \
; RUN:   | FileCheck %s --check-prefix=VP
;
; Peel-first clones count-type latch weights and stamps approxprofile.
; Runtime unroll writes probabilities and stamps. Expect latches do not stamp.
; Caller-sum runs on the module verifier after function(loop-unroll), not on a
; function-unit verify-pgo-flow.

declare void @f(i32)

; PEEL: *** PGO Flow Verification After verify-pgo-flow ***{{$}}
; PEEL: PGOFlowVerify[ApproxProfileSkip] peel_loop: skipping strict InstrProf verification (approxprofile)
; PEEL-NOT: PGOFlowVerify[BlockFrequencyMismatch]
; PEEL-NOT: PGOFlowVerify[EntryCountMismatch]

; PEEL-IR-LABEL: define void @peel_loop(
; PEEL-IR-SAME: #[[PEELATTR:[0-9]+]]
; PEEL-IR: attributes #[[PEELATTR]] = { approxprofile }

define void @peel_loop(i32 %n) !prof !0 {
entry:
  br label %do.body

do.body:
  %i = phi i32 [ 0, %entry ], [ %inc, %do.body ]
  %inc = add i32 %i, 1
  call void @f(i32 %i)
  %c = icmp sge i32 %inc, %n
  br i1 %c, label %do.end, label %do.body, !prof !1

do.end:
  ret void
}

; RT: *** PGO Flow Verification After verify-pgo-flow ***{{$}}
; RT: PGOFlowVerify[ApproxProfileSkip] runtime_loop: skipping strict InstrProf verification (approxprofile)
; RT-NOT: PGOFlowVerify[BlockFrequencyMismatch]
; RT-NOT: PGOFlowVerify[EntryCountMismatch]

; RT-IR-LABEL: define void @runtime_loop(
; RT-IR-SAME: #[[RTATTR:[0-9]+]]
; RT-IR: attributes #[[RTATTR]] = { approxprofile }

define void @runtime_loop(i32 %n) !prof !0 {
entry:
  br label %do.body

do.body:
  %i = phi i32 [ 0, %entry ], [ %inc, %do.body ]
  %inc = add i32 %i, 1
  call void @f(i32 %i)
  %c = icmp sge i32 %inc, %n
  br i1 %c, label %do.end, label %do.body, !prof !1

do.end:
  ret void
}

; CALLEE-PEEL: *** PGO Flow Verification After verify-pgo-flow ***{{$}}
; CALLEE-PEEL-NOT: PGOFlowVerify[EntryCountMismatch] peel_callee:

; CALLEE-RT: *** PGO Flow Verification After verify-pgo-flow ***{{$}}
; CALLEE-RT-NOT: PGOFlowVerify[EntryCountMismatch] peel_callee:

define internal void @peel_callee(i32 %x) !prof !30 {
entry:
  ret void
}

define void @peel_callee_loop(i32 %n) !prof !0 {
entry:
  br label %do.body

do.body:
  %i = phi i32 [ 0, %entry ], [ %inc, %do.body ]
  %inc = add i32 %i, 1
  call void @peel_callee(i32 %i), !prof !31
  %c = icmp sge i32 %inc, %n
  br i1 %c, label %do.end, label %do.body, !prof !1

do.end:
  ret void
}

; CALL-IR-LABEL: define void @peel_call_count(i32 %n)
; CALL-IR-NOT: define void @peel_call_count{{.*}}#

; VP: *** PGO Flow Verification After verify-pgo-flow ***{{$}}
; VP-NOT: PGOFlowVerify[ApproxProfileSkip] peel_expect_vp:
; VP-NOT: PGOFlowVerify[EntryCountMismatch]

define void @peel_call_count(i32 %n) !prof !0 {
entry:
  br label %do.body

do.body:
  %i = phi i32 [ 0, %entry ], [ %inc, %do.body ]
  %inc = add i32 %i, 1
  call void @f(i32 %i), !prof !31
  %c = icmp sge i32 %inc, %n
  br i1 %c, label %do.end, label %do.body, !prof !40

do.end:
  ret void
}

define void @peel_expect_vp(i32 %n) !prof !0 {
entry:
  br label %do.body

do.body:
  %i = phi i32 [ 0, %entry ], [ %inc, %do.body ]
  %inc = add i32 %i, 1
  call void @f(i32 %i), !prof !41
  %c = icmp sge i32 %inc, %n
  br i1 %c, label %do.end, label %do.body, !prof !40

do.end:
  ret void
}

define void @runtime_callee_loop(i32 %n) !prof !0 {
entry:
  br label %do.body

do.body:
  %i = phi i32 [ 0, %entry ], [ %inc, %do.body ]
  %inc = add i32 %i, 1
  call void @peel_callee(i32 %i), !prof !31
  %c = icmp sge i32 %inc, %n
  br i1 %c, label %do.end, label %do.body, !prof !1

do.end:
  ret void
}

!0 = !{!"function_entry_count", i64 10}
!1 = !{!"branch_weights", i32 1, i32 9}
!30 = !{!"function_entry_count", i64 1}
!31 = !{!"branch_weights", i32 5}
!40 = !{!"branch_weights", !"expected", i32 1, i32 9}
!41 = !{!"VP", i32 0, i64 10, i64 123, i64 10}

!llvm.module.flags = !{!10}
!10 = !{i32 1, !"ProfileSummary", !11}
!11 = !{!12, !13, !14, !15, !16, !17, !18, !19}
!12 = !{!"ProfileFormat", !"InstrProf"}
!13 = !{!"TotalCount", i64 20}
!14 = !{!"MaxCount", i64 10}
!15 = !{!"MaxInternalCount", i64 9}
!16 = !{!"MaxFunctionCount", i64 10}
!17 = !{!"NumCounts", i64 4}
!18 = !{!"NumFunctions", i64 2}
!19 = !{!"DetailedSummary", !20}
!20 = !{!21}
!21 = !{i32 10000, i64 10, i32 1}
