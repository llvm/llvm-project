; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu \
; RUN:     -passes='loop-unroll,verify-pgo-flow' -unroll-force-peel-count=2 \
; RUN:     -verify-pgo-flow-funcs=peel_expect \
; RUN:     -verify-pgo-flow-print-diagnostics=false -S \
; RUN:   | FileCheck %s --check-prefix=PEEL-IR --implicit-check-not=approxprofile
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu \
; RUN:     -passes='loop-unroll,verify-pgo-flow' -unroll-force-peel-count=2 \
; RUN:     -verify-pgo-flow-funcs=peel_expect -disable-output 2>&1 \
; RUN:   | FileCheck %s --check-prefix=PEEL
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu \
; RUN:     -passes='loop-unroll,verify-pgo-flow' -unroll-runtime -unroll-count=2 \
; RUN:     -verify-pgo-flow-funcs=runtime_expect \
; RUN:     -verify-pgo-flow-print-diagnostics=false -S \
; RUN:   | FileCheck %s --check-prefix=RT-IR --implicit-check-not=approxprofile
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu \
; RUN:     -passes='loop-unroll,verify-pgo-flow' -unroll-runtime -unroll-count=2 \
; RUN:     -verify-pgo-flow-funcs=runtime_expect -disable-output 2>&1 \
; RUN:   | FileCheck %s --check-prefix=RT
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -passes=loop-unroll \
; RUN:     -scev-cheap-expansion-budget=3 -S \
; RUN:   | FileCheck %s --check-prefix=PEEL-LAST-IR \
; RUN:       --implicit-check-not='!{!"branch_weights", i32' \
; RUN:       --implicit-check-not=approxprofile
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -passes=loop-unroll \
; RUN:     -unroll-runtime -unroll-runtime-epilog=false -unroll-count=2 -S \
; RUN:   | FileCheck %s --check-prefix=RT-PROLOG-IR \
; RUN:       --implicit-check-not='!{!"branch_weights", i32' \
; RUN:       --implicit-check-not=approxprofile
;
; llvm.expect is a hint, not a count. Peel-first and unroll keep !"expected".
; Peel-last's BTC guard is new CFG, so expect becomes unknown. Never approxprofile.

; PEEL: *** PGO Flow Verification After verify-pgo-flow ***{{$}}
; PEEL-NOT: PGOFlowVerify[BlockFrequencyMismatch]
; PEEL-NOT: PGOFlowVerify[ApproxProfileSkip]
; PEEL-NOT: PGOFlowVerify[EntryCountMismatch]

; PEEL-IR-LABEL: define void @peel_expect(i32 %n)
; PEEL-IR-NOT: define void @peel_expect{{.*}}#
; PEEL-IR: !{{[0-9]+}} = !{!"branch_weights", !"expected",

; RT: *** PGO Flow Verification After verify-pgo-flow ***{{$}}
; RT-NOT: PGOFlowVerify[BlockFrequencyMismatch]
; RT-NOT: PGOFlowVerify[ApproxProfileSkip]
; RT-NOT: PGOFlowVerify[EntryCountMismatch]

; RT-IR-LABEL: define void @runtime_expect(i32 %n)
; RT-IR-NOT: define void @runtime_expect{{.*}}#
; RT-IR: !{{[0-9]+}} = !{!"branch_weights", !"expected",

; PEEL-LAST-IR-LABEL: define i32 @peel_last_expect(
; PEEL-LAST-IR: br i1 {{.*}}, label %{{.*}}, label %exit.peel.begin, !prof
; PEEL-LAST-IR: exit.peel.begin:
; PEEL-LAST-IR-NOT: !{!"branch_weights", i32
; PEEL-LAST-IR: !{!"branch_weights", !"expected",
; PEEL-LAST-IR: !{!"unknown", !"loop-peel"}

; RT-PROLOG-IR-LABEL: define void @runtime_expect(
; RT-PROLOG-IR: do.body.prol:
; RT-PROLOG-IR-NOT: !{!"branch_weights", i32
; RT-PROLOG-IR: !{!"branch_weights", !"expected",

declare void @f(i32)

define void @peel_expect(i32 %n) !prof !0 {
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

define void @runtime_expect(i32 %n) !prof !0 {
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

define i32 @peel_last_expect(i32 %start, i32 %end) !prof !0 {
entry:
  %sub = add i32 %end, -1
  br label %loop.header

loop.header:
  %iv = phi i32 [ %start, %entry ], [ %iv.next, %loop.latch ]
  %c = icmp eq i32 %iv, %sub
  br i1 %c, label %then, label %loop.latch, !prof !2

then:
  br label %loop.latch

loop.latch:
  %iv.next = add nsw i32 %iv, 1
  %ec = icmp eq i32 %iv.next, %end
  br i1 %ec, label %exit, label %loop.header, !prof !3

exit:
  ret i32 0
}

!0 = !{!"function_entry_count", i64 10}
!1 = !{!"branch_weights", !"expected", i32 1, i32 2000}
!2 = !{!"branch_weights", !"expected", i32 2, i32 3}
!3 = !{!"branch_weights", !"expected", i32 1, i32 50}

!llvm.module.flags = !{!10}
!10 = !{i32 1, !"ProfileSummary", !11}
!11 = !{!12, !13, !14, !15, !16, !17, !18, !19}
!12 = !{!"ProfileFormat", !"InstrProf"}
!13 = !{!"TotalCount", i64 20}
!14 = !{!"MaxCount", i64 10}
!15 = !{!"MaxInternalCount", i64 10}
!16 = !{!"MaxFunctionCount", i64 10}
!17 = !{!"NumCounts", i64 4}
!18 = !{!"NumFunctions", i64 2}
!19 = !{!"DetailedSummary", !20}
!20 = !{!21}
!21 = !{i32 10000, i64 10, i32 1}
