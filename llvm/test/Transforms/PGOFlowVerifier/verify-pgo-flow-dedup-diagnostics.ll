; RUN: opt < %s -verify-pgo-flow -passes='strip-dead-prototypes,globaldce' \
; RUN:     -disable-output 2>&1 | FileCheck %s
; RUN: opt < %s -verify-pgo-flow -verify-pgo-flow-dedup-diagnostics=false \
; RUN:     -passes='strip-dead-prototypes,globaldce' -disable-output 2>&1 \
; RUN:   | FileCheck %s --check-prefix=REPEAT
;
; StripDeadPrototypes drops @dead_decl and GlobalDCE drops @dead_global, so
; both module walks see a changed module and re-check @bad_callee's entry
; count. Dedup (default) prints EntryCountMismatch once, disabling it prints
; the same mismatch after every changing module walk.

; CHECK: *** PGO Flow Verification After StripDeadPrototypesPass ***{{$}}
; CHECK: PGOFlowVerify[EntryCountMismatch] bad_callee: entry=1 vs caller-sum=10
; CHECK: *** PGO Flow Verification After GlobalDCEPass ***{{$}}
; CHECK-NOT: PGOFlowVerify[EntryCountMismatch] bad_callee:

; REPEAT: *** PGO Flow Verification After StripDeadPrototypesPass ***{{$}}
; REPEAT: PGOFlowVerify[EntryCountMismatch] bad_callee: entry=1 vs caller-sum=10
; REPEAT: *** PGO Flow Verification After GlobalDCEPass ***{{$}}
; REPEAT: PGOFlowVerify[EntryCountMismatch] bad_callee: entry=1 vs caller-sum=10
; REPEAT-NOT: PGOFlowVerify[EntryCountMismatch] bad_callee:

define internal i32 @bad_callee(i32 %x) !prof !0 {
entry:
  %y = add i32 %x, 0
  ret i32 %y
}

define i32 @bad_caller(i32 %x) !prof !1 {
entry:
  %r = call i32 @bad_callee(i32 %x), !prof !2
  ret i32 %r
}

@dead_global = internal global i32 42

declare void @dead_decl()

!0 = !{!"function_entry_count", i64 1}
!1 = !{!"function_entry_count", i64 1}
!2 = !{!"branch_weights", i32 10}

!llvm.module.flags = !{!10}
!10 = !{i32 1, !"ProfileSummary", !11}
!11 = !{!12, !13, !14, !15, !16, !17, !18, !19}
!12 = !{!"ProfileFormat", !"InstrProf"}
!13 = !{!"TotalCount", i64 12}
!14 = !{!"MaxCount", i64 10}
!15 = !{!"MaxInternalCount", i64 10}
!16 = !{!"MaxFunctionCount", i64 10}
!17 = !{!"NumCounts", i64 3}
!18 = !{!"NumFunctions", i64 2}
!19 = !{!"DetailedSummary", !20}
!20 = !{!21}
!21 = !{i32 10000, i64 10, i32 1}
