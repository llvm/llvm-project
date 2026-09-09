; RUN: opt < %s -passes=verify-pgo-flow -disable-output 2>&1 | FileCheck %s
; RUN: opt < %s -passes=verify-pgo-flow \
; RUN:     -verify-pgo-flow-credit-indirect-callers -disable-output 2>&1 \
; RUN:   | FileCheck %s
; RUN: opt < %s -passes=verify-pgo-flow -verify-pgo-flow-aggressive \
; RUN:     -disable-output 2>&1 | FileCheck %s
;
; Leftover VP on an unreachable block is not live indirect traffic. Do not
; credit it into the callee's caller-sum (would be entry=1 vs caller-sum=101).

; CHECK: *** PGO Flow Verification After verify-pgo-flow ***{{$}}
; CHECK-NOT: PGOFlowVerify[EntryCountMismatch]

define internal i32 @target(i32 %x) !prof !0 !PGOFuncName !5 {
entry:
  ret i32 %x
}

define i32 @caller(ptr %fp, i32 %x) !prof !1 {
entry:
  %r = call i32 @target(i32 %x), !prof !2
  ret i32 %r

dead:
  %d = call i32 %fp(i32 %x), !prof !3
  ret i32 %d
}

!0 = !{!"function_entry_count", i64 1}
!1 = !{!"function_entry_count", i64 1}
!2 = !{!"branch_weights", i32 1}
; GUID of "target" is 15699497730709368386
!3 = !{!"VP", i32 0, i64 100, i64 15699497730709368386, i64 100}
!5 = !{!"target"}

!llvm.module.flags = !{!10}
!10 = !{i32 1, !"ProfileSummary", !11}
!11 = !{!12, !13, !14, !15, !16, !17, !18, !19}
!12 = !{!"ProfileFormat", !"InstrProf"}
!13 = !{!"TotalCount", i64 2}
!14 = !{!"MaxCount", i64 1}
!15 = !{!"MaxInternalCount", i64 1}
!16 = !{!"MaxFunctionCount", i64 1}
!17 = !{!"NumCounts", i64 3}
!18 = !{!"NumFunctions", i64 2}
!19 = !{!"DetailedSummary", !20}
!20 = !{!21}
!21 = !{i32 10000, i64 1, i32 1}
