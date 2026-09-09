; RUN: opt < %s -passes=verify-pgo-flow \
; RUN:     -verify-pgo-flow-report-entry-count-undercount \
; RUN:     -disable-output 2>&1 | FileCheck %s --check-prefix=DIRECT
; RUN: opt < %s -passes=verify-pgo-flow \
; RUN:     -verify-pgo-flow-report-entry-count-undercount \
; RUN:     -verify-pgo-flow-credit-indirect-callers -disable-output 2>&1 \
; RUN:   | FileCheck %s --check-prefix=CREDIT
; RUN: opt < %s -passes=verify-pgo-flow -verify-pgo-flow-aggressive \
; RUN:     -disable-output 2>&1 | FileCheck %s --check-prefix=CREDIT
;
; Undercount on, no credit: mixed looks short (direct 6, entry 15).
; With credit, VP 9 fills it. real_mismatch stays wrong (5+6 != 20).
; callee_mixed is internal; VP GUID is MD5 of PGOFuncName "callee_mixed".

; DIRECT: PGOFlowVerify[EntryCountMismatch] callee_mixed: entry=15 vs caller-sum=6
; DIRECT: PGOFlowVerify[EntryCountMismatch] callee_real_mismatch: entry=20 vs caller-sum=5

; CREDIT-NOT: PGOFlowVerify[EntryCountMismatch] callee_mixed:
; CREDIT: PGOFlowVerify[EntryCountMismatch] callee_real_mismatch: entry=20 vs caller-sum=11
; CREDIT-NOT: PGOFlowVerify[EntryCountMismatch] callee_mixed:

define internal i32 @callee_mixed(i32 %x) !prof !0 !PGOFuncName !5 {
entry:
  ret i32 %x
}

define i32 @caller_direct_mixed(i32 %x) !prof !1 {
entry:
  %r = call i32 @callee_mixed(i32 %x), !prof !2
  ret i32 %r
}

define i32 @caller_indirect_mixed(ptr %fp, i32 %x) !prof !3 {
entry:
  %r = call i32 %fp(i32 %x), !prof !4
  ret i32 %r
}

define i32 @callee_real_mismatch(i32 %x) !prof !10 {
entry:
  ret i32 %x
}

define i32 @caller_direct_real_mismatch(i32 %x) !prof !11 {
entry:
  %r = call i32 @callee_real_mismatch(i32 %x), !prof !12
  ret i32 %r
}

define i32 @caller_indirect_real_mismatch(ptr %fp, i32 %x) !prof !13 {
entry:
  %r = call i32 %fp(i32 %x), !prof !14
  ret i32 %r
}

!0 = !{!"function_entry_count", i64 15}
!1 = !{!"function_entry_count", i64 6}
!2 = !{!"branch_weights", i32 6}
!3 = !{!"function_entry_count", i64 9}
; GUID of "callee_mixed" is 5958130667041295651
!4 = !{!"VP", i32 0, i64 9, i64 5958130667041295651, i64 9}
!5 = !{!"callee_mixed"}

!10 = !{!"function_entry_count", i64 20}
!11 = !{!"function_entry_count", i64 5}
!12 = !{!"branch_weights", i32 5}
!13 = !{!"function_entry_count", i64 6}
; GUID of "callee_real_mismatch" is 13528617892774658545
!14 = !{!"VP", i32 0, i64 6, i64 13528617892774658545, i64 6}

!llvm.module.flags = !{!100}
!100 = !{i32 1, !"ProfileSummary", !101}
!101 = !{!102, !103, !104, !105, !106, !107, !108, !109}
!102 = !{!"ProfileFormat", !"InstrProf"}
!103 = !{!"TotalCount", i64 50}
!104 = !{!"MaxCount", i64 20}
!105 = !{!"MaxInternalCount", i64 20}
!106 = !{!"MaxFunctionCount", i64 20}
!107 = !{!"NumCounts", i64 8}
!108 = !{!"NumFunctions", i64 6}
!109 = !{!"DetailedSummary", !110}
!110 = !{!111}
!111 = !{i32 10000, i64 20, i32 1}
