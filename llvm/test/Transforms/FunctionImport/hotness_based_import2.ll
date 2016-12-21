; Test to check that callee reached from cold and then hot path gets
; hot thresholds.
; RUN: opt -module-summary %s -o %t.bc
; RUN: opt -module-summary %p/Inputs/hotness_based_import2.ll -o %t2.bc
; RUN: llvm-lto -thinlto -o %t3 %t.bc %t2.bc

; Teset with limit set to 10 and multipliers set to 1. Since cold call to
; hot is first in the other module, we'll first add calledFromHot to worklist
; with threshold decayed by default 0.7 factor. Test ensures that when we
; encounter it again from hot path, we re-enqueue with higher non-decayed
; threshold which will allow it to be imported.
; RUN: opt -function-import -summary-file %t3.thinlto.bc %t.bc -import-instr-limit=10 -import-hot-multiplier=1.0 -import-cold-multiplier=1.0 -S | FileCheck %s --check-prefix=CHECK
; CHECK-DAG: define available_externally void @hot()
; CHECK-DAG: define available_externally void @calledFromHot()

; ModuleID = 'thinlto-function-summary-callgraph.ll'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; This function has a high profile count, so entry block is hot.
define void @hot_function(i1 %a, i1 %a2) !prof !28 {
entry:
  call void @hot()
  ret void
}

; This function has a low profile count, so entry block is hot.
define void @cold_function(i1 %a, i1 %a2) !prof !29 {
entry:
  call void @hot()
  ret void
}

declare void @hot() #1

!llvm.module.flags = !{!1}

!1 = !{i32 1, !"ProfileSummary", !2}
!2 = !{!3, !4, !5, !6, !7, !8, !9, !10}
!3 = !{!"ProfileFormat", !"InstrProf"}
!4 = !{!"TotalCount", i64 222}
!5 = !{!"MaxCount", i64 110}
!6 = !{!"MaxInternalCount", i64 1}
!7 = !{!"MaxFunctionCount", i64 110}
!8 = !{!"NumCounts", i64 4}
!9 = !{!"NumFunctions", i64 3}
!10 = !{!"DetailedSummary", !11}
!11 = !{!12, !13, !14}
!12 = !{i32 10000, i64 110, i32 2}
!13 = !{i32 999000, i64 2, i32 4}
!14 = !{i32 999999, i64 2, i32 4}
!28 = !{!"function_entry_count", i64 110}
!29 = !{!"function_entry_count", i64 1}
