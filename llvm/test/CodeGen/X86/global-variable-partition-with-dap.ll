target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

;; A minimal test case. Subsequent PRs will expand on this test case
;; (e.g., with more functions, variables and profiles) and test the hotness
;; reconcillation implementation.
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -relocation-model=pic \
; RUN:     -partition-static-data-sections=true \
; RUN:     -data-sections=true  -unique-section-names=false \
; RUN:     %s -o - 2>&1 | FileCheck %s --check-prefix=IR

; IR: .section .bss.hot.,"aw"

@hot_bss = internal global i32 0, !section_prefix !17

define void @hot_func() !prof !14 {
  %9 = load i32, ptr @hot_bss
  %11 = call i32 (...) @func_taking_arbitrary_param(i32 %9)
  ret void
}

declare i32 @func_taking_arbitrary_param(...)

!llvm.module.flags = !{!1}

!1 = !{i32 1, !"ProfileSummary", !2}
!2 = !{!3, !4, !5, !6, !7, !8, !9, !10}
!3 = !{!"ProfileFormat", !"InstrProf"}
!4 = !{!"TotalCount", i64 1460183}
!5 = !{!"MaxCount", i64 849024}
!6 = !{!"MaxInternalCount", i64 32769}
!7 = !{!"MaxFunctionCount", i64 849024}
!8 = !{!"NumCounts", i64 23627}
!9 = !{!"NumFunctions", i64 3271}
!10 = !{!"DetailedSummary", !11}
!11 = !{!12, !13}
!12 = !{i32 990000, i64 166, i32 73}
!13 = !{i32 999999, i64 3, i32 1443}
!14 = !{!"function_entry_count", i64 100000}
!15 = !{!"function_entry_count", i64 1}
!16 = !{!"branch_weights", i32 1, i32 99999}
!17 = !{!"section_prefix", !"hot"}
