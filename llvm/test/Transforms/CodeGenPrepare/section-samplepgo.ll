; RUN: opt < %s -codegenprepare -S | FileCheck %s

target triple = "x86_64-pc-linux-gnu"

; This tests that hot/cold functions get correct section prefix assigned

; CHECK: hot_func{{.*}}!section_prefix ![[HOT_ID:[0-9]+]]
; The entry is hot
define void @hot_func() !prof !15 {
  ret void
}

; CHECK: hot_call_func{{.*}}!section_prefix ![[HOT_ID]]
; The sum of 2 callsites are hot
define void @hot_call_func() !prof !16 {
  call void @hot_func(), !prof !17
  call void @hot_func(), !prof !17
  ret void
}

; CHECK-NOT: normal_func{{.*}}!section_prefix
; The sum of all callsites are neither hot or cold
define void @normal_func() !prof !16 {
  call void @hot_func(), !prof !17
  call void @hot_func(), !prof !18
  call void @hot_func(), !prof !18
  ret void
}

; CHECK: cold_func{{.*}}!section_prefix ![[COLD_ID:[0-9]+]]
; The entry and the callsite are both cold
define void @cold_func() !prof !16 {
  call void @hot_func(), !prof !18
  ret void
}

; CHECK: ![[HOT_ID]] = !{!"function_section_prefix", !".hot"}
; CHECK: ![[COLD_ID]] = !{!"function_section_prefix", !".unlikely"}
!llvm.module.flags = !{!1}
!1 = !{i32 1, !"ProfileSummary", !2}
!2 = !{!3, !4, !5, !6, !7, !8, !9, !10}
!3 = !{!"ProfileFormat", !"SampleProfile"}
!4 = !{!"TotalCount", i64 10000}
!5 = !{!"MaxCount", i64 1000}
!6 = !{!"MaxInternalCount", i64 1}
!7 = !{!"MaxFunctionCount", i64 1000}
!8 = !{!"NumCounts", i64 3}
!9 = !{!"NumFunctions", i64 3}
!10 = !{!"DetailedSummary", !11}
!11 = !{!12, !13, !14}
!12 = !{i32 10000, i64 100, i32 1}
!13 = !{i32 999000, i64 100, i32 1}
!14 = !{i32 999999, i64 1, i32 2}
!15 = !{!"function_entry_count", i64 1000}
!16 = !{!"function_entry_count", i64 1}
!17 = !{!"branch_weights", i32 80}
!18 = !{!"branch_weights", i32 1}
