; RUN: opt < %s -codegenprepare -S | FileCheck --check-prefixes=CHECK-OPT %s
; RUN: llc < %s -o - | FileCheck --check-prefixes=CHECK-LLC %s

target triple = "x86_64-pc-linux-gnu"

; This tests that hot/cold functions get correct section prefix assigned

; CHECK-OPT: hot_func{{.*}}!section_prefix ![[HOT_ID:[0-9]+]]
; CHECK-LLC: .section .text.hot
; CHECK-LLC-NEXT: .globl hot_func
define void @hot_func() !prof !15 {
  ret void
}

; CHECK-OPT: cold_func{{.*}}!section_prefix ![[COLD_ID:[0-9]+]]
; CHECK-LLC: .section .text.cold
; CHECK-LLC-NEXT: .globl cold_func
define void @cold_func() !prof !16 {
  ret void
}

; CHECK-OPT: ![[HOT_ID]] = !{!"function_section_prefix", !".hot"}
; CHECK-OPT: ![[COLD_ID]] = !{!"function_section_prefix", !".cold"}
!llvm.module.flags = !{!1}
!1 = !{i32 1, !"ProfileSummary", !2}
!2 = !{!3, !4, !5, !6, !7, !8, !9, !10}
!3 = !{!"ProfileFormat", !"InstrProf"}
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
