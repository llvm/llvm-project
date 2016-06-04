; RUN: opt < %s -disable-output -passes=print-profile-summary -S 2>&1 | FileCheck %s

define void @f1() !prof !20 {
; CHECK-LABEL: f1 :hot

  ret void
}

define void @f2() !prof !21 {
; CHECK-LABEL: f2 :cold

  ret void
}

define void @f3() !prof !22 {
; CHECK-LABEL: f3

  ret void
}

!llvm.module.flags = !{!1}
!20 = !{!"function_entry_count", i64 400}
!21 = !{!"function_entry_count", i64 1}
!22 = !{!"function_entry_count", i64 100}

!1 = !{i32 1, !"ProfileSummary", !2}
!2 = !{!3, !4, !5, !6, !7, !8, !9, !10}
!3 = !{!"ProfileFormat", !"InstrProf"}
!4 = !{!"TotalCount", i64 10000}
!5 = !{!"MaxCount", i64 10}
!6 = !{!"MaxInternalCount", i64 1}
!7 = !{!"MaxFunctionCount", i64 1000}
!8 = !{!"NumCounts", i64 3}
!9 = !{!"NumFunctions", i64 3}
!10 = !{!"DetailedSummary", !11}
!11 = !{!12, !13, !14}
!12 = !{i32 10000, i64 100, i32 1}
!13 = !{i32 999000, i64 100, i32 1}
!14 = !{i32 999999, i64 1, i32 2}
