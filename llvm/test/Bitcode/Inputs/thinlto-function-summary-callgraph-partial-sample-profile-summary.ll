; ModuleID = 'thinlto-function-summary-callgraph-profile-summary2.ll'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @hot1() #1 {
  ret void
}
define void @hot2() #1 {
  ret void
}
define void @hot3() #1 {
  ret void
}
define void @cold1() #1 {
  ret void
}
define void @cold2() #1 {
  ret void
}
define void @cold3() #1 {
  ret void
}
define void @none1() #1 {
  ret void
}
define void @none2() #1 {
  ret void
}
define void @none3() #1 {
  ret void
}

!llvm.module.flags = !{!1}

!1 = !{i32 1, !"ProfileSummary", !2}
!2 = !{!3, !4, !5, !6, !7, !8, !9, !10, !11, !12}
!3 = !{!"ProfileFormat", !"SampleProfile"}
!4 = !{!"TotalCount", i64 10000}
!5 = !{!"MaxCount", i64 10}
!6 = !{!"MaxInternalCount", i64 1}
!7 = !{!"MaxFunctionCount", i64 1000}
!8 = !{!"NumCounts", i64 3}
!9 = !{!"NumFunctions", i64 3}
!10 = !{!"IsPartialProfile", i64 1}
!11 = !{!"PartialProfileRatio", double 0.5}
!12 = !{!"DetailedSummary", !13}
!13 = !{!14, !15, !16}
!14 = !{i32 10000, i64 100, i32 1}
!15 = !{i32 999000, i64 100, i32 1}
!16 = !{i32 999999, i64 1, i32 2}
