; ModuleID = 'thinlto-function-summary-callgraph-profile-summary2.ll'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"


define void @hot1() #1 {
  ret void
}
define void @hot2() #1 !prof !20  {
  call void @calledFromHot()
  call void @calledFromHot()
  ret void
}
define void @hot3() #1 !prof !20 {
  call void @calledFromHot()
  call void @calledFromHot()
  call void @calledFromHot()
  ret void
}
define void @cold() #1 !prof !0 {
  ret void
}
define void @cold2() #1 !prof !0  {
  call void @calledFromCold()
  call void @calledFromCold()
  ret void
}

define void @none1() #1 {
  ret void
}

define void @none2() #1 {
  call void @calledFromNone()
  ret void
}
define void @none3() #1 {
  call void @calledFromNone()
  call void @calledFromNone()
  ret void
}

define void @calledFromCold() {
  ret void
}

define void @calledFromHot() !prof !20 {
  call void @calledFromHot2()
  ret void
}

define void @calledFromHot2() !prof !20 {
  call void @calledFromHot3()
  ret void
}

define void @calledFromNone() !prof !0 {
  ret void
}

declare void @calledFromHot3()

!0 = !{!"function_entry_count", i64 1}
!20 = !{!"function_entry_count", i64 110}

!llvm.module.flags = !{!1}

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