; ModuleID = 'thinlto-function-summary-callgraph-profile-summary2.ll'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"


define void @hot() #1 !prof !28  {
  call void @calledFromHot()
  ret void
}

; 9 instructions so it is above decayed cold threshold of 7 and below
; decayed hot threshold of 10.
define void @calledFromHot() !prof !28 {
  %b = alloca i32, align 4
  store i32 1, i32* %b, align 4
  store i32 1, i32* %b, align 4
  store i32 1, i32* %b, align 4
  store i32 1, i32* %b, align 4
  store i32 1, i32* %b, align 4
  store i32 1, i32* %b, align 4
  store i32 1, i32* %b, align 4
  ret void
}

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
