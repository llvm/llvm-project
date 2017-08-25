; For SamplePGO, if -profile-sample-accurate is specified, cold callsite
; heuristics should be honored if the caller has no profile.

; RUN: opt < %s -inline -S -inline-cold-callsite-threshold=0 | FileCheck %s

define i32 @callee(i32 %x) {
  %x1 = add i32 %x, 1
  %x2 = add i32 %x1, 1
  %x3 = add i32 %x2, 1
  call void @extern()
  call void @extern()
  ret i32 %x3
}

define i32 @caller(i32 %y1) {
; CHECK-LABEL: @caller
; CHECK-NOT: call i32 @callee
  %y2 = call i32 @callee(i32 %y1)
  ret i32 %y2
}

define i32 @caller_accurate(i32 %y1) #0 {
; CHECK-LABEL: @caller_accurate
; CHECK: call i32 @callee
  %y2 = call i32 @callee(i32 %y1)
  ret i32 %y2
}

declare void @extern()

attributes #0 = { "profile-sample-accurate" }

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
