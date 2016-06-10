; RUN: opt < %s -inline  -inlinecold-threshold=0 -S | FileCheck %s

; This tests that a cold callee gets the (lower) inlinecold-threshold even without
; Cold hint and does not get inlined because the cost exceeds the inlinecold-threshold.
; A callee with identical body does gets inlined because cost fits within the
; inline-threshold

define i32 @callee1(i32 %x) !prof !21 {
  %x1 = add i32 %x, 1
  %x2 = add i32 %x1, 1
  %x3 = add i32 %x2, 1

  ret i32 %x3
}

define i32 @callee2(i32 %x) !prof !22 {
; CHECK-LABEL: @callee2(
  %x1 = add i32 %x, 1
  %x2 = add i32 %x1, 1
  %x3 = add i32 %x2, 1

  ret i32 %x3
}

define i32 @caller2(i32 %y1) !prof !22 {
; CHECK-LABEL: @caller2(
; CHECK: call i32 @callee2
; CHECK-NOT: call i32 @callee1
; CHECK: ret i32 %x3.i
  %y2 = call i32 @callee2(i32 %y1)
  %y3 = call i32 @callee1(i32 %y2)
  ret i32 %y3
}

!llvm.module.flags = !{!1}
!21 = !{!"function_entry_count", i64 100}
!22 = !{!"function_entry_count", i64 1}

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
