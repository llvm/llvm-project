; RUN: opt < %s -inline -inline-threshold=0 -hot-callsite-threshold=100 -S | FileCheck %s

; This tests that a hot callsite gets the (higher) inlinehint-threshold even without
; without inline hints and gets inlined because the cost is less than
; inlinehint-threshold. A cold callee with identical body does not get inlined because
; cost exceeds the inline-threshold

define i32 @callee1(i32 %x) {
  %x1 = add i32 %x, 1
  %x2 = add i32 %x1, 1
  %x3 = add i32 %x2, 1

  ret i32 %x3
}

define i32 @callee2(i32 %x) {
; CHECK-LABEL: @callee2(
  %x1 = add i32 %x, 1
  %x2 = add i32 %x1, 1
  %x3 = add i32 %x2, 1

  ret i32 %x3
}

define i32 @caller2(i32 %y1) {
; CHECK-LABEL: @caller2(
; CHECK: call i32 @callee2
; CHECK-NOT: call i32 @callee1
; CHECK: ret i32 %x3.i
  %y2 = call i32 @callee2(i32 %y1), !prof !22
  %y3 = call i32 @callee1(i32 %y2), !prof !21
  ret i32 %y3
}

!llvm.module.flags = !{!1}
!21 = !{!"branch_weights", i64 300}
!22 = !{!"branch_weights", i64 1}

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
