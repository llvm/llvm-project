; RUN: opt < %s -inline -inline-threshold=0 -inlinehint-threshold=100 -S | FileCheck %s

; This tests that a hot callee gets the (higher) inlinehint-threshold even without
; inline hints and gets inlined because the cost is less than inlinehint-threshold.
; A cold callee with identical body does not get inlined because cost exceeds the
; inline-threshold

define i32 @callee1(i32 %x) !prof !20 {
  %x1 = add i32 %x, 1
  %x2 = add i32 %x1, 1
  %x3 = add i32 %x2, 1

  ret i32 %x3
}

define i32 @callee2(i32 %x) !prof !21 {
; CHECK-LABEL: @callee2(
  %x1 = add i32 %x, 1
  %x2 = add i32 %x1, 1
  %x3 = add i32 %x2, 1

  ret i32 %x3
}

define i32 @caller2(i32 %y1) !prof !21 {
; CHECK-LABEL: @caller2(
; CHECK: call i32 @callee2
; CHECK-NOT: call i32 @callee1
; CHECK: ret i32 %x3.i
  %y2 = call i32 @callee2(i32 %y1)
  %y3 = call i32 @callee1(i32 %y2)
  ret i32 %y3
}

!llvm.module.flags = !{!1}
!20 = !{!"function_entry_count", i64 10}
!21 = !{!"function_entry_count", i64 1}

!1 = !{i32 1, !"ProfileSummary", !2}
!2 = !{!3, !4, !5, !6, !7, !8, !9, !10}
!3 = !{!"ProfileFormat", !"InstrProf"}
!4 = !{!"TotalCount", i64 10000}
!5 = !{!"MaxBlockCount", i64 10}
!6 = !{!"MaxInternalBlockCount", i64 1}
!7 = !{!"MaxFunctionCount", i64 10}
!8 = !{!"NumBlocks", i64 3}
!9 = !{!"NumFunctions", i64 3}
!10 = !{!"DetailedSummary", !11}
!11 = !{!12}
!12 = !{i32 10000, i64 0, i32 0}
