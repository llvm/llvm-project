; RUN: opt < %s -inline -inline-threshold=0 -inlinehint-threshold=100 -S | FileCheck %s

; This tests that a hot callee gets the (higher) inlinehint-threshold even without
; inline hints and gets inlined because the cost is less than inlinehint-threshold.
; A cold callee with identical body does not get inlined because cost exceeds the
; inline-threshold

define i32 @callee1(i32 %x) !prof !1 {
  %x1 = add i32 %x, 1
  %x2 = add i32 %x1, 1
  %x3 = add i32 %x2, 1

  ret i32 %x3
}

define i32 @callee2(i32 %x) !prof !2 {
; CHECK-LABEL: @callee2(
  %x1 = add i32 %x, 1
  %x2 = add i32 %x1, 1
  %x3 = add i32 %x2, 1

  ret i32 %x3
}

define i32 @caller2(i32 %y1) !prof !2 {
; CHECK-LABEL: @caller2(
; CHECK: call i32 @callee2
; CHECK-NOT: call i32 @callee1
; CHECK: ret i32 %x3.i
  %y2 = call i32 @callee2(i32 %y1)
  %y3 = call i32 @callee1(i32 %y2)
  ret i32 %y3
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"MaxFunctionCount", i32 10}
!1 = !{!"function_entry_count", i64 10}
!2 = !{!"function_entry_count", i64 1}

