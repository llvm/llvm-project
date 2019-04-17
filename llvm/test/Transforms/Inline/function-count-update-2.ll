; RUN: opt < %s -passes='require<profile-summary>,cgscc(inline)' -S | FileCheck %s

; This tests that the function count of a callee gets correctly updated after it
; has been inlined into a two callsites.

; CHECK: @callee() !prof [[COUNT:![0-9]+]]
define i32 @callee() !prof !1 {
  ret i32 0
}

define i32 @caller1() !prof !2 {
; CHECK-LABEL: @caller1
; CHECK-NOT: callee
; CHECK: ret
  %i = call i32 @callee()
  ret i32 %i
}

define i32 @caller2() !prof !3 {
; CHECK-LABEL: @caller2
; CHECK-NOT: callee
; CHECK: ret
  %i = call i32 @callee()
  ret i32 %i
}

!llvm.module.flags = !{!0}
; CHECK: [[COUNT]] = !{!"function_entry_count", i64 0}
!0 = !{i32 1, !"MaxFunctionCount", i32 1000}
!1 = !{!"function_entry_count", i64 1000}
!2 = !{!"function_entry_count", i64 600}
!3 = !{!"function_entry_count", i64 400}

