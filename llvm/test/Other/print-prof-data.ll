; RUN: opt %s -print-prof-data -S | FileCheck %s

define void @foo(ptr %p) !prof !0 {
  %isnull = icmp eq ptr %p, null
  br i1 %isnull, label %yes, label %no, !prof !1
yes:
  %something = select i1 %isnull, i32 1, i32 2, !prof !2
  br label %exit
no:
  call void %p(), !prof !3
  br label %exit
exit:
  ret void
}

!0 = !{!"function_entry_count", i64 42}
!1 = !{!"branch_weights", i64 20, i64 101}
!2 = !{!"branch_weights", i64 5, i64 70}
!3 = !{!"VP", i32 0, i64 4, i64 4445083295448962937, i64 2, i64 -2718743882639408571, i64 2}

; CHECK: define void @foo(ptr %p) !0 = !{!"function_entry_count", i64 42} !prof !0 {
; CHECK: br i1 %isnull, label %yes, label %no, !prof !1 ; !1 = !{!"branch_weights", i64 20, i64 101}
; CHECK: %something = select i1 %isnull, i32 1, i32 2, !prof !2 ; !2 = !{!"branch_weights", i64 5, i64 70}
; CHECK: call void %p(), !prof !3 ; !3 = !{!"VP", i32 0, i64 4, i64 4445083295448962937, i64 2, i64 -2718743882639408571, i64 2}