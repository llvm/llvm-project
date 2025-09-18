; Test that prof-inject does not modify existing metadata (incl. "unknown")

; RUN: opt -passes=prof-verify %s -S --disable-output

define void @foo(i32 %i) !prof !0 {
  %c = icmp eq i32 %i, 0
  br i1 %c, label %yes, label %no, !prof !1
yes:
  br i1 %c, label %yes2, label %no, !prof !2
yes2:
  ret void
no:
  ret void
}

!0 = !{!"function_entry_count", i32 1}
!1 = !{!"branch_weights", i32 1, i32 2}
!2 = !{!"unknown", !"test"}
; CHECK: define void @foo(i32 %i) !prof !0
; CHECK: br i1 %c, label %yes, label %no, !prof !1
; CHECK: !0 = !{!"function_entry_count", i64 1}
; CHECK: !1 = !{!"branch_weights", i32 1, i32 2}
; CHECK: !2 = !{!"unknown", !"test"}
