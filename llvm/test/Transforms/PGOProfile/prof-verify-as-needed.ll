; Test that prof-inject only injects missing metadata

; RUN: opt -passes=prof-inject -profcheck-default-function-entry-count=10 %s -S -o - | FileCheck %s

define void @foo(i32 %i) {
  %c = icmp eq i32 %i, 0
  br i1 %c, label %yes, label %no, !prof !0
yes:
  br i1 %c, label %yes2, label %no
yes2:
  ret void
no:
  ret void
}

define void @cold(i32 %i) !prof !1 {
  %c = icmp eq i32 %i, 0
  br i1 %c, label %yes, label %no
yes:
  br i1 %c, label %yes2, label %no
yes2:
  ret void
no:
  ret void
}
!0 = !{!"branch_weights", i32 1, i32 2}
!1 = !{!"function_entry_count", i32 0}

; CHECK-LABEL: @foo
; CHECK: br i1 %c, label %yes, label %no, !prof !1
; CHECK: br i1 %c, label %yes2, label %no, !prof !2
; CHECK-LABEL: @cold
; CHECK: br i1 %c, label %yes, label %no{{$}}
; CHECK: br i1 %c, label %yes2, label %no{{$}}
; CHECK: !0 = !{!"function_entry_count", i64 10}
; CHECK: !1 = !{!"branch_weights", i32 1, i32 2}
; CHECK: !2 = !{!"branch_weights", i32 3, i32 5}
; CHECK: !3 = !{!"function_entry_count", i32 0}
