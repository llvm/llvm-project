; Test that prof-inject only injects missing metadata

; RUN: opt -passes=prof-inject %s -S -o - | FileCheck %s

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

!0 = !{!"branch_weights", i32 1, i32 2}
; CHECK: br i1 %c, label %yes, label %no, !prof !0
; CHECK: br i1 %c, label %yes2, label %no, !prof !1
; CHECK: !0 = !{!"branch_weights", i32 1, i32 2}
; CHECK: !1 = !{!"branch_weights", i32 3, i32 5}
