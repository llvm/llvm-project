; RUN: opt -debug-only=branch-prob -jump-threading -S %s 2>&1 | FileCheck %s
; REQUIRES: asserts

; Make sure that we clear edge probabilities for bb1 as we fold
; the conditional branch in it.

; CHECK: eraseBlock bb1

define i32 @foo() !prof !0 {
; CHECK-LABEL: @foo
bb1:
  br i1 true, label %bb2, label %bb3

bb2:
  ret i32 3

bb3:
; CHECK-NOT: bb3:
  ret i32 7
}

!0 = !{!"function_entry_count", i64 0}
