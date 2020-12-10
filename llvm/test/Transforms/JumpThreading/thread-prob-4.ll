; RUN: opt -debug-only=branch-prob -jump-threading -S %s 2>&1 | FileCheck %s
; REQUIRES: asserts

; Make sure that we clear edge probabilities for bb1 as we fold
; the conditional branch in it.

; CHECK: eraseBlock bb1

define i32 @foo(i32 %arg) !prof !0 {
; CHECK-LABEL: @foo
  %cond1 = icmp eq i32 %arg, 42
  br i1 %cond1, label %bb1, label %bb2

bb2:
  ret i32 2

bb1:
  %cond2 = icmp eq i32 %arg, 42
  br i1 %cond2, label %bb3, label %bb4

bb3:
  ret i32 3

bb4:
; CHECK-NOT: bb4:
  ret i32 4
}

!0 = !{!"function_entry_count", i64 0}
