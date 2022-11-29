; RUN: opt -debug-only=branch-prob -passes=jump-threading -S %s 2>&1 | FileCheck %s
; REQUIRES: asserts

; Make sure that we clear edge probabilities for bb1 as we fold
; the conditional branch in it.

; CHECK: eraseBlock bb1

define void @foo(i32 %i, i32 %len) !prof !0 {
; CHECK-LABEL: @foo
  %i.inc = add nuw i32 %i, 1
  %c0 = icmp ult i32 %i.inc, %len
  br i1 %c0, label %bb1, label %bb2

bb1:
; CHECK: bb1:
  %c1 = icmp ult i32 %i, %len
  br i1 %c1, label %bb2, label %bb3

bb2:
  ret void

bb3:
; CHECK-NOT: bb3:
  ret void
}

!0 = !{!"function_entry_count", i64 0}
