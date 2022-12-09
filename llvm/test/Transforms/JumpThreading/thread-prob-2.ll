; RUN: opt -debug-only=branch-prob -passes=jump-threading -S %s 2>&1 | FileCheck %s
; REQUIRES: asserts

; Make sure that we clear edge probabilities for bb.cond as we fold
; the conditional branch in it.

; CHECK: eraseBlock bb.cond

define i32 @foo(i1 %cond) !prof !0 {
; CHECK-LABEL: @foo
; CHECK: bb.entry:
; CHECK-NEXT: br i1 %cond, label %bb.31, label %bb.12
; CHECK-NOT: bb.cond:
bb.entry:
  br i1 %cond, label %bb.31, label %bb.cond

bb.cond:
  br i1 %cond, label %bb.31, label %bb.12

bb.31:
  ret i32 31

bb.12:
  ret i32 12
}

!0 = !{!"function_entry_count", i64 0}
