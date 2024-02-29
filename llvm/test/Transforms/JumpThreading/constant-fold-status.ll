; RUN: opt -passes=jump-threading < %s -S -o - | FileCheck %s

; Reproducer for PR47297.

; The pass did previously not report a correct Modified status in the case
; where a terminator's condition was successfully constant folded, but there
; were no other transformations done. This was caught by the pass return
; status check that is hidden under EXPENSIVE_CHECKS.

; CHECK-LABEL: entry:
; CHECK-NEXT: br i1 icmp eq (i32 ptrtoint (ptr @a to i32), i32 0), label %overflow, label %cont

@a = internal global i16 0

define void @foo(i16 %d) {
entry:
  %.not = icmp eq i32 ptrtoint (ptr @a to i32), 0
  br i1 %.not, label %overflow, label %cont

overflow:                                         ; preds = %entry
  call void @bar()
  br label %cont

cont:                                             ; preds = %overflow, %entry
  ret void
}

declare void @bar()
