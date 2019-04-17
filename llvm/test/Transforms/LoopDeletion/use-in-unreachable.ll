; RUN: opt < %s -loop-deletion -S | FileCheck %s

; Checking that possible users of instruction from the loop in
; unreachable blocks are handled.

define i64 @foo() {
entry:
  br label %invloop
; CHECK-LABEL-NOT: invloop
invloop:
  %indvar1 = phi i64 [ 3, %entry ], [ %indvar2, %invloop_iter ]
  %check = icmp ult i64 %indvar1, 400
  br i1 %check, label %invloop_iter, label %loopexit
invloop_iter:
  %indvar2 = add i64 %indvar1, 1
  %baddef = add i64 0, 0
  br label %invloop
loopexit:
  ret i64 0
deadcode:
; CHECK-LABEL: deadcode
; CHECK: ret i64 undef
  ret i64 %baddef
}
