; RUN: opt -S < %s -jump-threading | FileCheck %s
; CHECK-LABEL: @foo
; CHECK-NEXT: exit2:
; CHECK-NEXT: ret void
define void @foo() {
entry:
  br label %bb1

entry2:
  br label %bb1

bb1:
  %a0 = phi i32 [ undef, %entry2 ], [ 0, %entry ]
  %b = icmp ne i32 %a0, 0
  br i1 %b, label %bb2, label %exit2

bb2:
  br label %exit1

exit1:
  %a1 = phi i32 [ %a0, %bb2 ]
  ret void

exit2:
  ret void
}
