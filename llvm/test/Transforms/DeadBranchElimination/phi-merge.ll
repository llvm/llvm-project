; RUN: opt -passes=dead-branch-elim -S %s | FileCheck %s

; PHI nodes at the merge point must be rewired to the surviving side's value,
; in both fold directions.

; CHECK-LABEL: define i32 @then_side_dead()
; CHECK-NOT: then:
; CHECK: else:
; CHECK: %b = add nsw i32 %x, 1
; CHECK-NOT: %x.next = phi
; CHECK: ret i32 %b
define i32 @then_side_dead() {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %merge ]
  %x = phi i32 [ 0, %entry ], [ %x.next, %merge ]
  %cmp = icmp sgt i32 %i, 1000
  br i1 %cmp, label %then, label %else

then:
  %a = add nsw i32 %x, 100
  br label %merge

else:
  %b = add nsw i32 %x, 1
  br label %merge

merge:
  %x.next = phi i32 [ %a, %then ], [ %b, %else ]
  %i.next = add nsw i32 %i, 1
  %cond = icmp slt i32 %i.next, 10
  br i1 %cond, label %loop, label %exit

exit:
  ret i32 %x.next
}

; CHECK-LABEL: define i32 @else_side_dead()
; CHECK: then:
; CHECK: %a = add nsw i32 %x, 3
; CHECK-NOT: else:
; CHECK-NOT: %x.next = phi
; CHECK: ret i32 %a
define i32 @else_side_dead() {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %merge ]
  %x = phi i32 [ 0, %entry ], [ %x.next, %merge ]
  %cmp = icmp slt i32 %i, 1000
  br i1 %cmp, label %then, label %else

then:
  %a = add nsw i32 %x, 3
  br label %merge

else:
  %b = add nsw i32 %x, 7
  br label %merge

merge:
  %x.next = phi i32 [ %a, %then ], [ %b, %else ]
  %i.next = add nsw i32 %i, 1
  %cond = icmp slt i32 %i.next, 10
  br i1 %cond, label %loop, label %exit

exit:
  ret i32 %x.next
}
