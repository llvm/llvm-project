; RUN: opt < %s -analyze -iv-users | grep "{1,+,3,+,2}<%loop> (post-inc with loop %loop)"

; The value of %r is dependent on a polynomial iteration expression.

define i64 @foo(i64 %n) {
entry:
  br label %loop

loop:
  %indvar = phi i64 [ 0, %entry ], [ %indvar.next, %loop ]
  %indvar.next = add i64 %indvar, 1
  %c = icmp eq i64 %indvar.next, %n
  br i1 %c, label %exit, label %loop

exit:
  %r = mul i64 %indvar.next, %indvar.next
  ret i64 %r
}
