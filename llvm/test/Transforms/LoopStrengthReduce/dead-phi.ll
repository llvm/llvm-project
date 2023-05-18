; RUN: opt < %s -loop-reduce -S | grep phi | count 1

define void @foo(i32 %n) {
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]

  ; These three instructions form an isolated cycle and can be deleted.
  %j = phi i32 [ 0, %entry ], [ %j.y, %loop ]
  %j.x = add i32 %j, 1
  %j.y = mul i32 %j.x, 2

  %i.next = add i32 %i, 1
  %c = icmp ne i32 %i.next, %n
  br i1 %c, label %loop, label %exit

exit:
  ret void
}
