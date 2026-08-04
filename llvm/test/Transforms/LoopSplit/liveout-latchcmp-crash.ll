; REQUIRES: asserts
; RUN: not --crash opt -passes=loop-split-utils -loop-split-points=50 -disable-output < %s

declare void @use(i32)

define i1 @latchcmp_liveout(i32 %n) {
entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop ]
  call void @use(i32 %iv)
  %iv.next = add i32 %iv, 1
  %cmp = icmp slt i32 %iv.next, %n
  br i1 %cmp, label %loop, label %exit

exit:
  %cmp.lcssa = phi i1 [ %cmp, %loop ]
  ret i1 %cmp.lcssa
}
