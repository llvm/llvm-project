; REQUIRES: asserts
; RUN: not --crash opt -passes=loop-vectorize -force-vector-width=4 -disable-output %s

define i8 @pr87407(i8 %x, i64 %y, i64 %n) {
entry:
  %zext.x = zext i8 %x to i64
  br label %loop

loop:
  %iv = phi i64 [ %iv.next, %loop ], [ 0, %entry ]
  %max = tail call i64 @llvm.umax.i64(i64 %zext.x, i64 %y)
  %cmp.max.0 = icmp ne i64 %max, 0
  %zext.cmp = zext i1 %cmp.max.0 to i64
  %trunc = trunc i64 %zext.cmp to i32
  %shl = shl i32 %trunc, 8
  %res = trunc i32 %shl to i8
  %iv.next = add i64 %iv, 1
  %exit.cond = icmp ne i64 %iv.next, %n
  br i1 %exit.cond, label %loop, label %exit

exit:
  ret i8 %res
}
