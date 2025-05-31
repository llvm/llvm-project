; RUN: not --crash opt -passes=loop-vectorize -disable-output %s

target triple = "x86_64"

define i8 @pr141968(i1 %cond, i8 %v) {
entry:
  %zext.true = zext i1 true to i16
  %sext = sext i8 %v to i16
  br label %loop.header

loop.header:                                      ; preds = %loop.latch, %entry
  %iv = phi i8 [ %iv.next, %loop.latch ], [ 0, %entry ]
  br i1 %cond, label %loop.latch, label %cond.false

cond.false:                                       ; preds = %loop.header
  %sdiv = sdiv i16 %sext, %zext.true
  %sdiv.trunc = trunc i16 %sdiv to i8
  br label %loop.latch

loop.latch:                                       ; preds = %cond.false, %loop.header
  %ret = phi i8 [ %sdiv.trunc, %cond.false ], [ 0, %loop.header ]
  %iv.next = add i8 %iv, 1
  %exitcond = icmp eq i8 %iv.next, 0
  br i1 %exitcond, label %exit, label %loop.header

exit:                                             ; preds = %loop.latch
  ret i8 %ret
}
