; RUN: opt -passes=loop-interchange -loop-interchange-profitabilities=ignore --disable-output %s
; Loop-interchange pass doesn't crash if the inner loop preheader has PHI nodes.


define void @f(ptr %A) {
entry:
  br label %outer.header

outer.header:
  %i = phi i64 [ 0, %entry ], [ %i.inc, %outer.latch ]
  br label %inner.ph

inner.ph:
  %p = phi i64 [ 42, %outer.header ]
  br label %inner

inner:
  %j = phi i64 [ 0, %inner.ph ], [ %j.inc, %inner ]
  %j.inc = add i64 %j, 1
  %ec.j = icmp eq i64 %j.inc, 10
  br i1 %ec.j, label %outer.body, label %inner

outer.body:
  br label %outer.latch

outer.latch:
  %i.inc = add i64 %i, 1
  %ec.i = icmp eq i64 %i.inc, 10
  br i1 %ec.i, label %exit, label %outer.header

exit:
  ret void
}
