; RUN: opt < %s -passes=loop-interchange -loop-interchange-profitabilities=ignore \
; RUN:     -verify-dom-info -verify-loop-info -verify-loop-lcssa \
; RUN:     -pass-remarks-output=%t -disable-output
; RUN: FileCheck -input-file %t %s

; After interchanging the innermost and the middle loop, an lcssa phi in the new
; inner latch feeds the latch's exit condition. Interchanging the (new) middle
; and outermost loop would clone that condition and leave the lcssa phi with a
; stale incoming block, producing invalid IR. Make sure the second interchange
; is rejected instead of crashing, even though the outermost latch has a single
; predecessor. 

; CHECK:  --- !Passed
; CHECK:  Pass:            loop-interchange
; CHECK:  Name:            Interchanged
; CHECK:  Function:        t
; CHECK:  ...
; CHECK:  --- !Missed
; CHECK:  Pass:            loop-interchange
; CHECK:  Name:            UnsupportedInnerLatchPHI
; CHECK:  Function:        t
; CHECK:    - String:          Cannot interchange loops because unsupported PHI nodes found in inner loop latch.
; CHECK:  ...

define void @t(i32 %b) {
entry:
  br label %outer.header

outer.header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %outer.latch ]
  br label %mid.header

mid.header:
  %j = phi i64 [ 0, %outer.header ], [ %j.next, %mid.latch ]
  %mwide = zext i32 %b to i64                 ; non-induction value, middle-loop invariant
  br label %inner.header

inner.header:
  %k = phi i64 [ 0, %mid.header ], [ %k.next, %inner.latch ]
  br label %inner.latch

inner.latch:
  %k.next = add nsw i64 %k, 1
  %ec.k = icmp eq i64 %k.next, %mwide         ; inner exit uses the middle-header value
  br i1 %ec.k, label %mid.latch, label %inner.header

mid.latch:
  %j.next = add nsw i64 %j, 1
  %ec.j = icmp eq i64 %j.next, 100
  br i1 %ec.j, label %outer.latch, label %mid.header

outer.latch:
  %i.next = add nsw i64 %i, 1
  %ec.i = icmp eq i64 %i.next, 100
  br i1 %ec.i, label %exit, label %outer.header

exit:
  ret void
}
