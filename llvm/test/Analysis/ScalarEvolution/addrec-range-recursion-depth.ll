; RUN: opt -disable-output -passes='print<scalar-evolution>' %s
; RUN: opt -disable-output -passes='print<scalar-evolution>' \
; RUN:   -scalar-evolution-max-add-rec-range-depth=0 %s

; Computing the range of an affine addrec requires the loop's backedge-taken
; count, whose computation can recurse back into range computation through
; loop-guard reasoning. On pathological inputs this mutual recursion can chain
; across many loops and overflow the stack. ScalarEvolution bounds the depth of
; this recursion via -scalar-evolution-max-add-rec-range-depth; verify that
; analysis still succeeds (and does not crash) when that refinement is limited.

define void @nested(i32 %n, i32 %m, ptr %p) {
entry:
  br label %outer.header

outer.header:
  %i = phi i32 [ 0, %entry ], [ %i.next, %outer.latch ]
  %outer.cmp = icmp slt i32 %i, %n
  br i1 %outer.cmp, label %inner.header, label %exit

inner.header:
  %j = phi i32 [ 0, %outer.header ], [ %j.next, %inner.header ]
  %idx = add nsw i32 %i, %j
  %gep = getelementptr inbounds i32, ptr %p, i32 %idx
  store i32 %idx, ptr %gep, align 4
  %j.next = add nsw i32 %j, 1
  %inner.cmp = icmp slt i32 %j.next, %m
  br i1 %inner.cmp, label %inner.header, label %outer.latch

outer.latch:
  %i.next = add nsw i32 %i, 1
  br label %outer.header

exit:
  ret void
}
