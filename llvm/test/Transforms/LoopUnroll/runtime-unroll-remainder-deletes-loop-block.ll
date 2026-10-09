; RUN: opt -passes='loop-unroll,verify<domtree>,verify<loops>,verify' -unroll-remainder -disable-output %s

; The redundant %p.o PHI prevents the inner latch from being folded during the
; first loop simplification. Unrolling the outer loop's runtime remainder
; simplifies its parent loop nest and deletes that latch. The outer unroll must
; snapshot its original blocks after remainder generation, or its dominator
; tree update will access the deleted block.

define void @test(ptr %out, i32 %n, i32 %seed, i1 %keep.going) {
entry:
  br label %wrapper.header

wrapper.header:
  br label %outer.header

outer.header:
  %i = phi i32 [ 0, %wrapper.header ], [ %i.next, %outer.latch ]
  %p.o = phi i32 [ %seed, %wrapper.header ], [ %seed, %outer.latch ]
  br label %inner.header

inner.header:
  %j = phi i32 [ 0, %outer.header ], [ 1, %inner.latch ]
  %j.next1 = add i32 0, 0
  br i1 %keep.going, label %inner.latch, label %outer.latch

inner.latch:
  %inner.cond = icmp ult i32 0, 0
  br i1 %inner.cond, label %inner.header, label %outer.latch, !llvm.loop !0

outer.latch:
  %p.inner.lcssa = phi i32 [ %p.o, %inner.header ], [ %seed, %inner.latch ]
  %i.next = add i32 %i, 1
  %outer.cond = icmp ult i32 %i, %n
  br i1 %outer.cond, label %outer.header, label %outer.exit, !llvm.loop !2

outer.exit:
  br i1 %keep.going, label %wrapper.header, label %exit

exit:
  store i32 0, ptr %out
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.disable"}
!2 = distinct !{!2, !3}
!3 = !{!"llvm.loop.unroll.count", i32 8}
