; RUN: opt < %s -loop-reduce -S \
; RUN:    | grep "icmp eq i2 %lsr.iv.next, %xmp4344"

; Don't reverse the iteration if the rhs of the compare is defined
; inside the loop.

define void @Fill_Buffer(i2* %p) nounwind {
entry:
	br label %bb8

bb8:
	%indvar34 = phi i32 [ 0, %entry ], [ %indvar.next35, %bb8 ]
	%indvar3451 = trunc i32 %indvar34 to i2
	%xmp4344 = load i2* %p
	%xmp104 = icmp eq i2 %indvar3451, %xmp4344
	%indvar.next35 = add i32 %indvar34, 1
	br i1 %xmp104, label %bb10, label %bb8

bb10:
	unreachable
}
