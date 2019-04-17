; RUN: opt < %s -loop-unroll -unroll-count=3 -S | grep bb72.2

define void @foo(i32 %trips) {
entry:
	br label %cond_true.outer

cond_true.outer:
	%indvar1.ph = phi i32 [ 0, %entry ], [ %indvar.next2, %bb72 ]
	br label %bb72

bb72:
	%indvar.next2 = add i32 %indvar1.ph, 1
	%exitcond3 = icmp eq i32 %indvar.next2, %trips
	br i1 %exitcond3, label %cond_true138, label %cond_true.outer

cond_true138:
	ret void
}
