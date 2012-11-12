; RUN: llc < %s -mtriple=arm-apple-darwin

define void @gcov_exit() nounwind {
entry:
	br i1 false, label %bb24, label %bb33.thread

bb24:		; preds = %entry
	br label %bb39

bb33.thread:		; preds = %entry
	%0 = alloca i8, i32 0		; <i8*> [#uses=1]
	br label %bb39

bb39:		; preds = %bb33.thread, %bb24
	%.reg2mem.0 = phi i8* [ %0, %bb33.thread ], [ null, %bb24 ]		; <i8*> [#uses=0]
	ret void
}
