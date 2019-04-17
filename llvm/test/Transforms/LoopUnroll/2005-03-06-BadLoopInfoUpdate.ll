; RUN: opt < %s -loop-unroll -loop-simplify -disable-output

define void @print_board() {
entry:
	br label %no_exit.1
no_exit.1:		; preds = %cond_false.2, %entry
	br label %no_exit.2
no_exit.2:		; preds = %no_exit.2, %no_exit.1
	%indvar1 = phi i32 [ 0, %no_exit.1 ], [ %indvar.next2, %no_exit.2 ]		; <i32> [#uses=1]
	%indvar.next2 = add i32 %indvar1, 1		; <i32> [#uses=2]
	%exitcond3 = icmp ne i32 %indvar.next2, 7		; <i1> [#uses=1]
	br i1 %exitcond3, label %no_exit.2, label %loopexit.2
loopexit.2:		; preds = %no_exit.2
	br i1 false, label %cond_true.2, label %cond_false.2
cond_true.2:		; preds = %loopexit.2
	ret void
cond_false.2:		; preds = %loopexit.2
	br i1 false, label %no_exit.1, label %loopexit.1
loopexit.1:		; preds = %cond_false.2
	ret void
}

