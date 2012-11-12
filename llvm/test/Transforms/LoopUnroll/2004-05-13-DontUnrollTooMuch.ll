; RUN: opt < %s -loop-unroll -disable-output

define i32 @main() {
entry:
	br label %no_exit
no_exit:		; preds = %no_exit, %entry
	%indvar = phi i32 [ 0, %entry ], [ %indvar.next, %no_exit ]		; <i32> [#uses=1]
	%indvar.next = add i32 %indvar, 1		; <i32> [#uses=2]
	%exitcond = icmp ne i32 %indvar.next, -2147483648		; <i1> [#uses=1]
	br i1 %exitcond, label %no_exit, label %loopexit
loopexit:		; preds = %no_exit
	ret i32 0
}

