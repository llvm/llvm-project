; Make sure that the compare instruction occurs after the increment to avoid
; having overlapping live ranges that result in copies.  We want the setcc 
; instruction immediately before the conditional branch.
;
; RUN: llvm-as < %s | opt -loop-reduce | llvm-dis | \
; RUN:    %prcontext {br i1} 1 | grep icmp

define void @foo(float* %D, i32 %E) {
entry:
	br label %no_exit
no_exit:		; preds = %no_exit, %entry
	%indvar = phi i32 [ 0, %entry ], [ %indvar.next, %no_exit ]		; <i32> [#uses=1]
	volatile store float 0.000000e+00, float* %D
	%indvar.next = add i32 %indvar, 1		; <i32> [#uses=2]
	%exitcond = icmp eq i32 %indvar.next, %E		; <i1> [#uses=1]
	br i1 %exitcond, label %loopexit, label %no_exit
loopexit:		; preds = %no_exit
	ret void
}

