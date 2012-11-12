; RUN: %lli -mtriple=%mcjit_triple -use-mcjit %s > /dev/null

define i32 @main() {
; <label>:0
	br label %Loop
Loop:		; preds = %Loop, %0
	%I = phi i32 [ 0, %0 ], [ %i2, %Loop ]		; <i32> [#uses=1]
	%i2 = add i32 %I, 1		; <i32> [#uses=2]
	%C = icmp eq i32 %i2, 10		; <i1> [#uses=1]
	br i1 %C, label %Out, label %Loop
Out:		; preds = %Loop
	ret i32 0
}

