; RUN: %lli -mtriple=%mcjit_triple -use-mcjit %s > /dev/null

define i32 @main() {
; <label>:0
	br label %Loop
Loop:		; preds = %Loop, %0
	%X = phi i32 [ 0, %0 ], [ 1, %Loop ]		; <i32> [#uses=1]
	br i1 true, label %Out, label %Loop
Out:		; preds = %Loop
	ret i32 %X
}

