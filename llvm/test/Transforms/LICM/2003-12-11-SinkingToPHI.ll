; RUN: opt < %s -licm | lli %defaultjit

define i32 @main() {
entry:
	br label %Loop
Loop:		; preds = %LoopCont, %entry
	br i1 true, label %LoopCont, label %Out
LoopCont:		; preds = %Loop
	%X = add i32 1, 0		; <i32> [#uses=1]
	br i1 true, label %Out, label %Loop
Out:		; preds = %LoopCont, %Loop
	%V = phi i32 [ 2, %Loop ], [ %X, %LoopCont ]		; <i32> [#uses=1]
	%V2 = sub i32 %V, 1		; <i32> [#uses=1]
	ret i32 %V2
}

