; Test that LICM works when there is not a loop-preheader
; RUN: opt < %s -licm | llvm-dis

define void @testfunc(i32 %i.s, i1 %ifcond) {
	br i1 %ifcond, label %Then, label %Else
Then:		; preds = %0
	br label %Loop
Else:		; preds = %0
	br label %Loop
Loop:		; preds = %Loop, %Else, %Then
	%j = phi i32 [ 0, %Then ], [ 12, %Else ], [ %Next, %Loop ]		; <i32> [#uses=1]
	%i = bitcast i32 %i.s to i32		; <i32> [#uses=1]
	%i2 = mul i32 %i, 17		; <i32> [#uses=1]
	%Next = add i32 %j, %i2		; <i32> [#uses=2]
	%cond = icmp eq i32 %Next, 0		; <i1> [#uses=1]
	br i1 %cond, label %Out, label %Loop
Out:		; preds = %Loop
	ret void
}

