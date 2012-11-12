; Check that variable strides are reduced to adds instead of multiplies.
; RUN: opt < %s -loop-reduce -S | not grep mul

declare i1 @pred(i32)

define void @test([10000 x i32]* %P, i32 %STRIDE) {
; <label>:0
	br label %Loop
Loop:		; preds = %Loop, %0
	%INDVAR = phi i32 [ 0, %0 ], [ %INDVAR2, %Loop ]		; <i32> [#uses=2]
	%Idx = mul i32 %INDVAR, %STRIDE		; <i32> [#uses=1]
	%cond = call i1 @pred( i32 %Idx )		; <i1> [#uses=1]
	%INDVAR2 = add i32 %INDVAR, 1		; <i32> [#uses=1]
	br i1 %cond, label %Loop, label %Out
Out:		; preds = %Loop
	ret void
}

