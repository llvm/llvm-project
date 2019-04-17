; Check that this test makes INDVAR and related stuff dead, because P[indvar]
; gets reduced, making INDVAR dead.

; RUN: opt < %s -loop-reduce -S | not grep INDVAR

target datalayout = "e-p:32:32:32-n32"

declare i1 @pred()

declare i32 @getidx()

define void @test([10000 x i32]* %P) {
; <label>:0
	br label %Loop
Loop:		; preds = %Loop, %0
	%INDVAR = phi i32 [ 0, %0 ], [ %INDVAR2, %Loop ]		; <i32> [#uses=2]
	%idx = call i32 @getidx( )		; <i32> [#uses=1]
	%STRRED = getelementptr [10000 x i32], [10000 x i32]* %P, i32 %INDVAR, i32 %idx		; <i32*> [#uses=1]
	store i32 0, i32* %STRRED
	%INDVAR2 = add i32 %INDVAR, 1		; <i32> [#uses=1]
	%cond = call i1 @pred( )		; <i1> [#uses=1]
	br i1 %cond, label %Loop, label %Out
Out:		; preds = %Loop
	ret void
}

