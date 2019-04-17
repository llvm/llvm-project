; Check that the index of 'P[outer]' is pulled out of the loop.
; RUN: opt < %s -loop-reduce -S | \
; RUN:   not grep "getelementptr.*%outer.*%INDVAR"

target datalayout = "e-p:32:32:32-n32"
declare i1 @pred()

define void @test([10000 x i32]* %P, i32 %outer) {
; <label>:0
	br label %Loop
Loop:		; preds = %Loop, %0
	%INDVAR = phi i32 [ 0, %0 ], [ %INDVAR2, %Loop ]		; <i32> [#uses=2]
	%STRRED = getelementptr [10000 x i32], [10000 x i32]* %P, i32 %outer, i32 %INDVAR		; <i32*> [#uses=1]
	store i32 0, i32* %STRRED
	%INDVAR2 = add i32 %INDVAR, 1		; <i32> [#uses=1]
	%cond = call i1 @pred( )		; <i1> [#uses=1]
	br i1 %cond, label %Loop, label %Out
Out:		; preds = %Loop
	ret void
}

