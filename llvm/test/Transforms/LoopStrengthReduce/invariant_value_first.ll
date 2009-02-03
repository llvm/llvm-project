; Check that the index of 'P[outer]' is pulled out of the loop.
; RUN: llvm-as < %s | opt -loop-reduce | llvm-dis | \
; RUN:   not grep {getelementptr.*%outer.*%INDVAR}

declare i1 @pred()

declare i32 @foo()

define void @test([10000 x i32]* %P) {
; <label>:0
	%outer = call i32 @foo( )		; <i32> [#uses=1]
	br label %Loop
Loop:		; preds = %Loop, %0
	%INDVAR = phi i32 [ 0, %0 ], [ %INDVAR2, %Loop ]		; <i32> [#uses=2]
	%STRRED = getelementptr [10000 x i32]* %P, i32 %outer, i32 %INDVAR		; <i32*> [#uses=1]
	store i32 0, i32* %STRRED
	%INDVAR2 = add i32 %INDVAR, 1		; <i32> [#uses=1]
	%cond = call i1 @pred( )		; <i1> [#uses=1]
	br i1 %cond, label %Loop, label %Out
Out:		; preds = %Loop
	ret void
}

