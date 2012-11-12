; Check that this test makes INDVAR and related stuff dead.
; RUN: opt < %s -loop-reduce -S | not grep INDVAR

declare i1 @pred()

define void @test(i32* %P) {
; <label>:0
	br label %Loop
Loop:		; preds = %Loop, %0
        %i = phi i32 [ 0, %0 ], [ %i.next, %Loop ]
	%INDVAR = phi i32 [ 0, %0 ], [ %INDVAR2, %Loop ]		; <i32> [#uses=2]
	%STRRED = getelementptr i32* %P, i32 %INDVAR		; <i32*> [#uses=1]
	store i32 0, i32* %STRRED
	%INDVAR2 = add i32 %INDVAR, 1		; <i32> [#uses=1]
        %i.next = add i32 %i, 1
	%cond = call i1 @pred( )		; <i1> [#uses=1]
	br i1 %cond, label %Loop, label %Out
Out:		; preds = %Loop
	ret void
}

