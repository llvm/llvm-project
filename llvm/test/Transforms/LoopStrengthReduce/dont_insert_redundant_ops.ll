; Check that this test makes INDVAR and related stuff dead.
; RUN: llvm-as < %s | opt -loop-reduce | llvm-dis | grep phi | count 2

declare i1 @pred()

define void @test1({ i32, i32 }* %P) {
; <label>:0
	br label %Loop
Loop:		; preds = %Loop, %0
	%INDVAR = phi i32 [ 0, %0 ], [ %INDVAR2, %Loop ]		; <i32> [#uses=3]
	%gep1 = getelementptr { i32, i32 }* %P, i32 %INDVAR, i32 0		; <i32*> [#uses=1]
	store i32 0, i32* %gep1
	%gep2 = getelementptr { i32, i32 }* %P, i32 %INDVAR, i32 1		; <i32*> [#uses=1]
	store i32 0, i32* %gep2
	%INDVAR2 = add i32 %INDVAR, 1		; <i32> [#uses=1]
	%cond = call i1 @pred( )		; <i1> [#uses=1]
	br i1 %cond, label %Loop, label %Out
Out:		; preds = %Loop
	ret void
}

define void @test2([2 x i32]* %P) {
; <label>:0
	br label %Loop
Loop:		; preds = %Loop, %0
	%INDVAR = phi i32 [ 0, %0 ], [ %INDVAR2, %Loop ]		; <i32> [#uses=3]
	%gep1 = getelementptr [2 x i32]* %P, i32 %INDVAR, i64 0		; <i32*> [#uses=1]
	store i32 0, i32* %gep1
	%gep2 = getelementptr [2 x i32]* %P, i32 %INDVAR, i64 1		; <i32*> [#uses=1]
	store i32 0, i32* %gep2
	%INDVAR2 = add i32 %INDVAR, 1		; <i32> [#uses=1]
	%cond = call i1 @pred( )		; <i1> [#uses=1]
	br i1 %cond, label %Loop, label %Out
Out:		; preds = %Loop
	ret void
}
