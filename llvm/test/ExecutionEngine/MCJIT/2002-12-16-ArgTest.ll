; RUN: %lli -jit-kind=mcjit %s > /dev/null
; RUN: %lli %s > /dev/null

@.LC0 = internal global [10 x i8] c"argc: %d\0A\00"		; <[10 x i8]*> [#uses=1]

declare i32 @puts(i8*)

define void @getoptions(i32* %argc) {
bb0:
	ret void
}

declare i32 @printf(i8*, ...)

define i32 @main(i32 %argc, i8** %argv) {
bb0:
	call i32 (i8*, ...) @printf( i8* getelementptr ([10 x i8], [10 x i8]* @.LC0, i64 0, i64 0), i32 %argc )		; <i32>:0 [#uses=0]
	%cast224 = bitcast i8** %argv to i8*		; <i8*> [#uses=1]
	%local = alloca i8*		; <i8**> [#uses=3]
	store i8* %cast224, i8** %local
	%cond226 = icmp sle i32 %argc, 0		; <i1> [#uses=1]
	br i1 %cond226, label %bb3, label %bb2
bb2:		; preds = %bb2, %bb0
	%cann-indvar = phi i32 [ 0, %bb0 ], [ %add1-indvar, %bb2 ]		; <i32> [#uses=2]
	%add1-indvar = add i32 %cann-indvar, 1		; <i32> [#uses=2]
	%cann-indvar-idxcast = sext i32 %cann-indvar to i64		; <i64> [#uses=1]
	%CT = bitcast i8** %local to i8***		; <i8***> [#uses=1]
	%reg115 = load i8**, i8*** %CT		; <i8**> [#uses=1]
	%cast235 = getelementptr i8*, i8** %reg115, i64 %cann-indvar-idxcast		; <i8**> [#uses=1]
	%reg117 = load i8*, i8** %cast235		; <i8*> [#uses=1]
	%reg236 = call i32 @puts( i8* %reg117 )		; <i32> [#uses=0]
	%cond239 = icmp slt i32 %add1-indvar, %argc		; <i1> [#uses=1]
	br i1 %cond239, label %bb2, label %bb3
bb3:		; preds = %bb2, %bb0
	%cast243 = bitcast i8** %local to i32*		; <i32*> [#uses=1]
	call void @getoptions( i32* %cast243 )
	ret i32 0
}
