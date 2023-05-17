; RUN: %lli -jit-kind=mcjit %s > /dev/null
; RUN: %lli %s > /dev/null

@.LC0 = internal global [10 x i8] c"argc: %d\0A\00"		; <ptr> [#uses=1]

declare i32 @puts(ptr)

define void @getoptions(ptr %argc) {
bb0:
	ret void
}

declare i32 @printf(ptr, ...)

define i32 @main(i32 %argc, ptr %argv) {
bb0:
	call i32 (ptr, ...) @printf( ptr @.LC0, i32 %argc )		; <i32>:0 [#uses=0]
	%local = alloca ptr		; <ptr> [#uses=3]
	store ptr %argv, ptr %local
	%cond226 = icmp sle i32 %argc, 0		; <i1> [#uses=1]
	br i1 %cond226, label %bb3, label %bb2
bb2:		; preds = %bb2, %bb0
	%cann-indvar = phi i32 [ 0, %bb0 ], [ %add1-indvar, %bb2 ]		; <i32> [#uses=2]
	%add1-indvar = add i32 %cann-indvar, 1		; <i32> [#uses=2]
	%cann-indvar-idxcast = sext i32 %cann-indvar to i64		; <i64> [#uses=1]
	%reg115 = load ptr, ptr %local		; <ptr> [#uses=1]
	%cast235 = getelementptr ptr, ptr %reg115, i64 %cann-indvar-idxcast		; <ptr> [#uses=1]
	%reg117 = load ptr, ptr %cast235		; <ptr> [#uses=1]
	%reg236 = call i32 @puts( ptr %reg117 )		; <i32> [#uses=0]
	%cond239 = icmp slt i32 %add1-indvar, %argc		; <i1> [#uses=1]
	br i1 %cond239, label %bb2, label %bb3
bb3:		; preds = %bb2, %bb0
	call void @getoptions( ptr %local )
	ret i32 0
}
