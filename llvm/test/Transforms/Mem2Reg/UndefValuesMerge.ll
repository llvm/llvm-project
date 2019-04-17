; RUN: opt < %s -mem2reg -S | not grep phi

define i32 @testfunc(i1 %C, i32 %i, i8 %j) {
	%I = alloca i32		; <i32*> [#uses=2]
	br i1 %C, label %T, label %Cont
T:		; preds = %0
	store i32 %i, i32* %I
	br label %Cont
Cont:		; preds = %T, %0
	%Y = load i32, i32* %I		; <i32> [#uses=1]
	ret i32 %Y
}

