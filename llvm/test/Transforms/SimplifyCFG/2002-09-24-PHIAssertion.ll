; RUN: llvm-as < %s | opt -simplifycfg

define i32 @test(i32 %A, i32 %B, i1 %cond) {
J:
	%C = add i32 %A, 12		; <i32> [#uses=3]
	br i1 %cond, label %L, label %L
L:		; preds = %J, %J
	%Q = phi i32 [ %C, %J ], [ %C, %J ]		; <i32> [#uses=1]
	%D = add i32 %C, %B		; <i32> [#uses=1]
	%E = add i32 %Q, %D		; <i32> [#uses=1]
	ret i32 %E
}

